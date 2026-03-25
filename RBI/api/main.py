from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
import websockets
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from docker_runner import build_container_name, new_internal_secret, start_session_container, stop_container
from session_store import store

APP_START = time.time()


def _host_allowed(url: str, allowed_hosts: list[str]) -> bool:
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        if not host:
            return False
        for h in allowed_hosts:
            h = h.lower().strip().strip(".")
            if host == h or host.endswith("." + h):
                return True
        return False
    except Exception:
        return False


async def _cleanup_expired() -> None:
    while True:
        await asyncio.sleep(15)
        now = time.time()
        for s in list(store.list()):
            if s.expires_at <= now:
                stop_container(s.container_name)
                store.delete(s.id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_expired())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="RBI Demo API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_admin(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != settings.admin_token:
        raise HTTPException(status_code=403, detail="invalid admin token")


class CreateSessionBody(BaseModel):
    start_url: str
    allowed_hosts: list[str] = Field(default_factory=lambda: ["example.com"])
    ttl_seconds: int = Field(900, ge=30, le=86400)


class SessionCreatedResponse(BaseModel):
    session_id: str
    viewer_token: str
    expires_at: float
    container_name: str
    viewer_path: str


async def _wait_for_session_http(container_name: str, timeout_sec: float = 90.0) -> None:
    url = f"http://{container_name}:8080/health"
    deadline = time.time() + timeout_sec
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.5)
    raise HTTPException(status_code=504, detail=f"session container not healthy: {container_name}")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "uptime_sec": round(time.time() - APP_START, 3)}


@app.post("/admin/sessions", dependencies=[Depends(require_admin)])
async def admin_create_session(body: CreateSessionBody) -> SessionCreatedResponse:
    if not _host_allowed(body.start_url, body.allowed_hosts):
        raise HTTPException(status_code=400, detail="start_url host not in allowed_hosts")

    internal_secret = new_internal_secret()

    session_id = uuid.uuid4().hex
    viewer_token = uuid.uuid4().hex
    container_name = build_container_name(session_id)

    try:
        start_session_container(
            session_id=session_id,
            start_url=body.start_url,
            ttl_seconds=body.ttl_seconds,
            internal_secret=internal_secret,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"docker error: {e}") from e

    sess = store.create(
        start_url=body.start_url,
        allowed_hosts=body.allowed_hosts,
        ttl_seconds=body.ttl_seconds,
        container_name=container_name,
        internal_secret=internal_secret,
        session_id=session_id,
        viewer_token=viewer_token,
    )

    await _wait_for_session_http(container_name)

    return SessionCreatedResponse(
        session_id=sess.id,
        viewer_token=sess.viewer_token,
        expires_at=sess.expires_at,
        container_name=sess.container_name,
        viewer_path=f"/#/view?s={sess.id}&t={sess.viewer_token}",
    )


@app.get("/admin/sessions", dependencies=[Depends(require_admin)])
def admin_list_sessions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in store.list():
        out.append(
            {
                "session_id": s.id,
                "container_name": s.container_name,
                "start_url": s.start_url,
                "allowed_hosts": s.allowed_hosts,
                "created_at": s.created_at,
                "expires_at": s.expires_at,
            }
        )
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


@app.delete("/admin/sessions/{session_id}", dependencies=[Depends(require_admin)])
def admin_delete_session(session_id: str) -> dict[str, Any]:
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="not found")
    stop_container(s.container_name)
    store.delete(session_id)
    return {"ok": True}


class NavigateBody(BaseModel):
    url: str


@app.post("/api/sessions/{session_id}/navigate")
async def api_navigate(
    session_id: str,
    body: NavigateBody,
    token: str = Query(...),
) -> dict[str, Any]:
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="not found")
    if token != s.viewer_token:
        raise HTTPException(status_code=403, detail="bad token")
    if time.time() > s.expires_at:
        raise HTTPException(status_code=410, detail="expired")
    if not _host_allowed(body.url, s.allowed_hosts):
        raise HTTPException(status_code=400, detail="url host not allowed")

    url = f"http://{s.container_name}:8080/navigate"
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json={"url": body.url},
            headers={"X-RBI-Secret": s.internal_secret},
            timeout=30.0,
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text)
    return {"ok": True}


class InputBody(BaseModel):
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/sessions/{session_id}/input")
async def api_input(
    session_id: str,
    body: InputBody,
    token: str = Query(...),
) -> dict[str, Any]:
    s = store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="not found")
    if token != s.viewer_token:
        raise HTTPException(status_code=403, detail="bad token")
    if time.time() > s.expires_at:
        raise HTTPException(status_code=410, detail="expired")

    url = f"http://{s.container_name}:8080/input"
    async with httpx.AsyncClient() as client:
        r = await client.post(
            url,
            json=body.model_dump(),
            headers={"X-RBI-Secret": s.internal_secret},
            timeout=10.0,
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text)
    return {"ok": True}


@app.websocket("/ws/session/{session_id}/stream")
async def ws_stream(
    ws: WebSocket,
    session_id: str,
    token: str = Query(...),
) -> None:
    s = store.get(session_id)
    if not s:
        await ws.close(code=4004)
        return
    if token != s.viewer_token:
        await ws.close(code=4003)
        return
    if time.time() > s.expires_at:
        await ws.close(code=4005)
        return

    await ws.accept()
    backend_url = f"ws://{s.container_name}:8080/stream"
    try:
        async with websockets.connect(backend_url, max_size=None) as backend:
            async for data in backend:
                if isinstance(data, (bytes, bytearray)):
                    await ws.send_bytes(bytes(data))
                else:
                    await ws.send_text(str(data))
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close(code=1011)
        except Exception:
            pass
