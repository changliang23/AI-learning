from __future__ import annotations

import uuid

import docker

from config import settings


def _client() -> docker.DockerClient:
    return docker.from_env()


def build_container_name(session_id: str) -> str:
    short = session_id[:12]
    return f"rbi-sess-{short}"


def start_session_container(
    *,
    session_id: str,
    start_url: str,
    ttl_seconds: int,
    internal_secret: str,
) -> str:
    client = _client()
    name = build_container_name(session_id)
    env = {
        "START_URL": start_url,
        "SESSION_TTL_SEC": str(ttl_seconds),
        "RBI_INTERNAL_SECRET": internal_secret,
    }
    client.containers.run(
        image=settings.session_image,
        name=name,
        detach=True,
        remove=True,
        network=settings.docker_network,
        environment=env,
        labels={"rbi.session": session_id, "rbi.project": "demo"},
    )
    return name


def stop_container(name: str) -> None:
    try:
        c = _client().containers.get(name)
        c.stop(timeout=10)
    except docker.errors.NotFound:
        pass
    except Exception:
        pass


def new_internal_secret() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex
