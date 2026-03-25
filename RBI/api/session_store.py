from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class RBISession:
    id: str
    viewer_token: str
    container_name: str
    start_url: str
    allowed_hosts: list[str]
    created_at: float
    expires_at: float
    internal_secret: str = field(repr=False)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, RBISession] = {}

    def create(
        self,
        *,
        start_url: str,
        allowed_hosts: list[str],
        ttl_seconds: int,
        container_name: str,
        internal_secret: str,
        session_id: str | None = None,
        viewer_token: str | None = None,
    ) -> RBISession:
        sid = session_id or uuid.uuid4().hex
        viewer = viewer_token or uuid.uuid4().hex
        now = time.time()
        sess = RBISession(
            id=sid,
            viewer_token=viewer,
            container_name=container_name,
            start_url=start_url,
            allowed_hosts=list(allowed_hosts),
            created_at=now,
            expires_at=now + max(30, ttl_seconds),
            internal_secret=internal_secret,
        )
        self._sessions[sid] = sess
        return sess

    def get(self, session_id: str) -> RBISession | None:
        return self._sessions.get(session_id)

    def list(self) -> list[RBISession]:
        return list(self._sessions.values())

    def delete(self, session_id: str) -> RBISession | None:
        return self._sessions.pop(session_id, None)


store = SessionStore()
