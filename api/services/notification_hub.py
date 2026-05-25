"""Process-wide fan-out of JSON notifications to browser WebSocket subscribers.

Notifications are **in-memory per API process**. Multiple uvicorn workers or replicas each
maintain separate subscriber sets; use a single worker for consistent delivery in dev, or
add a cross-process broker (for example Redis pub/sub) if you need multi-replica fan-out.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from starlette.websockets import WebSocket

_log = logging.getLogger(__name__)

NotificationLevel = Literal["error", "warning", "info"]

_MAX_MESSAGE_LEN = 500
_MAX_CONTEXT_JSON_KEYS = 16

_hub: NotificationHub | None = None
_hub_lock = asyncio.Lock()


class NotificationHub:
    """Register Starlette :class:`~starlette.websockets.WebSocket` clients and broadcast JSON."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def attach(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def detach(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("notification ws send failed, dropping client: %s", exc)
                dead.append(ws)
        for ws in dead:
            await self.detach(ws)


async def get_notification_hub() -> NotificationHub:
    """Singleton hub for the current process."""
    global _hub  # noqa: PLW0603
    async with _hub_lock:
        if _hub is None:
            _hub = NotificationHub()
        return _hub


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _truncate_message(text: str) -> str:
    t = text.strip()
    if len(t) <= _MAX_MESSAGE_LEN:
        return t
    return t[: _MAX_MESSAGE_LEN - 1] + "…"


def _sanitize_context(ctx: dict[str, Any] | None) -> dict[str, Any] | None:
    if not ctx:
        return None
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(ctx.items()):
        if i >= _MAX_CONTEXT_JSON_KEYS:
            break
        if not isinstance(k, str) or len(k) > 64:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v if not isinstance(v, str) or len(v) <= 256 else v[:255] + "…"
        elif isinstance(v, list) and len(v) <= 20:
            out[k] = [
                x
                for x in v[:20]
                if isinstance(x, (str, int, float, bool)) or x is None
            ]
    return out or None


async def publish_notification(
    *,
    level: NotificationLevel,
    message: str,
    code: str | None = None,
    title: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Build a v1 ``notification`` frame and fan out to all subscribers."""
    body: dict[str, Any] = {
        "type": "notification",
        "schema": 1,
        "id": str(uuid.uuid4()),
        "ts": _utc_now_iso(),
        "level": level,
        "message": _truncate_message(message),
    }
    if code:
        body["code"] = str(code)[:128]
    if title:
        body["title"] = _truncate_message(title)
    safe_ctx = _sanitize_context(context)
    if safe_ctx is not None:
        body["context"] = safe_ctx
    hub = await get_notification_hub()
    await hub.publish(body)
