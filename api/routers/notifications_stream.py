"""WebSocket: server-pushed notifications and errors (in-process fan-out).

See :mod:`api.services.notification_hub` for semantics and multi-worker caveats.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from api.services.notification_hub import get_notification_hub

_log = logging.getLogger(__name__)

router = APIRouter(tags=["notifications"])


async def _wait_websocket_disconnect(websocket: WebSocket) -> None:
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


@router.websocket("/notifications/stream")
async def notifications_stream_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    hub = await get_notification_hub()
    try:
        await hub.attach(websocket)
    except Exception as exc:  # noqa: BLE001
        _log.warning("notification stream attach failed: %s", exc)
        await websocket.close(code=1011)
        return

    try:
        await websocket.send_json({"type": "hello", "schema": 1})
        await _wait_websocket_disconnect(websocket)
    finally:
        await hub.detach(websocket)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
