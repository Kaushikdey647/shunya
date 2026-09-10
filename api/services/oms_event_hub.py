"""Broadcast hub for OMS order/fill events: OMS callbacks (any thread) -> browser WebSockets.

Publishing crosses from the Alpaca trade-update asyncio callback (sync call inside
:class:`~shunya.oms.service.InstitutionalOMS` overrides) into the API's event loop.
Use :meth:`OrderEventHub.publish_threadsafe` from any thread; never call the async
:meth:`publish` directly from a non-event-loop thread (no ``run_coroutine_threadsafe``
+ ``.result()`` here — that pattern can deadlock if called from the loop's own thread).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from starlette.websockets import WebSocket

_log = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 1000


class OrderEventHub:
    """Process-wide fan-out of OMS event payloads to attached browser WebSockets."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._queue: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_task: asyncio.Task[None] | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the running loop and start the drain task (call once at lifespan startup)."""
        self._loop = loop
        self._drain_task = loop.create_task(self._drain(), name="oms-event-hub-drain")

    async def stop(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

    async def attach(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def detach(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    def publish_threadsafe(self, payload: dict[str, Any]) -> None:
        """Enqueue ``payload`` for broadcast; safe to call from any thread."""
        if self._loop is None:
            return

        def _put() -> None:
            try:
                self._queue.put_nowait(payload)
            except asyncio.QueueFull:
                _log.warning("OMS event hub queue full; dropping oldest event")
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(payload)
                except asyncio.QueueEmpty:
                    pass

        self._loop.call_soon_threadsafe(_put)

    async def _drain(self) -> None:
        while True:
            payload = await self._queue.get()
            await self.publish(payload)

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("OMS event send failed, detaching client: %s", exc)
                await self.detach(ws)
