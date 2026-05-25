"""Process-wide market clock fan-out: WebSocket clients and in-process subscribers.

The tick loop (started from :func:`api.main.lifespan`) builds a
:class:`~shunya.time.market_clock.MarketClockSnapshot` on a fixed interval and
broadcasts it to every attached browser and every registered internal queue.

Internal API usage::

    q = await create_market_clock_subscription()
    try:
        snap = await asyncio.wait_for(q.get(), timeout=5.0)
    finally:
        await release_market_clock_subscription(q)

Multi-worker / multi-replica: each process has its own hub and loop, same caveats as
:mod:`api.services.notification_hub`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from starlette.websockets import WebSocket

from shunya.time.market_clock import MarketClockSnapshot, build_market_clock_snapshot

_log = logging.getLogger(__name__)

_hub: MarketClockHub | None = None
_hub_lock = asyncio.Lock()


def _tick_interval_seconds() -> float:
    raw = os.environ.get("SHUNYA_MARKET_CLOCK_TICK_SECONDS", "1").strip()
    try:
        interval = float(raw)
    except ValueError:
        interval = 1.0
    return max(0.25, min(60.0, interval))


def snapshot_to_tick_payload(snap: MarketClockSnapshot) -> dict[str, Any]:
    """JSON frame for WebSocket ``tick`` messages (schema v1)."""
    return {
        "type": "tick",
        "schema": 1,
        "utc_iso": snap.utc_iso,
        "us_line": snap.us_line,
        "in_line": snap.in_line,
        "us_listed_rth_open": snap.us_listed_rth_open,
        "alpaca_l1_us_equities_stream_allowed": snap.alpaca_l1_us_equities_stream_allowed,
    }


def _enqueue_drop_oldest(q: asyncio.Queue[MarketClockSnapshot], snap: MarketClockSnapshot) -> None:
    try:
        q.put_nowait(snap)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(snap)
        except asyncio.QueueFull:
            pass


class MarketClockHub:
    """Attach Starlette WebSockets and optional :class:`asyncio.Queue` subscribers."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._queues: set[asyncio.Queue[MarketClockSnapshot]] = set()
        self._lock = asyncio.Lock()

    async def attach_websocket(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def detach_websocket(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def register_queue(self, queue: asyncio.Queue[MarketClockSnapshot]) -> None:
        async with self._lock:
            self._queues.add(queue)

    async def unregister_queue(self, queue: asyncio.Queue[MarketClockSnapshot]) -> None:
        async with self._lock:
            self._queues.discard(queue)

    async def broadcast(self, snap: MarketClockSnapshot) -> None:
        payload = snapshot_to_tick_payload(snap)
        async with self._lock:
            clients = list(self._clients)
            queues = list(self._queues)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("market clock ws send failed, dropping client: %s", exc)
                dead.append(ws)
        for ws in dead:
            await self.detach_websocket(ws)

        for q in queues:
            _enqueue_drop_oldest(q, snap)


async def get_market_clock_hub() -> MarketClockHub:
    """Singleton hub for the current process."""
    global _hub  # noqa: PLW0603
    async with _hub_lock:
        if _hub is None:
            _hub = MarketClockHub()
        return _hub


async def create_market_clock_subscription(*, maxsize: int = 4) -> asyncio.Queue[MarketClockSnapshot]:
    """Register an in-process subscriber queue; must pair with :func:`release_market_clock_subscription`."""
    hub = await get_market_clock_hub()
    q: asyncio.Queue[MarketClockSnapshot] = asyncio.Queue(maxsize=maxsize)
    await hub.register_queue(q)
    return q


async def release_market_clock_subscription(queue: asyncio.Queue[MarketClockSnapshot]) -> None:
    hub = await get_market_clock_hub()
    await hub.unregister_queue(queue)


async def run_market_clock_loop(stop: asyncio.Event) -> None:
    """Broadcast ``build_market_clock_snapshot()`` until ``stop`` is set."""
    interval = _tick_interval_seconds()
    hub = await get_market_clock_hub()
    while True:
        if stop.is_set():
            return
        snap = build_market_clock_snapshot()
        try:
            await hub.broadcast(snap)
        except Exception as exc:  # noqa: BLE001
            _log.debug("market clock broadcast failed: %s", exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            if stop.is_set():
                return
        except asyncio.TimeoutError:
            continue
