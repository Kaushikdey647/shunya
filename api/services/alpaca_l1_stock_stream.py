"""Alpaca live L1 :class:`~alpaca.data.live.stock.StockDataStream` with safer ``_run_forever`` behavior.

The upstream ``alpaca-py`` loop treats only ``insufficient subscription`` as a fatal
:class:`ValueError`. Other auth-related failures such as **connection limit exceeded**
are logged and retried in a tight loop, which opens many TCP connections and makes the
problem worse. This subclass exits cleanly (after :meth:`~alpaca.data.live.websocket.DataStream.close`)
so the API process can back off and the hub can reset shared state.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

import websockets
from alpaca.data.live.stock import StockDataStream

_log = logging.getLogger(__name__)

# Hub registers this to surface Alpaca ``T: error`` / ``T: subscription`` to browser clients.
ShunyaControlHandler = Callable[[dict[str, Any]], Awaitable[None]]


def _fatal_market_data_value_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc).lower()
    if "insufficient subscription" in msg:
        return True
    if "connection limit exceeded" in msg:
        return True
    if "connection limit" in msg and "exceeded" in msg:
        return True
    return False


class ShunyaL1StockDataStream(StockDataStream):
    """Same constructor as :class:`~alpaca.data.live.stock.StockDataStream`; stricter fatal errors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shunya_control_handler: ShunyaControlHandler | None = None

    def set_shunya_control_handler(self, handler: ShunyaControlHandler | None) -> None:
        """Notify hub on Alpaca control messages (``error``, ``subscription``) before default dispatch."""
        self._shunya_control_handler = handler

    async def _dispatch(self, msg: dict[str, Any]) -> None:  # noqa: SLF001
        handler = self._shunya_control_handler
        t = msg.get("T")
        if handler is not None and t in ("error", "subscription"):
            try:
                await handler(msg)
            except Exception:  # noqa: BLE001
                _log.exception("Alpaca L1 control handler failed for T=%s", t)
        await super()._dispatch(msg)

    async def _run_forever(self) -> None:  # noqa: SLF001 — intentional; same contract as alpaca-py
        self._loop = asyncio.get_running_loop()
        while not any(
            v
            for k, v in self._handlers.items()
            if k not in ("cancelErrors", "corrections")
        ):
            if not self._stop_stream_queue.empty():
                self._stop_stream_queue.get(timeout=1)
                return
            await asyncio.sleep(0)
        _log.info("started %s stream", self._name)
        self._should_run = True
        self._running = False
        while True:
            try:
                if not self._should_run:
                    _log.info("%s stream stopped", self._name)
                    return
                if not self._running:
                    _log.info("starting %s websocket connection", self._name)
                    await self._start_ws()
                    await self._send_subscribe_msg()
                    self._running = True
                    _log.info(
                        "Alpaca L1 market-data WebSocket up endpoint=%s",
                        getattr(self, "_endpoint", "?"),
                    )
                await self._consume()
            except websockets.WebSocketException as wse:
                await self.close()
                self._running = False
                _log.warning("data websocket error, restarting connection: %s", wse)
            except ValueError as ve:
                if _fatal_market_data_value_error(ve):
                    await self.close()
                    self._running = False
                    _log.error("fatal Alpaca market-data stream error: %s", ve)
                    return
                _log.exception("error during websocket communication: %s", ve)
            except Exception as e:
                _log.exception("error during websocket communication: %s", e)
            finally:
                await asyncio.sleep(0)
