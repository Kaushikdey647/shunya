"""Process-wide multiplexing of browser WebSockets onto one Alpaca :class:`~alpaca.data.live.stock.StockDataStream` per API key + feed."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from alpaca.data.enums import DataFeed

from api.services.alpaca_l1_payload import (
    alpaca_quote_to_dict,
    alpaca_trade_cancel_to_dict,
    alpaca_trade_correction_to_dict,
    alpaca_trade_to_dict,
)
from shunya.integration.alpaca_settings import AlpacaRuntimeSettings, build_stock_data_stream

from starlette.websockets import WebSocket

if TYPE_CHECKING:
    from alpaca.data.live.stock import StockDataStream

_log = logging.getLogger(__name__)

_hubs: dict[tuple[str, bool, str], "AlpacaL1FeedHub"] = {}
_hubs_lock = threading.Lock()


def _max_symbols_from_env() -> int:
    raw = os.environ.get("SHUNYA_ALPACA_L1_MAX_SYMBOLS", "30").strip()
    try:
        n = int(raw)
    except ValueError:
        return 30
    return max(1, min(n, 500))


class SymbolLimitExceeded(Exception):
    """Raised when attaching a new symbol would exceed ``SHUNYA_ALPACA_L1_MAX_SYMBOLS``."""

    def __init__(self, max_symbols: int) -> None:
        super().__init__(f"symbol limit ({max_symbols}) reached for shared Alpaca L1 stream")
        self.max_symbols = max_symbols


def get_alpaca_l1_hub(rt: AlpacaRuntimeSettings, *, feed: DataFeed = DataFeed.IEX) -> AlpacaL1FeedHub:
    """Return the singleton hub for this key + feed (one ``StockDataStream`` per process)."""
    key = (rt.api_key_id, rt.paper, feed.value)
    with _hubs_lock:
        if key not in _hubs:
            _hubs[key] = AlpacaL1FeedHub(rt, feed=feed, max_symbols=_max_symbols_from_env())
        return _hubs[key]


def _symbol_upper(obj: Any) -> str:
    if isinstance(obj, dict):
        s = obj.get("S") or obj.get("symbol")
        return str(s).strip().upper() if isinstance(s, str) else ""
    s = getattr(obj, "symbol", None)
    return str(s).strip().upper() if isinstance(s, str) else ""


class AlpacaL1FeedHub:
    """
    One Alpaca market-data WebSocket (``StockDataStream``) shared by all browser sessions.

    Alpaca-py allows only **one handler per symbol per channel**; this hub installs a single
    pair of coroutines and fans out to all ``WebSocket`` clients subscribed to each symbol.
    """

    def __init__(self, rt: AlpacaRuntimeSettings, *, feed: DataFeed, max_symbols: int) -> None:
        self._rt = rt
        self._feed = feed
        self._max_symbols = max_symbols
        self._lock = asyncio.Lock()
        self._ref: dict[str, int] = {}
        self._clients: dict[str, set[WebSocket]] = {}
        self._stream: "StockDataStream | None" = None
        self._runner_task: asyncio.Task[None] | None = None
        self._corrections_registered = False

    @property
    def max_symbols(self) -> int:
        return self._max_symbols

    async def would_reject_new_symbol(self, sym: str) -> bool:
        """True if ``sym`` is not yet tracked and distinct-symbol cap is already reached."""
        async with self._lock:
            if sym in self._ref:
                return False
            return len(self._ref) >= self._max_symbols

    async def attach(self, websocket: WebSocket, sym: str) -> None:
        """Subscribe ``websocket`` to Alpaca L1 for ``sym`` (refcounted)."""
        async with self._lock:
            if sym in self._ref:
                self._ref[sym] += 1
                self._clients.setdefault(sym, set()).add(websocket)
                return

            if len(self._ref) >= self._max_symbols:
                raise SymbolLimitExceeded(self._max_symbols)

            self._ref[sym] = 1
            self._clients.setdefault(sym, set()).add(websocket)

            if self._stream is None:
                self._stream = build_stock_data_stream(self._rt, feed=self._feed)
                self._stream.subscribe_quotes(self._on_quote, sym)
                self._stream.subscribe_trades(self._on_trade, sym)
                if not self._corrections_registered:
                    self._stream.register_trade_corrections(self._on_trade_correction)
                    self._stream.register_trade_cancels(self._on_trade_cancel)
                    self._corrections_registered = True
                self._runner_task = asyncio.create_task(
                    self._stream._run_forever(),  # noqa: SLF001
                    name="alpaca-l1-shared-run-forever",
                )
            else:
                self._stream.subscribe_quotes(self._on_quote, sym)
                self._stream.subscribe_trades(self._on_trade, sym)

    async def detach(self, websocket: WebSocket, sym: str) -> None:
        """Unsubscribe ``websocket`` from ``sym``; may tear down the Alpaca stream when unused."""
        async with self._lock:
            if sym not in self._ref:
                return
            clients = self._clients.get(sym)
            if not clients or websocket not in clients:
                return
            clients.discard(websocket)
            self._ref[sym] -= 1
            if self._ref[sym] > 0:
                return
            del self._ref[sym]
            del self._clients[sym]

            stream = self._stream
            if stream is None:
                return

            try:
                stream.unsubscribe_quotes(sym)
                stream.unsubscribe_trades(sym)
            except Exception as exc:  # noqa: BLE001
                _log.debug("unsubscribe %s: %s", sym, exc)

            if not self._ref:
                await self._shutdown_stream_locked()

    async def _shutdown_stream_locked(self) -> None:
        """Stop Alpaca stream; caller must hold ``self._lock``."""
        stream = self._stream
        self._stream = None
        self._corrections_registered = False
        task = self._runner_task
        self._runner_task = None

        if stream is not None:
            try:
                await stream.stop_ws()
            except Exception as exc:  # noqa: BLE001
                _log.debug("stop_ws: %s", exc)

        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _on_quote(self, q: Any) -> None:
        sym = _symbol_upper(q)
        if not sym:
            return
        try:
            body = alpaca_quote_to_dict(q)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skip malformed quote: %s", exc)
            return
        payload = {"type": "quote", **body}
        await self._fanout_json(sym, payload)

    async def _on_trade(self, t: Any) -> None:
        sym = _symbol_upper(t)
        if not sym:
            return
        try:
            body = alpaca_trade_to_dict(t)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skip malformed trade: %s", exc)
            return
        payload = {"type": "trade", **body}
        await self._fanout_json(sym, payload)

    async def _on_trade_correction(self, c: Any) -> None:
        sym = _symbol_upper(c)
        if not sym:
            return
        try:
            body = alpaca_trade_correction_to_dict(c)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skip malformed correction: %s", exc)
            return
        payload = {"type": "trade_correction", **body}
        await self._fanout_json(sym, payload)

    async def _on_trade_cancel(self, c: Any) -> None:
        sym = _symbol_upper(c)
        if not sym:
            return
        try:
            body = alpaca_trade_cancel_to_dict(c)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skip malformed cancel: %s", exc)
            return
        payload = {"type": "trade_cancel", **body}
        await self._fanout_json(sym, payload)

    async def _fanout_json(self, sym: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients.get(sym, ()))
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("ws send failed, dropping client for %s: %s", sym, exc)
                await self.detach(ws, sym)
