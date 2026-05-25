"""Process-wide multiplexing of browser WebSockets onto one Alpaca :class:`~alpaca.data.live.stock.StockDataStream` per API key + feed."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from alpaca.data.enums import DataFeed

from api.services.alpaca_l1_payload import (
    alpaca_quote_to_dict,
    alpaca_trade_cancel_to_dict,
    alpaca_trade_correction_to_dict,
    alpaca_trade_to_dict,
)
from api.services.alpaca_l1_stock_stream import ShunyaL1StockDataStream
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


def _reconnect_cooldown_sec() -> float:
    """Seconds to wait before opening a new Alpaca TCP session after teardown (Alpaca-side limit)."""
    raw = os.environ.get("SHUNYA_ALPACA_L1_RECONNECT_COOLDOWN_SEC", "5").strip()
    try:
        return max(0.0, min(float(raw), 120.0))
    except ValueError:
        return 5.0


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


def _sync_subscribe_quotes_trades(
    stream: Any,
    on_quote: Any,
    on_trade: Any,
    sym: str,
) -> None:
    """Synchronous Alpaca subscribe (must run in a worker thread when ``stream._running``)."""
    stream.subscribe_quotes(on_quote, sym)
    stream.subscribe_trades(on_trade, sym)


def _sync_unsubscribe_quotes_trades(stream: Any, sym: str) -> None:
    """Synchronous Alpaca unsubscribe (must run in a worker thread when ``stream._running``)."""
    stream.unsubscribe_quotes(sym)
    stream.unsubscribe_trades(sym)


def _symbol_upper(obj: Any) -> str:
    if isinstance(obj, dict):
        s = obj.get("S") or obj.get("symbol")
        return str(s).strip().upper() if isinstance(s, str) else ""
    s = getattr(obj, "symbol", None)
    return str(s).strip().upper() if isinstance(s, str) else ""


def _clear_shunya_stream_control_handler(stream: Any) -> None:
    if stream is not None and hasattr(stream, "set_shunya_control_handler"):
        stream.set_shunya_control_handler(None)


def _json_safe_alpaca_obj(obj: Any) -> Any:
    """Make Alpaca msgpack-derived structures JSON-serializable; strip obvious secret keys."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = k.decode("utf-8", "replace") if isinstance(k, bytes) else str(k)
            if ks.lower() in ("key", "secret", "password", "token"):
                continue
            out[ks] = _json_safe_alpaca_obj(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe_alpaca_obj(x) for x in obj]
    return str(obj)


class AlpacaL1FeedHub:
    """
    One Alpaca market-data WebSocket (``StockDataStream``) shared by all browser sessions.

    Alpaca-py allows only **one handler per symbol per channel**; this hub installs a single
    pair of coroutines and fans out to all ``WebSocket`` clients subscribed to each symbol.

    **Event loop:** ``StockDataStream._subscribe`` / ``_unsubscribe`` call
    ``asyncio.run_coroutine_threadsafe(...).result()`` when the stream is already running.
    Invoking that from the FastAPI/uvicorn **event-loop thread** deadlocks the process (the
    loop cannot run the scheduled coroutine while blocked in ``.result()``). When
    ``stream._running`` is true, this hub runs subscribe/unsubscribe in ``asyncio.to_thread``
    so the blocking ``.result()`` happens on a worker thread.
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
        self._reconnect_not_before: float = 0.0
        self._logged_first_md_payload: bool = False

    @property
    def max_symbols(self) -> int:
        return self._max_symbols

    async def would_reject_new_symbol(self, sym: str) -> bool:
        """True if ``sym`` is not yet tracked and distinct-symbol cap is already reached."""
        async with self._lock:
            if sym in self._ref:
                return False
            return len(self._ref) >= self._max_symbols

    def _schedule_reconnect_cooldown(self) -> None:
        """Alpaca often still counts a session briefly after close; delay the next TCP open."""
        self._reconnect_not_before = time.monotonic() + _reconnect_cooldown_sec()

    async def _respect_reconnect_cooldown(self) -> None:
        delay = self._reconnect_not_before - time.monotonic()
        if delay > 0:
            _log.info("Alpaca L1: waiting %.1fs before reconnect (cooldown)", delay)
            await asyncio.sleep(delay)

    async def _subscribe_quotes_trades(self, stream: Any, sym: str) -> None:
        """Register symbol with Alpaca without blocking the event loop when the stream is live."""
        if getattr(stream, "_running", False):
            await asyncio.to_thread(
                _sync_subscribe_quotes_trades,
                stream,
                self._on_quote,
                self._on_trade,
                sym,
            )
        else:
            _sync_subscribe_quotes_trades(stream, self._on_quote, self._on_trade, sym)

    async def _unsubscribe_quotes_trades(self, stream: Any, sym: str) -> None:
        """Remove symbol from Alpaca without blocking the event loop when the stream is live."""
        if getattr(stream, "_running", False):
            await asyncio.to_thread(_sync_unsubscribe_quotes_trades, stream, sym)
        else:
            _sync_unsubscribe_quotes_trades(stream, sym)

    async def _alpaca_runner_supervisor(self, stream: Any) -> None:
        """Run Alpaca ``_run_forever``; when it exits (fatal error or end), reset hub state."""
        cancelled = False
        try:
            await stream._run_forever()  # noqa: SLF001
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if not cancelled:
                await self._finalize_stream_after_runner(stream)

    async def _finalize_stream_after_runner(self, stream: Any) -> None:
        """Clear shared stream state and tell browser clients the Alpaca link is gone."""
        pending: list[WebSocket] = []
        async with self._lock:
            if self._stream is not stream:
                return
            _clear_shunya_stream_control_handler(self._stream)
            self._stream = None
            self._runner_task = None
            self._corrections_registered = False
            for _sym, clients in list(self._clients.items()):
                pending.extend(list(clients))
            self._clients.clear()
            self._ref.clear()

        err = {
            "type": "error",
            "code": "alpaca_market_data_stopped",
            "message": (
                "Alpaca market-data connection ended (for example connection limit or plan). "
                "Wait a few seconds, then press Connect again."
            ),
        }
        self._schedule_reconnect_cooldown()
        for ws in pending:
            try:
                await ws.send_json(err)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Alpaca L1: could not push error to browser ws: %s", exc)
                try:
                    await ws.close(code=1011)
                except Exception:  # noqa: BLE001
                    pass

    async def attach(self, websocket: WebSocket, sym: str) -> None:
        """Subscribe ``websocket`` to Alpaca L1 for ``sym`` (refcounted)."""
        start_runner = False
        stream_for_remote_sub: Any = None
        sym_for_remote_sub: str | None = None

        while True:
            async with self._lock:
                if self._runner_task is not None and self._runner_task.done():
                    _log.warning("resetting stale Alpaca L1 hub state: runner task finished unexpectedly")
                    old_stream = self._stream
                    _clear_shunya_stream_control_handler(old_stream)
                    self._stream = None
                    self._runner_task = None
                    self._corrections_registered = False
                    self._clients.clear()
                    self._ref.clear()

                if sym in self._ref:
                    self._ref[sym] += 1
                    self._clients.setdefault(sym, set()).add(websocket)
                    return

                if len(self._ref) >= self._max_symbols:
                    raise SymbolLimitExceeded(self._max_symbols)

                if self._stream is not None:
                    self._ref[sym] = 1
                    self._clients.setdefault(sym, set()).add(websocket)
                    stream_for_remote_sub = self._stream
                    sym_for_remote_sub = sym
                    break

            await self._respect_reconnect_cooldown()

            async with self._lock:
                if sym in self._ref:
                    self._ref[sym] += 1
                    self._clients.setdefault(sym, set()).add(websocket)
                    return
                if self._stream is not None:
                    continue

                self._ref[sym] = 1
                self._clients.setdefault(sym, set()).add(websocket)
                self._stream = build_stock_data_stream(
                    self._rt,
                    feed=self._feed,
                    stream_cls=ShunyaL1StockDataStream,
                )
                setter = getattr(self._stream, "set_shunya_control_handler", None)
                if callable(setter):
                    setter(self._on_alpaca_control)
                self._logged_first_md_payload = False
                await self._subscribe_quotes_trades(self._stream, sym)
                if not self._corrections_registered:
                    self._stream.register_trade_corrections(self._on_trade_correction)
                    self._stream.register_trade_cancels(self._on_trade_cancel)
                    self._corrections_registered = True
                start_runner = True
                break

        if stream_for_remote_sub is not None and sym_for_remote_sub is not None:
            await self._subscribe_quotes_trades(stream_for_remote_sub, sym_for_remote_sub)

        if start_runner and self._stream is not None:
            s = self._stream
            self._runner_task = asyncio.create_task(
                self._alpaca_runner_supervisor(s),
                name="alpaca-l1-shared-run-forever",
            )

    async def detach(self, websocket: WebSocket, sym: str) -> None:
        """Unsubscribe ``websocket`` from ``sym``; may tear down the Alpaca stream when unused."""
        sym_unsub: str | None = None
        stream_unsub: Any = None
        want_shutdown = False

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

            sym_unsub = sym
            stream_unsub = stream
            want_shutdown = not self._ref

        # Let a concurrent attach for the same symbol run before we hit Alpaca unsubscribe.
        await asyncio.sleep(0)

        cancel_unsub = False
        async with self._lock:
            if sym in self._ref and self._clients.get(sym):
                cancel_unsub = True

        if cancel_unsub or sym_unsub is None or stream_unsub is None:
            return

        try:
            await self._unsubscribe_quotes_trades(stream_unsub, sym_unsub)
        except Exception as exc:  # noqa: BLE001
            _log.debug("unsubscribe %s: %s", sym_unsub, exc)

        if not want_shutdown:
            return

        async with self._lock:
            if self._ref or self._stream is not stream_unsub:
                return
            await self._shutdown_stream_locked()

    async def _shutdown_stream_locked(self) -> None:
        """Stop Alpaca stream; caller must hold ``self._lock``."""
        stream = self._stream
        self._stream = None
        self._corrections_registered = False
        task = self._runner_task
        self._runner_task = None

        _clear_shunya_stream_control_handler(stream)

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

        self._schedule_reconnect_cooldown()

    async def _on_quote(self, q: Any) -> None:
        sym = _symbol_upper(q)
        if not sym:
            return
        try:
            body = alpaca_quote_to_dict(q)
        except Exception as exc:  # noqa: BLE001
            _log.debug("skip malformed quote: %s", exc)
            return
        if not self._logged_first_md_payload:
            self._logged_first_md_payload = True
            _log.info("Alpaca L1: first market-data payload (quote) symbol=%s", sym)
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
        if not self._logged_first_md_payload:
            self._logged_first_md_payload = True
            _log.info("Alpaca L1: first market-data payload (trade) symbol=%s", sym)
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

    async def _on_alpaca_control(self, msg: dict[str, Any]) -> None:
        """Forward Alpaca ``T: error`` / ``T: subscription`` control frames to all browser sockets."""
        t = msg.get("T")
        if t == "error":
            code = msg.get("code")
            body = msg.get("msg")
            parts: list[str] = []
            if code is not None:
                parts.append(str(code))
            if body is not None:
                parts.append(str(body))
            text = ": ".join(parts) if parts else "Alpaca market-data error"
            await self._fanout_all_browser_ws(
                {"type": "error", "code": "alpaca_upstream", "message": text}
            )
        elif t == "subscription":
            safe = _json_safe_alpaca_obj(msg)
            await self._fanout_all_browser_ws({"type": "subscription", "alpaca": safe})

    async def _fanout_all_browser_ws(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets: list[WebSocket] = []
            for clients in self._clients.values():
                targets.extend(clients)
        seen: set[int] = set()
        for ws in targets:
            wid = id(ws)
            if wid in seen:
                continue
            seen.add(wid)
            try:
                await ws.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Alpaca L1 broadcast send failed: %s", exc)

    async def _fanout_json(self, sym: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients.get(sym, ()))
        for ws in targets:
            try:
                await ws.send_json(payload)
            except Exception as exc:  # noqa: BLE001
                _log.debug("ws send failed, dropping client for %s: %s", sym, exc)
                await self.detach(ws, sym)
