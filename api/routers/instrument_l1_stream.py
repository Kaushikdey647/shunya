"""WebSocket bridge: browser ↔ shared Alpaca :class:`~alpaca.data.live.stock.StockDataStream` (IEX L1 quotes + trades).

One **process-wide** Alpaca market-data connection per API key + feed is shared across all browser
sessions; see :mod:`api.services.alpaca_l1_feed_hub`. Distinct symbols are capped by
``SHUNYA_ALPACA_L1_MAX_SYMBOLS`` (default 30). For strict one TCP session per key across **machines**,
run a single API replica / uvicorn worker or accept one stream per process.

See: https://docs.alpaca.markets/us/docs/streaming-market-data
"""

from __future__ import annotations

import asyncio
import logging

from alpaca.data.enums import DataFeed
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from api.services.alpaca_l1_feed_hub import SymbolLimitExceeded, get_alpaca_l1_hub
from api.services.instrument_dashboard import fetch_instrument_overview
from api.services.market_symbols import normalize_market_symbol
from api.settings import get_settings
from shunya.integration.alpaca_settings import try_load_alpaca_settings_from_env

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/instruments", tags=["instruments"])


async def _wait_websocket_disconnect(websocket: WebSocket) -> None:
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


@router.websocket("/{symbol}/stream/alpaca-l1")
async def instrument_alpaca_l1_ws(websocket: WebSocket, symbol: str) -> None:
    await websocket.accept()

    if not get_settings().alpaca_enabled:
        await websocket.send_json(
            {
                "type": "error",
                "code": "alpaca_disabled",
                "message": "Alpaca integration is disabled for this API deployment.",
            }
        )
        await websocket.close(code=1008)
        return

    rt = try_load_alpaca_settings_from_env()
    if rt is None:
        await websocket.send_json(
            {
                "type": "error",
                "code": "alpaca_disabled",
                "message": "Alpaca credentials are not configured.",
            }
        )
        await websocket.close(code=1008)
        return

    try:
        sym = normalize_market_symbol(symbol)
    except ValueError:
        await websocket.send_json(
            {
                "type": "error",
                "code": "invalid_symbol",
                "message": "Invalid market symbol.",
            }
        )
        await websocket.close(code=1008)
        return

    try:
        overview = await asyncio.to_thread(fetch_instrument_overview, sym)
    except Exception as exc:  # noqa: BLE001
        _log.warning("instrument L1 stream overview failed for %s: %s", sym, exc)
        await websocket.send_json(
            {
                "type": "error",
                "code": "stream_failed",
                "message": "Could not load instrument metadata.",
            }
        )
        await websocket.close(code=1011)
        return

    kind = overview.instrument_kind
    if kind not in ("equity", "etf"):
        await websocket.send_json(
            {
                "type": "error",
                "code": "unsupported_instrument",
                "message": f"Live Alpaca L1 is only available for stocks and ETFs (got {kind!r}).",
            }
        )
        await websocket.close(code=1008)
        return

    feed = DataFeed.IEX
    hub = get_alpaca_l1_hub(rt, feed=feed)

    if await hub.would_reject_new_symbol(sym):
        await websocket.send_json(
            {
                "type": "error",
                "code": "symbol_limit",
                "message": (
                    f"Too many distinct symbols on the shared Alpaca L1 stream "
                    f"(max {hub.max_symbols}). Close another instrument tab or raise "
                    "SHUNYA_ALPACA_L1_MAX_SYMBOLS if your Alpaca plan allows it."
                ),
            }
        )
        await websocket.close(code=1008)
        return

    await websocket.send_json(
        {
            "type": "hello",
            "schema": 1,
            "symbol": sym,
            "feed": feed.value,
            "channels": ["quotes", "trades"],
        }
    )

    try:
        await hub.attach(websocket, sym)
    except SymbolLimitExceeded:
        await websocket.send_json(
            {
                "type": "error",
                "code": "symbol_limit",
                "message": (
                    f"Too many distinct symbols on the shared Alpaca L1 stream "
                    f"(max {hub.max_symbols})."
                ),
            }
        )
        await websocket.close(code=1008)
        return

    try:
        await _wait_websocket_disconnect(websocket)
    finally:
        try:
            await hub.detach(websocket, sym)
        except Exception as exc:  # noqa: BLE001
            _log.debug("hub detach: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass
