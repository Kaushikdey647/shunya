"""Deprecated Alpaca bar WebSocket; use :mod:`api.routers.instrument_l1_stream` for L1 quotes + trades."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from api.services.market_symbols import normalize_market_symbol
from api.settings import get_settings
from shunya.integration.alpaca_settings import try_load_alpaca_settings_from_env

router = APIRouter(prefix="/instruments", tags=["instruments"])


@router.websocket("/{symbol}/stream/alpaca-bars")
async def instrument_alpaca_bars_ws(websocket: WebSocket, symbol: str) -> None:
    """Removed: clients must use ``/instruments/{symbol}/stream/alpaca-l1``."""
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

    if try_load_alpaca_settings_from_env() is None:
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

    await websocket.send_json(
        {
            "type": "error",
            "code": "deprecated_stream",
            "message": (
                "The alpaca-bars WebSocket is removed. "
                "Use /instruments/{symbol}/stream/alpaca-l1 for IEX BBO quotes and trades."
            ),
            "replacement_path": f"/instruments/{sym}/stream/alpaca-l1",
        }
    )
    await websocket.close(code=1008)
