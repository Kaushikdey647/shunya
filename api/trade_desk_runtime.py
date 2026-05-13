"""Process-wide Alpaca clients for the trade desk (built at FastAPI lifespan)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shunya.integration.alpaca_settings import (
    AlpacaRuntimeSettings,
    build_stock_historical_data_client,
    build_trading_client,
)

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient


@dataclass(slots=True)
class TradeDeskRuntime:
    """Shared Alpaca-py clients for :class:`~shunya.live.desk.InstitutionalPaperDesk`."""

    settings: AlpacaRuntimeSettings
    trading_client: "TradingClient"
    data_client: "StockHistoricalDataClient"


def build_trade_desk_runtime(settings: AlpacaRuntimeSettings) -> TradeDeskRuntime:
    return TradeDeskRuntime(
        settings=settings,
        trading_client=build_trading_client(settings),
        data_client=build_stock_historical_data_client(settings),
    )
