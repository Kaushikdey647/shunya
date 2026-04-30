from __future__ import annotations

import os
from typing import Optional

from shunya.data.fints import finTs
from shunya.data.providers import MarketDataProvider

from shunya.schemas import FinTsRequest, bar_spec_model_to_bar_spec

from backtest_api.errors import FinTsConfigurationError
from backtest_api.timescale_classifications import load_classifications_for_tickers


def resolve_market_data_provider(req: FinTsRequest) -> Optional[MarketDataProvider]:
    mode = req.market_data_provider
    if mode == "yfinance":
        return None
    if mode == "timescale":
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider
        except ImportError as exc:
            raise FinTsConfigurationError(
                "Timescale provider requires: pip install 'shunya-py[timescale]'",
                status_code=503,
            ) from exc
        try:
            return TimescaleMarketDataProvider()
        except ValueError as exc:
            raise FinTsConfigurationError(
                "Timescale provider requires DATABASE_URL or SHUNYA_DATABASE_URL.",
                status_code=503,
            ) from exc
    if os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL"):
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider

            return TimescaleMarketDataProvider()
        except (ImportError, ValueError):
            return None
    return None


def build_fin_ts(req: FinTsRequest) -> finTs:
    bar_spec = bar_spec_model_to_bar_spec(req.bar_spec)
    md = resolve_market_data_provider(req)
    if req.market_data_provider == "timescale" and md is None:
        raise FinTsConfigurationError(
            "market_data_provider=timescale but Timescale is not available (DSN or psycopg).",
            status_code=503,
        )
    classifications = None
    if md is not None and not req.attach_yfinance_classifications and req.ticker_list:
        classifications = load_classifications_for_tickers(list(req.ticker_list))

    kwargs: dict = {
        "start_date": req.start_date,
        "end_date": req.end_date,
        "ticker_list": req.ticker_list,
        "market_data": md,
        "classifications": classifications,
        "attach_yfinance_classifications": req.attach_yfinance_classifications,
        "attach_fundamentals": req.attach_fundamentals,
        "bar_spec": bar_spec,
        "strict_provider_universe": req.strict_provider_universe,
        "strict_ohlcv": req.strict_ohlcv,
        "strict_empty": req.strict_empty,
        "feature_mode": req.feature_mode,
        "trading_axis_mode": req.trading_axis_mode,
        "strict_trading_grid": req.strict_trading_grid,
    }
    if req.require_history_bars is not None:
        kwargs["require_history_bars"] = req.require_history_bars
    return finTs(**kwargs)
