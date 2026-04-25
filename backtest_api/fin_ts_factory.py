from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException

from shunya.data.fints import finTs
from shunya.data.providers import MarketDataProvider
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_spec

from backtest_api.schemas.models import FinTsRequest


def _bar_spec_from_model(req: FinTsRequest) -> BarSpec:
    if req.bar_spec is None:
        return default_bar_spec()
    unit = BarUnit(req.bar_spec.unit)
    return BarSpec(unit=unit, step=req.bar_spec.step)


def resolve_market_data_provider(req: FinTsRequest) -> Optional[MarketDataProvider]:
    mode = req.market_data_provider
    if mode == "yfinance":
        return None
    if mode == "timescale":
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="Timescale provider requires: pip install 'shunya-py[timescale]'",
            ) from exc
        try:
            return TimescaleMarketDataProvider()
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="Timescale provider requires DATABASE_URL or SHUNYA_DATABASE_URL.",
            ) from exc
    if os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL"):
        try:
            from shunya.data.timescale.market_provider import TimescaleMarketDataProvider

            return TimescaleMarketDataProvider()
        except (ImportError, ValueError):
            return None
    return None


def build_fin_ts(req: FinTsRequest) -> finTs:
    bar_spec = _bar_spec_from_model(req)
    md = resolve_market_data_provider(req)
    if req.market_data_provider == "timescale" and md is None:
        raise HTTPException(
            status_code=503,
            detail="market_data_provider=timescale but Timescale is not available (DSN or psycopg).",
        )
    kwargs: dict = {
        "start_date": req.start_date,
        "end_date": req.end_date,
        "ticker_list": req.ticker_list,
        "market_data": md,
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
