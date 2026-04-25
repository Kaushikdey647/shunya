"""Map :class:`~shunya.data.timeframes.BarSpec` to stable ``ohlcv_bars.interval`` text keys."""

from __future__ import annotations

from typing import Literal

from ..timeframes import BarSpec, BarUnit, bar_spec_to_yfinance_interval


def bar_spec_to_interval_key(spec: BarSpec) -> str:
    """
    Storage key for ``ohlcv_bars.interval`` (matches post-normalize bar cadence).

    Yearly bars fetched via monthly+yfinance resample are stored under ``1y``.
    """
    y = bar_spec_to_yfinance_interval(spec)
    if y == "__monthly_then_year_resample":
        return "1y"
    return str(y)
