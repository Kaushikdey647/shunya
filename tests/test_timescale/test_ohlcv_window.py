"""Unit tests for instrument OHLCV window mapping."""

from __future__ import annotations

import pandas as pd
import pytest

from shunya.data.timeframes import BarUnit
from shunya.data.timescale.ohlcv_window import period_to_utc_bounds, yfinance_interval_to_bar_spec


def test_yfinance_interval_to_bar_spec() -> None:
    assert yfinance_interval_to_bar_spec("1d").unit == BarUnit.DAYS
    assert yfinance_interval_to_bar_spec("60m").step == 60
    assert yfinance_interval_to_bar_spec("1h").unit == BarUnit.HOURS


def test_yfinance_interval_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        yfinance_interval_to_bar_spec("2h")


def test_period_to_utc_bounds_ytd_anchor() -> None:
    anchor = pd.Timestamp("2024-06-15", tz="UTC")
    start, end_excl = period_to_utc_bounds("ytd", anchor=anchor)
    assert start == pd.Timestamp("2024-01-01", tz="UTC")
    assert end_excl == pd.Timestamp("2024-06-16", tz="UTC")
