"""Unit tests for Timescale helpers (no database required)."""

from __future__ import annotations

import pandas as pd

from shunya.data.timeframes import BarSpec, BarUnit
from shunya.data.timescale.ingest_lib import fundamentals_eav_rows, rows_from_provider_ohlcv
from shunya.data.timescale.intervals import bar_spec_to_interval_key


def test_bar_spec_to_interval_key_daily_and_yearly() -> None:
    assert bar_spec_to_interval_key(BarSpec(BarUnit.DAYS, 1)) == "1d"
    assert bar_spec_to_interval_key(BarSpec(BarUnit.YEARS, 1)) == "1y"


def test_rows_from_provider_ohlcv_single_ticker() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    raw = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.5, 11.5],
            "Low": [9.5, 10.5],
            "Close": [10.2, 11.1],
            "Volume": [1000.0, 2000.0],
        },
        index=idx,
    )
    rows = rows_from_provider_ohlcv(raw, {"SPY": 42}, interval="1d", source="yfinance")
    assert len(rows) == 2
    assert rows[0][0] == 42 and rows[0][2] == "1d" and rows[0][8] == "yfinance"


def test_rows_from_provider_ohlcv_skips_nonfinite_and_negative_volume() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    raw = pd.DataFrame(
        {
            "Open": [10.0, float("nan"), 12.0],
            "High": [10.5, 11.5, 12.5],
            "Low": [9.5, 10.5, 11.5],
            "Close": [10.2, 11.1, 12.1],
            "Volume": [1000.0, 2000.0, -1.0],
        },
        index=idx,
    )
    rows = rows_from_provider_ohlcv(raw, {"SPY": 42}, interval="1d", source="yfinance")
    assert len(rows) == 1
    assert rows[0][6] == 10.2  # close first valid row


def test_timescale_wide_ohlcv_concat_inner_join_no_cross_sectional_nan() -> None:
    """Multi-ticker Timescale panels must not use outer concat (NaNs on union index)."""
    idx_a = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    idx_b = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    a = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.1, 2.1],
            "Low": [0.9, 1.9],
            "Close": [1.05, 2.05],
            "Volume": [100.0, 200.0],
        },
        index=idx_a,
    )
    b = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [10.1, 11.1],
            "Low": [9.9, 10.9],
            "Close": [10.05, 11.05],
            "Volume": [1000.0, 2000.0],
        },
        index=idx_b,
    )
    parts = [a, b]
    keys = ["AAA", "BBB"]
    wide = pd.concat(parts, keys=keys, axis=1, join="inner")
    wide.columns = wide.columns.set_names(["Ticker", None])
    assert len(wide) == 1
    assert wide.index[0] == pd.Timestamp("2024-01-03")
    ohlc = wide.loc[:, pd.IndexSlice[:, ["Open", "High", "Low", "Close", "Volume"]]]
    assert not ohlc.isna().any().any()


def test_fundamentals_eav_rows() -> None:
    idx = pd.MultiIndex.from_tuples(
        [("SPY", pd.Timestamp("2023-06-30")), ("SPY", pd.Timestamp("2023-09-30"))],
        names=["Ticker", "Date"],
    )
    periodic = pd.DataFrame({"Revenue": [1.0, 2.0], "Return_On_Equity": [0.1, 0.2]}, index=idx)
    rows = fundamentals_eav_rows(periodic, {"SPY": 7}, freq="quarterly", source="yfinance_statements")
    assert len(rows) == 4
    assert {r[3] for r in rows} == {"Revenue", "Return_On_Equity"}
