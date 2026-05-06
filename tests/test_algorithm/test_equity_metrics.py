"""Tests for shunya.algorithm.equity_metrics (no api imports)."""

from __future__ import annotations

import pandas as pd

from shunya.algorithm.equity_metrics import cagr_pct_from_equity_df, win_rate_pct_from_equity_returns


def test_win_rate_pct_simple() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    assert abs(win_rate_pct_from_equity_returns(r) - 50.0) < 1e-9


def test_cagr_pct_simple() -> None:
    idx = pd.date_range("2020-01-01", periods=366 * 2, freq="D")
    # ~double over 2 calendar years => CAGR ~41%
    eq = pd.DataFrame({"Equity": [100.0] + [100.0 + (100.0 * t / (len(idx) - 1)) for t in range(len(idx) - 1)]}, index=idx)
    # linear growth 100->200
    eq["Equity"] = [100.0 + 100.0 * i / (len(idx) - 1) for i in range(len(idx))]
    cagr = cagr_pct_from_equity_df(eq)
    assert cagr is not None
    assert 35.0 < cagr < 45.0
