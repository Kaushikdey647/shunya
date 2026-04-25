from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_api.risk_metrics import per_bar_return_stats_with_ppy, periods_per_year_from_bar_spec
from shunya.data.timeframes import BarSpec, BarUnit


def test_periods_per_year_daily() -> None:
    ppy = periods_per_year_from_bar_spec(BarSpec(BarUnit.DAYS, 1))
    assert ppy == 252.0


def test_per_bar_stats_flat_up() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    close = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    ret, risk, sharpe, sortino = per_bar_return_stats_with_ppy(close, 252.0)
    assert ret is not None and ret > 0
    assert risk is not None and risk >= 0
