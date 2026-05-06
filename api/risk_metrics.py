from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from shunya.data.timeframes import BarSpec


def periods_per_year_from_bar_spec(bar: BarSpec) -> float:
    if bar.unit.name == "SECONDS":
        return (252.0 * 6.5 * 60.0 * 60.0) / max(1.0, float(bar.step))
    if bar.unit.name == "MINUTES":
        return (252.0 * 6.5 * 60.0) / max(1.0, float(bar.step))
    if bar.unit.name == "HOURS":
        return (252.0 * 6.5) / max(1.0, float(bar.step))
    if bar.unit.name == "DAYS":
        return 252.0 / max(1.0, float(bar.step))
    if bar.unit.name == "WEEKS":
        return 52.0 / max(1.0, float(bar.step))
    if bar.unit.name == "MONTHS":
        return 12.0 / max(1.0, float(bar.step))
    return 1.0 / max(1.0, float(bar.step))


def per_bar_return_stats_with_ppy(
    close: pd.Series, periods_per_year: float
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """From aligned Close (ascending), (total_return_pct, ann_vol_pct, sharpe, sortino)."""
    s = close.astype(float).dropna()
    if len(s) < 2:
        return None, None, None, None
    c0 = float(s.iloc[0])
    c1 = float(s.iloc[-1])
    ret_pct = ((c1 / c0 - 1.0) * 100.0) if c0 > 0 else None
    r = s.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return ret_pct, None, None, None
    std = float(r.std(ddof=1))
    mean = float(r.mean())
    if std <= 0.0 or not math.isfinite(std):
        sharpe: Optional[float] = None
        risk_ann = None
    else:
        sharpe = float(math.sqrt(periods_per_year) * (mean / std))
        risk_ann = float(std * math.sqrt(periods_per_year) * 100.0)
    neg = r[r < 0.0]
    if len(neg) < 2:
        sortino: Optional[float] = None
    else:
        dstd = float(neg.std(ddof=1))
        if dstd <= 0.0 or not math.isfinite(dstd):
            sortino = None
        else:
            sortino = float(math.sqrt(periods_per_year) * (mean / dstd))
    return ret_pct, risk_ann, sharpe, sortino
