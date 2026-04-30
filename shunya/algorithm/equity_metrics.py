"""Derived headline metrics from equity curves (FinBT + tune-window recomputation)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def cagr_pct_from_equity_df(equity: pd.DataFrame) -> Optional[float]:
    """
    Compound annual growth rate from first to last equity using calendar span (365.25-day years).

    Returns None when span is too short or values are non-positive.
    """
    if equity.empty or "Equity" not in equity.columns or len(equity) < 2:
        return None
    start_val = float(equity["Equity"].iloc[0])
    end_val = float(equity["Equity"].iloc[-1])
    if start_val <= 0 or end_val <= 0:
        return None
    t0 = pd.Timestamp(equity.index[0])
    t1 = pd.Timestamp(equity.index[-1])
    delta = t1 - t0
    days = delta.days + delta.seconds / 86400.0 + delta.microseconds / 86_400_000_000.0
    if days < 1.0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    ratio = end_val / start_val
    if ratio <= 0 or not np.isfinite(ratio):
        return None
    return float((ratio ** (1.0 / years) - 1.0) * 100.0)


def win_rate_pct_from_equity_returns(eq_ret: pd.Series) -> float:
    """
    Fraction of bars with strictly positive equity pct-change, times 100.

    Aligns with per-bar strategy evaluation frequency (daily when bars are daily).
    """
    if eq_ret.empty:
        return 0.0
    r = eq_ret.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return 0.0
    return float((r > 0.0).mean() * 100.0)
