"""Pyfolio-style summaries and Fama–French single-factor regressions (no pyfolio dependency)."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from shunya.data.ff_factors import fetch_ff_factors_daily

FactorFrameFn = Callable[[], pd.DataFrame]


def compute_return_quantiles(returns: pd.Series) -> dict[str, Any]:
    """Quintiles / deciles of bar returns (fractional, e.g. 0.01 = 1%)."""
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return {
            "count": int(len(r)),
            "q05": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "q95": None,
        }
    qs = r.quantile([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "count": int(len(r)),
        "q05": float(qs.loc[0.05]) if pd.notna(qs.loc[0.05]) else None,
        "q10": float(qs.loc[0.1]) if pd.notna(qs.loc[0.1]) else None,
        "q25": float(qs.loc[0.25]) if pd.notna(qs.loc[0.25]) else None,
        "median": float(qs.loc[0.5]) if pd.notna(qs.loc[0.5]) else None,
        "q75": float(qs.loc[0.75]) if pd.notna(qs.loc[0.75]) else None,
        "q90": float(qs.loc[0.9]) if pd.notna(qs.loc[0.9]) else None,
        "q95": float(qs.loc[0.95]) if pd.notna(qs.loc[0.95]) else None,
    }


def compute_tearsheet_summary(
    equity: pd.DataFrame,
    *,
    periods_per_year: float,
    max_drawdown_len: int,
) -> dict[str, Any]:
    """Risk / return style summary from equity curve (trimmed window)."""
    if equity.empty or "Equity" not in equity.columns:
        return {
            "ann_mean_return_pct": None,
            "ann_volatility_pct": None,
            "skew": None,
            "kurtosis": None,
            "worst_bar_returns_pct": [],
            "max_drawdown_len_bars": int(max_drawdown_len),
        }
    r = equity["Equity"].astype(float).pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return {
            "ann_mean_return_pct": None,
            "ann_volatility_pct": None,
            "skew": None,
            "kurtosis": None,
            "worst_bar_returns_pct": [],
            "max_drawdown_len_bars": int(max_drawdown_len),
        }
    mean_b = float(r.mean())
    std_b = float(r.std(ddof=1))
    ann_mean = float(mean_b * periods_per_year * 100.0)
    ann_vol = float(std_b * np.sqrt(periods_per_year) * 100.0) if std_b > 0 else None
    skew = float(stats.skew(r.to_numpy(), bias=False)) if len(r) > 2 else None
    kurt = float(stats.kurtosis(r.to_numpy(), bias=False)) if len(r) > 3 else None
    worst = r.nsmallest(min(5, len(r))).to_numpy(dtype=float)
    worst_pct = [float(x * 100.0) for x in worst.tolist()]
    return {
        "ann_mean_return_pct": ann_mean,
        "ann_volatility_pct": ann_vol,
        "skew": skew,
        "kurtosis": kurt,
        "worst_bar_returns_pct": worst_pct,
        "max_drawdown_len_bars": int(max_drawdown_len),
    }


def _single_factor_ols(
    y: np.ndarray,
    x: np.ndarray,
    *,
    periods_per_year: float,
) -> dict[str, Any]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    n = int(y.shape[0])
    if n < 10:
        return {"alpha_daily": None, "alpha_ann_pct": None, "beta": None, "r2": None, "n": n}
    X = np.column_stack([np.ones(n, dtype=float), x.astype(float)])
    coef, _, _, _ = np.linalg.lstsq(X, y.astype(float), rcond=None)
    alpha_d = float(coef[0])
    beta = float(coef[1])
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else None
    alpha_ann_pct = float(((1.0 + alpha_d) ** periods_per_year - 1.0) * 100.0)
    return {
        "alpha_daily": alpha_d,
        "alpha_ann_pct": alpha_ann_pct,
        "beta": beta,
        "r2": r2,
        "n": n,
    }


def compute_ff_single_factor(
    equity: pd.DataFrame,
    *,
    fetch_factors: Optional[FactorFrameFn] = None,
) -> dict[str, Any]:
    """
    Univariate CAPM-style regressions: (R - RF) ~ alpha + beta * F for F in Mkt-RF, SMB, HML.

    Uses daily Fama–French factors; strategy returns are aligned on calendar dates (normalized).
    """
    fetch_factors = fetch_factors or fetch_ff_factors_daily
    out: dict[str, Any] = {"factors": {}, "error": None}
    if equity.empty or "Equity" not in equity.columns:
        out["error"] = "empty_equity"
        return out
    r = equity["Equity"].astype(float).pct_change()
    r.index = pd.DatetimeIndex(r.index).normalize()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 10:
        out["error"] = "insufficient_returns"
        return out
    try:
        ff = fetch_factors()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"ff_fetch_failed:{exc}"
        return out
    ff = ff.copy()
    ff.index = pd.DatetimeIndex(ff.index).normalize()
    joined = pd.concat([r.rename("R"), ff], axis=1, join="inner").dropna()
    if len(joined) < 10:
        out["error"] = "insufficient_overlap"
        return out
    rf = joined["RF"].to_numpy(dtype=float)
    excess = joined["R"].to_numpy(dtype=float) - rf
    for fname in ("Mkt-RF", "SMB", "HML"):
        fac = joined[fname].to_numpy(dtype=float)
        out["factors"][fname] = _single_factor_ols(excess, fac, periods_per_year=252.0)
    return out


def build_turnover_pct_history(turnover_df: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Align turnover USD with equity; ``TurnoverPct`` = turnover / equity (fraction, not %)."""
    if turnover_df.empty or equity.empty or "Equity" not in equity.columns:
        return pd.DataFrame(columns=["TurnoverPct"])
    if "TurnoverUSD" not in turnover_df.columns:
        return pd.DataFrame(columns=["TurnoverPct"])
    aligned = turnover_df.join(equity[["Equity"]], how="left").ffill()
    eqv = aligned["Equity"].replace(0, np.nan).astype(float)
    pct = (aligned["TurnoverUSD"].astype(float) / eqv).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"TurnoverPct": pct}, index=aligned.index)
