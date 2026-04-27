"""Trim FinBT ``results()`` output to the tune window when test period is excluded from results."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest_api.backtest_windows import BACKTEST_TEST_START


def _cutoff_timestamp(equity_index: pd.DatetimeIndex) -> pd.Timestamp:
    c = pd.Timestamp(BACKTEST_TEST_START)
    tz = getattr(equity_index, "tz", None)
    if tz is not None:
        if c.tzinfo is None:
            c = c.tz_localize(tz)
        else:
            c = c.tz_convert(tz)
    return c


def _periods_per_year(bar_unit: str, bar_step: int) -> float:
    step = max(1.0, float(bar_step))
    if bar_unit == "SECONDS":
        return (252.0 * 6.5 * 60.0 * 60.0) / step
    if bar_unit == "MINUTES":
        return (252.0 * 6.5 * 60.0) / step
    if bar_unit == "HOURS":
        return (252.0 * 6.5) / step
    if bar_unit == "DAYS":
        return 252.0 / step
    if bar_unit == "WEEKS":
        return 52.0 / step
    if bar_unit == "MONTHS":
        return 12.0 / step
    return 1.0 / step


def _recompute_metrics(
    equity: pd.DataFrame,
    turnover_df: pd.DataFrame,
    target_history: list[tuple[Any, Any]],
    group_exposure_history: list[tuple[Any, Any]],
    *,
    bar_unit: str,
    bar_step: int,
    cash: float,
    prior_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Mirror key FinBT.results() metrics for a possibly sliced equity curve."""
    if equity.empty or "Equity" not in equity.columns:
        return {**prior_metrics, "total_return_pct": 0.0, "end_value": float(cash)}

    eq = equity.copy()
    eq["Peak"] = eq["Equity"].cummax()
    eq["DrawdownPct"] = (eq["Equity"] / eq["Peak"] - 1.0) * 100.0

    eq_ret = eq["Equity"].pct_change().dropna()
    start_val = float(eq["Equity"].iloc[0])
    end_val = float(eq["Equity"].iloc[-1])
    total_return_pct = ((end_val / start_val - 1.0) * 100.0) if start_val > 0 else 0.0
    avg_bar_return_pct = float(eq_ret.mean() * 100.0) if len(eq_ret) else 0.0
    max_drawdown_pct = float(abs(eq["DrawdownPct"].min())) if len(eq) else 0.0
    in_dd = (eq["DrawdownPct"] < 0).astype(int) if len(eq) else pd.Series(dtype=int)
    max_dd_len = 0
    run = 0
    for v in in_dd.to_numpy(dtype=int):
        if v:
            run += 1
            if run > max_dd_len:
                max_dd_len = run
        else:
            run = 0

    periods_per_year = _periods_per_year(bar_unit, bar_step)
    if len(eq_ret) > 1 and float(eq_ret.std(ddof=1)) > 0:
        sharpe_trimmed = float(np.sqrt(periods_per_year) * (eq_ret.mean() / eq_ret.std(ddof=1)))
    else:
        sharpe_trimmed = None

    exec_start = prior_metrics.get("execution_start")
    trimmed_pre = prior_metrics.get("trimmed_pre_execution")

    metrics: dict[str, Any] = {
        "start_value": start_val,
        "end_value": end_val,
        "total_return_pct": total_return_pct,
        "avg_daily_return_pct": avg_bar_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_len": int(max_dd_len),
        "sharpe_ratio": sharpe_trimmed,
        "execution_start": exec_start,
        "trimmed_pre_execution": trimmed_pre,
        "bar_unit": bar_unit,
        "bar_step": int(bar_step),
        "analyzer_total_return_pct": None,
        "analyzer_avg_return_pct": None,
        "analyzer_max_drawdown_pct": None,
        "analyzer_sharpe_ratio": None,
    }

    if not turnover_df.empty and not eq.empty:
        aligned = turnover_df.join(eq[["Equity"]], how="left").ffill()
        turnover_pct = aligned["TurnoverUSD"] / aligned["Equity"].replace(0, np.nan)
        metrics["avg_turnover_pct"] = float(turnover_pct.mean(skipna=True) * 100.0)
        metrics["max_turnover_pct"] = float(turnover_pct.max(skipna=True) * 100.0)
        metrics["rebalance_count"] = int(len(turnover_df))
    else:
        metrics["avg_turnover_pct"] = 0.0
        metrics["max_turnover_pct"] = 0.0
        metrics["rebalance_count"] = 0

    if target_history:
        latest_targets = target_history[-1][1]
        gross = sum(abs(v) for v in latest_targets.values())
        top = max((abs(v) for v in latest_targets.values()), default=0.0)
        metrics["top_name_gross_share_pct"] = float((top / gross) * 100.0) if gross > 0 else 0.0
    else:
        metrics["top_name_gross_share_pct"] = 0.0

    if group_exposure_history:
        _, ge = group_exposure_history[-1]
        gross_map = ge.get("gross_by_group", {})
        net_map = ge.get("net_by_group", {})
        total_g = sum(float(v) for v in gross_map.values())
        max_group_gross = max((float(v) for v in gross_map.values()), default=0.0)
        max_group_net_abs = max((abs(float(v)) for v in net_map.values()), default=0.0)
        metrics["max_group_gross_share_pct"] = (
            float((max_group_gross / total_g) * 100.0) if total_g > 0 else 0.0
        )
        metrics["max_group_net_share_pct"] = (
            float((max_group_net_abs / total_g) * 100.0) if total_g > 0 else 0.0
        )
    else:
        metrics["max_group_gross_share_pct"] = 0.0
        metrics["max_group_net_share_pct"] = 0.0

    return metrics


def apply_tune_only_to_finbt_results(raw: dict[str, Any], *, include_test: bool) -> dict[str, Any]:
    """
    When ``include_test`` is False, drop rows at or after ``BACKTEST_TEST_START`` from
    equity / turnover / histories and recompute headline metrics; clear Cerebro analyzer dicts.
    """
    if include_test:
        return raw

    equity: pd.DataFrame = raw["equity_curve"]
    if equity.empty:
        return raw

    cutoff = _cutoff_timestamp(equity.index)
    mask = equity.index < cutoff
    eq_s = equity.loc[mask].copy()
    if eq_s.empty:
        return raw

    out = dict(raw)
    out["equity_curve"] = eq_s

    to = raw.get("turnover_history")
    if isinstance(to, pd.DataFrame) and not to.empty:
        out["turnover_history"] = to.loc[to.index < cutoff].copy()
    else:
        out["turnover_history"] = to

    th: list = list(raw.get("target_history") or [])
    out["target_history"] = [(dt, tg) for dt, tg in th if pd.Timestamp(dt) < cutoff]

    gh: list = list(raw.get("group_exposure_history") or [])
    out["group_exposure_history"] = [(dt, ge) for dt, ge in gh if pd.Timestamp(dt) < cutoff]

    pm: dict[str, Any] = dict(raw.get("metrics") or {})
    cash = float(pm.get("start_value", 100_000.0))
    bar_unit = str(pm.get("bar_unit", "DAYS"))
    bar_step = int(pm.get("bar_step", 1))
    to_df = out["turnover_history"] if isinstance(out["turnover_history"], pd.DataFrame) else pd.DataFrame()

    out["metrics"] = _recompute_metrics(
        eq_s,
        to_df,
        out["target_history"],
        out["group_exposure_history"],
        bar_unit=bar_unit,
        bar_step=bar_step,
        cash=cash,
        prior_metrics=pm,
    )
    out["returns_analysis"] = None
    out["drawdown_analysis"] = None
    out["sharpe_analysis"] = None
    return out
