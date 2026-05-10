from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError

from shunya.algorithm.finbt import FinBT
from shunya.algorithm.finstrat import FinStrat

from api.fin_ts_factory import build_fin_ts
from api.resolver import resolve_alpha_for_backtest
from api.result_tune_filter import apply_tune_only_to_finbt_results
from api.schemas.models import BacktestCreate, FinStratConfig, FinTsRequest
from api.serializer import (
    equity_curve_to_records,
    result_summary_from_metrics,
    serialize_backtest_result,
)
from api.settings import get_settings
from shunya.schemas import merge_finstrat_runtime_dict


def _merge_finstrat(stored: dict[str, Any], override: Optional[FinStratConfig]) -> dict[str, Any]:
    return merge_finstrat_runtime_dict(stored, override)


def _benchmark_equity_curve_records(
    strategy_equity: pd.DataFrame,
    bench_close: pd.Series,
) -> list[dict[str, Any]]:
    """Buy-and-hold benchmark equity in dollars, aligned to each strategy bar (forward-filled closes)."""
    if strategy_equity.empty or "Equity" not in strategy_equity.columns:
        return []
    idx = strategy_equity.index.sort_values()
    eq = strategy_equity["Equity"].astype(float).reindex(idx)
    b_sorted = bench_close.astype(float).sort_index()
    if b_sorted.empty:
        return []
    aligned = b_sorted.reindex(b_sorted.index.union(idx)).sort_index().ffill().reindex(idx)
    valid = aligned.notna() & np.isfinite(aligned.to_numpy()) & (aligned > 0) & eq.notna() & np.isfinite(eq.to_numpy())
    if not bool(valid.any()):
        return []
    first_ts = aligned[valid].index[0]
    base_b = float(aligned.loc[first_ts])
    base_e = float(eq.loc[first_ts])
    if base_b <= 0 or not np.isfinite(base_b) or not np.isfinite(base_e):
        return []
    bench_equity = (aligned / base_b) * base_e
    bench_equity = bench_equity.where(np.isfinite(bench_equity.to_numpy()))
    out_df = pd.DataFrame({"Equity": bench_equity}, index=idx)
    out_df.index.name = strategy_equity.index.name or "Date"
    return equity_curve_to_records(out_df)


def _benchmark_block(
    bench_ticker: str,
    fin_ts_request_dict: dict[str, Any],
    strategy_equity: pd.DataFrame,
) -> dict[str, Any]:
    req = FinTsRequest.model_validate({**fin_ts_request_dict, "ticker_list": [bench_ticker]})
    bfts = build_fin_ts(req)
    df = bfts.df
    if df.empty or "Close" not in df.columns:
        return {"error": "empty_benchmark_panel"}
    if isinstance(df.index, pd.MultiIndex):
        bench_close = df.xs(bench_ticker, level=0)["Close"].astype(float).sort_index()
    else:
        bench_close = df["Close"].astype(float).sort_index()
    br = bench_close.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if strategy_equity.empty or "Equity" not in strategy_equity.columns:
        return {"error": "no_strategy_equity"}
    eq = strategy_equity["Equity"].astype(float)
    sr = eq.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    joined = pd.DataFrame({"strat": sr, "bench": br}).dropna()
    cor: float | None
    if len(joined) >= 3:
        c = float(joined["strat"].corr(joined["bench"]))
        cor = c if np.isfinite(c) else None
    else:
        cor = None

    bench_tr_pct: float | None = (
        float((bench_close.iloc[-1] / bench_close.iloc[0] - 1.0) * 100.0)
        if len(bench_close) >= 2 and float(bench_close.iloc[0]) > 0
        else None
    )

    curve = _benchmark_equity_curve_records(strategy_equity, bench_close)

    out: dict[str, Any] = {
        "ticker": bench_ticker,
        "correlation": cor,
        "n_overlap": int(len(joined)),
        "benchmark_total_return_pct": bench_tr_pct,
    }
    if curve:
        out["benchmark_equity_curve"] = curve
    return out


def run_backtest_job(
    alpha_import_ref: str | None,
    source_code: str | None,
    finstrat_stored: dict[str, Any],
    body: BacktestCreate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    algo = resolve_alpha_for_backtest(alpha_import_ref, source_code)
    fts = build_fin_ts(body.fin_ts)
    fs_kw = _merge_finstrat(finstrat_stored, body.finstrat_override)
    fs = FinStrat(fts, algo, **fs_kw)
    bt_kw = body.finbt.model_dump(mode="json", exclude_none=True)
    bt = FinBT(fs, fts, **bt_kw).run()
    out = bt.results(show=False)
    out = apply_tune_only_to_finbt_results(
        out, include_test=body.include_test_period_in_results
    )
    serialized = serialize_backtest_result(
        out,
        max_target_history=settings.max_target_history_points,
        max_group_exposure_history=settings.max_group_exposure_history_points,
    )
    if body.benchmark_ticker:
        try:
            serialized["benchmark"] = _benchmark_block(
                body.benchmark_ticker,
                body.fin_ts.model_dump(mode="json"),
                out["equity_curve"],
            )
        except Exception as exc:  # noqa: BLE001
            serialized["benchmark"] = {"error": str(exc)}
    summary = result_summary_from_metrics(serialized["metrics"])
    return serialized, summary


def run_backtest_from_payload(
    request_dict: dict[str, Any],
    alpha_import_ref: str | None,
    source_code: str | None,
    finstrat_stored: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        body = BacktestCreate.model_validate(request_dict)
    except ValidationError as exc:
        raise ValueError(f"invalid stored request: {exc}") from exc
    return run_backtest_job(alpha_import_ref, source_code, finstrat_stored, body)
