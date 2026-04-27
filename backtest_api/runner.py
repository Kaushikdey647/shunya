from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError

from shunya.algorithm.finbt import FinBT
from shunya.algorithm.finstrat import FinStrat

from backtest_api.fin_ts_factory import build_fin_ts
from backtest_api.resolver import resolve_alpha_for_backtest
from backtest_api.result_tune_filter import apply_tune_only_to_finbt_results
from backtest_api.schemas.models import BacktestCreate, FinStratConfig
from backtest_api.serializer import result_summary_from_metrics, serialize_backtest_result
from backtest_api.settings import get_settings


def _merge_finstrat(stored: dict[str, Any], override: Optional[FinStratConfig]) -> dict[str, Any]:
    base = FinStratConfig.model_validate(stored).model_dump(mode="json", exclude_none=True)
    if override is not None:
        base.update(override.model_dump(mode="json", exclude_none=True))
    return FinStratConfig.model_validate(base).model_dump(mode="json", exclude_none=True)


def _benchmark_block(
    bench_ticker: str,
    fin_ts_request_dict: dict[str, Any],
    strategy_equity: pd.DataFrame,
) -> dict[str, Any]:
    from backtest_api.schemas.models import FinTsRequest

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
    if len(joined) < 3:
        return {"correlation": None, "n_overlap": int(len(joined))}
    cor = float(joined["strat"].corr(joined["bench"]))
    return {
        "ticker": bench_ticker,
        "correlation": cor if np.isfinite(cor) else None,
        "n_overlap": int(len(joined)),
        "benchmark_total_return_pct": float((bench_close.iloc[-1] / bench_close.iloc[0] - 1.0) * 100.0)
        if len(bench_close) >= 2 and float(bench_close.iloc[0]) > 0
        else None,
    }


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
