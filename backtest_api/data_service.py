from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_api.fin_ts_factory import build_fin_ts
from backtest_api.risk_metrics import periods_per_year_from_bar_spec, per_bar_return_stats_with_ppy
from backtest_api.schemas.models import DataSummaryRequest, DataSummaryResponse, TickerNanRow, TickerRiskRow


def compute_data_summary(req: DataSummaryRequest) -> DataSummaryResponse:
    fts = build_fin_ts(req)
    df = fts.df
    if df.empty:
        raise ValueError("fin_ts panel is empty for the requested range.")

    tickers = list(fts.ticker_list)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if req.columns is not None:
        cols = [c for c in req.columns if c in df.columns and c in numeric_cols]
        if not cols:
            cols = numeric_cols
    else:
        cols = numeric_cols

    nan_rows: list[TickerNanRow] = []
    if isinstance(df.index, pd.MultiIndex) and df.index.names[0] == "Ticker":
        for t in tickers:
            try:
                sub = df.xs(t, level=0)
            except KeyError:
                nan_rows.append(TickerNanRow(ticker=t, nan_counts={c: 0 for c in cols}))
                continue
            counts = {c: int(sub[c].isna().sum()) for c in cols if c in sub.columns}
            nan_rows.append(TickerNanRow(ticker=t, nan_counts=counts))
    else:
        counts = {c: int(df[c].isna().sum()) for c in cols if c in df.columns}
        nan_rows.append(TickerNanRow(ticker=tickers[0], nan_counts=counts))

    ppy = periods_per_year_from_bar_spec(fts.bar_spec)
    risk_rows: list[TickerRiskRow] = []
    if isinstance(df.index, pd.MultiIndex) and df.index.names[0] == "Ticker":
        for t in tickers:
            try:
                sub = df.xs(t, level=0)
            except KeyError:
                risk_rows.append(
                    TickerRiskRow(ticker=t, return_pct=None, risk_ann_pct=None, sharpe=None, sortino=None)
                )
                continue
            if "Close" not in sub.columns:
                risk_rows.append(
                    TickerRiskRow(ticker=t, return_pct=None, risk_ann_pct=None, sharpe=None, sortino=None)
                )
                continue
            close = sub["Close"].sort_index()
            ret_pct, risk_ann, sharpe, sortino = per_bar_return_stats_with_ppy(close, ppy)
            risk_rows.append(
                TickerRiskRow(
                    ticker=t,
                    return_pct=ret_pct,
                    risk_ann_pct=risk_ann,
                    sharpe=sharpe,
                    sortino=sortino,
                )
            )
    else:
        close = df["Close"].sort_index() if "Close" in df.columns else pd.Series(dtype=float)
        ret_pct, risk_ann, sharpe, sortino = per_bar_return_stats_with_ppy(close, ppy)
        risk_rows.append(
            TickerRiskRow(
                ticker=tickers[0],
                return_pct=ret_pct,
                risk_ann_pct=risk_ann,
                sharpe=sharpe,
                sortino=sortino,
            )
        )

    return DataSummaryResponse(
        tickers=tickers,
        columns_used=cols,
        nan_counts=nan_rows,
        per_ticker_metrics=risk_rows,
        bar_unit=str(fts.bar_spec.unit),
        bar_step=int(fts.bar_spec.step),
        periods_per_year=float(ppy),
    )
