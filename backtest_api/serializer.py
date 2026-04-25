from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _json_scalar(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (bool, str)):
        return x
    if isinstance(x, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(x).isoformat()
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _json_scalar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_scalar(v) for v in x]
    return str(x)


def equity_curve_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    tmp = df.reset_index()
    records: list[dict[str, Any]] = []
    for _, row in tmp.iterrows():
        rec: dict[str, Any] = {}
        for col in tmp.columns:
            v = row[col]
            if isinstance(v, pd.Timestamp):
                rec[str(col)] = v.isoformat()
            else:
                rec[str(col)] = _json_scalar(v)
        records.append(rec)
    return records


def turnover_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for dt, row in df.iterrows():
        item: dict[str, Any] = {"date": pd.Timestamp(dt).isoformat()}
        for c in df.columns:
            item[str(c)] = _json_scalar(row[c])
        out.append(item)
    return out


def serialize_backtest_result(
    raw: dict[str, Any],
    *,
    max_target_history: int = 500,
) -> dict[str, Any]:
    metrics = {k: _json_scalar(v) for k, v in raw["metrics"].items()}
    eq = raw["equity_curve"]
    equity_records = equity_curve_to_records(eq) if isinstance(eq, pd.DataFrame) else []
    to = raw.get("turnover_history")
    turnover_records = turnover_to_records(to) if isinstance(to, pd.DataFrame) else []

    th = raw.get("target_history") or []
    if isinstance(th, list) and len(th) > max_target_history:
        th = th[-max_target_history:]

    target_ser: list[Any] = []
    for item in th:
        if isinstance(item, tuple) and len(item) == 2:
            dt, mp = item
            target_ser.append(
                {"date": pd.Timestamp(dt).isoformat(), "targets": {k: _json_scalar(v) for k, v in mp.items()}}
            )
        else:
            target_ser.append(_json_scalar(item))

    return {
        "metrics": metrics,
        "equity_curve": equity_records,
        "turnover_history": turnover_records,
        "returns_analysis": _json_scalar(raw.get("returns_analysis")),
        "drawdown_analysis": _json_scalar(raw.get("drawdown_analysis")),
        "sharpe_analysis": _json_scalar(raw.get("sharpe_analysis")),
        "target_history": target_ser,
    }


def result_summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "end_value",
        "bar_unit",
        "bar_step",
    )
    return {k: metrics[k] for k in keys if k in metrics}
