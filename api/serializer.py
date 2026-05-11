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
        ts = pd.Timestamp(x)
        return None if pd.isna(ts) else ts.isoformat()
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
    try:
        if pd.api.types.is_scalar(x) and bool(pd.isna(x)):
            return None
    except (TypeError, ValueError):
        pass
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


def _serialize_group_exposure_history(
    raw_list: Any, *, max_points: int
) -> list[dict[str, Any]]:
    if not isinstance(raw_list, list) or not raw_list:
        return []
    tail = raw_list[-max_points:] if len(raw_list) > max_points else raw_list
    out: list[dict[str, Any]] = []
    for item in tail:
        if isinstance(item, tuple) and len(item) == 2:
            dt, payload = item
            if not isinstance(payload, dict):
                continue
            gross = payload.get("gross_by_group")
            net = payload.get("net_by_group")
            if not isinstance(gross, dict):
                gross = {}
            if not isinstance(net, dict):
                net = {}
            out.append(
                {
                    "date": pd.Timestamp(dt).isoformat(),
                    "gross_by_group": {str(k): _json_scalar(v) for k, v in gross.items()},
                    "net_by_group": {str(k): _json_scalar(v) for k, v in net.items()},
                }
            )
    return out


def _serialize_exposure_history(df: Any, *, max_points: int) -> list[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    tail = df.tail(max_points) if len(df) > max_points else df
    return turnover_to_records(tail)


def _serialize_trade_events(raw_events: Any, *, max_events: int) -> list[dict[str, Any]]:
    if not isinstance(raw_events, list) or not raw_events:
        return []
    tail = raw_events[-max_events:] if len(raw_events) > max_events else raw_events
    out: list[dict[str, Any]] = []
    for ev in tail:
        if not isinstance(ev, dict):
            continue
        ts = ev.get("ts")
        if isinstance(ts, pd.Timestamp):
            ts_iso = None if pd.isna(ts) else ts.isoformat()
        else:
            ts_iso = _json_scalar(ts)
        out.append(
            {
                "ts": ts_iso,
                "ticker": _json_scalar(ev.get("ticker")),
                "side": _json_scalar(ev.get("side")),
                "size": _json_scalar(ev.get("size")),
                "price": _json_scalar(ev.get("price")),
                "value": _json_scalar(ev.get("value")),
            }
        )
    return out


def serialize_backtest_result(
    raw: dict[str, Any],
    *,
    max_target_history: int = 500,
    max_group_exposure_history: int = 500,
    max_exposure_history: int = 500,
    max_trade_events: int = 2000,
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

    group_exp = _serialize_group_exposure_history(
        raw.get("group_exposure_history"),
        max_points=max_group_exposure_history,
    )

    tpct = raw.get("turnover_pct_history")
    turnover_pct_records = turnover_to_records(tpct) if isinstance(tpct, pd.DataFrame) else []

    exposure_records = _serialize_exposure_history(
        raw.get("exposure_history"),
        max_points=max_exposure_history,
    )

    trade_ser = _serialize_trade_events(raw.get("trade_events"), max_events=max_trade_events)

    return {
        "metrics": metrics,
        "equity_curve": equity_records,
        "turnover_history": turnover_records,
        "turnover_pct_history": turnover_pct_records,
        "returns_analysis": _json_scalar(raw.get("returns_analysis")),
        "drawdown_analysis": _json_scalar(raw.get("drawdown_analysis")),
        "sharpe_analysis": _json_scalar(raw.get("sharpe_analysis")),
        "target_history": target_ser,
        "group_exposure_history": group_exp,
        "exposure_history": exposure_records,
        "trade_events": trade_ser,
        "return_quantiles": _json_scalar(raw.get("return_quantiles")),
        "tearsheet_summary": _json_scalar(raw.get("tearsheet_summary")),
        "ff_single_factor": _json_scalar(raw.get("ff_single_factor")),
    }


def result_summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_return_pct",
        "cagr_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "end_value",
        "bar_unit",
        "bar_step",
    )
    return {k: metrics[k] for k in keys if k in metrics}
