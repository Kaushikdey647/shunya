"""Serialize yfinance pandas objects and parse key-statistics valuation tables."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from api.schemas.models import InstrumentValuationMetrics


def _jsonable_scalar(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    if hasattr(val, "isoformat") and callable(getattr(val, "isoformat")):
        try:
            return val.isoformat()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            return str(val)
    if isinstance(val, (pd.Timestamp,)):
        try:
            return val.isoformat()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            return str(val)
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


def dataframe_to_records(df: pd.DataFrame | None) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (columns, records) suitable for JSON / Pydantic models."""
    if df is None or df.empty:
        return [], []
    safe = df.reset_index()
    safe = safe.where(pd.notnull(safe), None)
    cols = [str(c) for c in safe.columns]
    records: list[dict[str, Any]] = []
    for _, row in safe.iterrows():
        rec: dict[str, Any] = {}
        for c in cols:
            rec[c] = _jsonable_scalar(row.get(c))
        records.append(rec)
    return cols, records


def _norm_metric_label(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    return re.sub(r"\s+", " ", s)


def _parse_measure_scalar(cell: Any) -> float | None:
    if cell is None:
        return None
    if isinstance(cell, bool):
        return None
    if isinstance(cell, (int, float)):
        if isinstance(cell, float) and (math.isnan(cell) or math.isinf(cell)):
            return None
        return float(cell)
    if not isinstance(cell, str):
        return None
    s = cell.strip().lower()
    if not s or s in ("n/a", "—", "-", "nan", "none", "null"):
        return None
    mult = 1.0
    if s.endswith("t"):
        mult = 1e12
        s = s[:-1].strip()
    elif s.endswith("b"):
        mult = 1e9
        s = s[:-1].strip()
    elif s.endswith("m") and re.match(r"^-?[\d.,]+\s*m$", s):
        mult = 1e6
        s = s[:-1].strip()
    is_pct = "%" in s
    s2 = re.sub(r"[%\s,]", "", s)
    try:
        v = float(s2) * mult
    except ValueError:
        return None
    if is_pct:
        return v / 100.0
    return v


def valuation_measures_to_metrics(df: pd.DataFrame | None) -> InstrumentValuationMetrics:
    """Map Yahoo key-statistics table (index = metric label) into InstrumentValuationMetrics."""
    out = InstrumentValuationMetrics()
    if df is None or df.empty:
        return out
    # Prefer first numeric column (often current / TTM)
    val_col = df.columns[0] if len(df.columns) > 0 else None
    if val_col is None:
        return out

    def row_val(label: str) -> float | None:
        try:
            if label not in df.index:
                return None
            raw = df.loc[label, val_col]
        except Exception:  # noqa: BLE001
            return None
        return _parse_measure_scalar(raw)

    # Exact index match first (stable when Yahoo keeps names)
    exact_map: dict[str, tuple[str, ...]] = {
        "trailing_pe": ("Trailing P/E", "PE Ratio (TTM)", "P/E (TTM)"),
        "forward_pe": ("Forward P/E", "Forward P/E "),
        "trailing_eps": ("EPS (TTM)", "Diluted EPS (ttm)"),
        "forward_eps": ("Forward EPS", "Forward Earnings"),
        "return_on_equity": ("Return on Equity (ttm)", "Return on Equity"),
        "return_on_assets": ("Return on Assets (ttm)", "Return on Assets"),
        "price_to_book": ("Price/Book (mrq)", "P/B"),
        "price_to_sales": ("Price/Sales (ttm)", "P/S"),
        "debt_to_equity": ("Total Debt/Equity (mrq)", "Debt/Equity"),
    }
    for field, candidates in exact_map.items():
        for cand in candidates:
            if cand in df.index:
                v = row_val(cand)
                if v is not None:
                    setattr(out, field, v)
                    break

    # Fuzzy fallback on normalized index labels
    idx_map: dict[str, str] = {}
    for idx in df.index:
        idx_map[_norm_metric_label(str(idx))] = str(idx)

    def fuzzy_pick(*needles: str) -> float | None:
        for norm_label, orig in idx_map.items():
            if all(n in norm_label for n in needles):
                return row_val(orig)
        return None

    if out.trailing_pe is None:
        for norm_label, orig in idx_map.items():
            if "forward" in norm_label:
                continue
            if ("trailing" in norm_label and "p" in norm_label and "e" in norm_label) or (
                "pe" in norm_label and "ttm" in norm_label
            ):
                v = row_val(orig)
                if v is not None:
                    out.trailing_pe = v
                    break
    if out.forward_pe is None:
        out.forward_pe = fuzzy_pick("forward", "p", "e")
    if out.trailing_eps is None:
        out.trailing_eps = fuzzy_pick("eps", "ttm") or fuzzy_pick("diluted", "eps")
    if out.forward_eps is None:
        out.forward_eps = fuzzy_pick("forward", "eps")
    if out.return_on_equity is None:
        v = fuzzy_pick("return", "equity")
        if v is not None and v > 2.0:
            v = v / 100.0
        out.return_on_equity = v
    if out.return_on_assets is None:
        v = fuzzy_pick("return", "assets")
        if v is not None and v > 2.0:
            v = v / 100.0
        out.return_on_assets = v
    if out.price_to_book is None:
        out.price_to_book = fuzzy_pick("price", "book") or fuzzy_pick("p", "b")
    if out.price_to_sales is None:
        out.price_to_sales = fuzzy_pick("price", "sales") or fuzzy_pick("p", "s")
    if out.debt_to_equity is None:
        out.debt_to_equity = fuzzy_pick("debt", "equity")

    return out


def beta_from_valuation_measures(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    val_col = df.columns[0]
    for idx in df.index:
        n = _norm_metric_label(str(idx))
        if n.startswith("beta") or " beta" in n:
            try:
                raw = df.loc[idx, val_col]
            except Exception:  # noqa: BLE001
                continue
            b = _parse_measure_scalar(raw)
            if b is not None:
                return b
    return None


def merge_valuation_metrics(
    info_metrics: InstrumentValuationMetrics,
    measures_metrics: InstrumentValuationMetrics,
) -> InstrumentValuationMetrics:
    """Prefer key-statistics table values when present; otherwise keep ``get_info()`` metrics."""

    def pick(a: float | None, b: float | None) -> float | None:
        return b if b is not None else a

    return InstrumentValuationMetrics(
        trailing_pe=pick(info_metrics.trailing_pe, measures_metrics.trailing_pe),
        forward_pe=pick(info_metrics.forward_pe, measures_metrics.forward_pe),
        trailing_eps=pick(info_metrics.trailing_eps, measures_metrics.trailing_eps),
        forward_eps=pick(info_metrics.forward_eps, measures_metrics.forward_eps),
        return_on_equity=pick(info_metrics.return_on_equity, measures_metrics.return_on_equity),
        return_on_assets=pick(info_metrics.return_on_assets, measures_metrics.return_on_assets),
        price_to_book=pick(info_metrics.price_to_book, measures_metrics.price_to_book),
        price_to_sales=pick(info_metrics.price_to_sales, measures_metrics.price_to_sales),
        debt_to_equity=pick(info_metrics.debt_to_equity, measures_metrics.debt_to_equity),
    )


def dict_to_jsonable(d: dict[str, Any] | None) -> dict[str, Any]:
    if not d:
        return {}
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[str(k)] = dict_to_jsonable(v)  # type: ignore[arg-type]
        elif isinstance(v, list):
            out[str(k)] = [_jsonable_scalar(x) if not isinstance(x, dict) else dict_to_jsonable(x) for x in v]  # type: ignore[misc]
        else:
            out[str(k)] = _jsonable_scalar(v)
    return out
