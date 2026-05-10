"""Normalize yfinance / API payloads into rows for wide fundamentals and event tables."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..fundamentals import (
    DAILY_FUNDAMENTAL_FIELDS,
    FUNDAMENTAL_FIELDS,
    validate_daily_fundamental_fields,
)

# DB column order for fundamentals_quarterly / fundamentals_annual (snake_case)
_FUNDAMENTAL_SQL_COLS: tuple[str, ...] = (
    "revenue",
    "net_income",
    "eps_diluted",
    "operating_cash_flow",
    "free_cash_flow",
    "total_assets",
    "total_equity",
    "total_debt",
    "current_ratio",
    "gross_margin",
    "operating_margin",
    "return_on_assets",
    "return_on_equity",
    "debt_to_equity",
    "free_cash_flow_yield",
    "price_to_earnings",
)

_PY_TO_SQL: dict[str, str] = {py: sql for py, sql in zip(FUNDAMENTAL_FIELDS, _FUNDAMENTAL_SQL_COLS, strict=True)}
SQL_TO_PY_PERIODIC: dict[str, str] = {sql: py for py, sql in _PY_TO_SQL.items()}
FUNDAMENTAL_SQL_COLS: tuple[str, ...] = _FUNDAMENTAL_SQL_COLS

DAILY_FIELD_TO_SQL: dict[str, str] = {
    "Market_Cap": "market_cap",
    "Enterprise_Value": "enterprise_value",
    "Trailing_PE": "trailing_pe",
    "Forward_PE": "forward_pe",
    "PEG_Ratio": "peg_ratio",
    "Price_To_Book": "price_to_book",
    "Dividend_Yield": "dividend_yield",
    "Beta": "beta",
    "Shares_Outstanding": "shares_outstanding",
}


def periodic_frame_to_wide_rows(
    periodic: pd.DataFrame,
    ticker_to_id: Mapping[str, int],
    *,
    source: str,
) -> list[tuple[Any, ...]]:
    """One upsert row per (symbol, fiscal_period_end) from a (Ticker, Date) wide periodic frame."""
    if periodic.empty:
        return []
    if tuple(periodic.index.names) != ("Ticker", "Date"):
        raise ValueError(f"expected MultiIndex ('Ticker', 'Date'), got {periodic.index.names!r}")
    rows: list[tuple[Any, ...]] = []
    for (ticker, dt), ser in periodic.iterrows():
        sid = ticker_to_id.get(str(ticker))
        if sid is None:
            continue
        pe = pd.Timestamp(dt).date()
        vals: list[Any] = []
        for py_col in FUNDAMENTAL_FIELDS:
            v = ser.get(py_col)
            if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(float(v))):
                vals.append(None)
            else:
                vals.append(float(v))
        rows.append((sid, pe, str(source), *vals))
    return rows


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def _float_opt(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.floating)):
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return None


def _parse_ts(val: Any) -> datetime | None:
    if val is None:
        return None
    try:
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return ts.to_pydatetime()
    except Exception:
        return None


def valuation_measures_to_daily_rows(
    symbol_id: int,
    *,
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    source: str,
) -> list[tuple[Any, ...]]:
    """
    Map ``InstrumentValuationMeasuresPayload``-style columns/records to ``fundamentals_daily`` rows.

    Yahoo column names vary; we match on normalized header tokens per record key.
    """
    _ = columns
    _metric_keys: dict[str, str] = {
        "marketcap": "market_cap",
        "market_cap": "market_cap",
        "enterprisevalue": "enterprise_value",
        "enterprise_value": "enterprise_value",
        "trailingpe": "trailing_pe",
        "trailing_pe": "trailing_pe",
        "forwardpe": "forward_pe",
        "forward_pe": "forward_pe",
        "pegratio": "peg_ratio",
        "peg_ratio": "peg_ratio",
        "pricetobook": "price_to_book",
        "price_to_book": "price_to_book",
        "dividendyield": "dividend_yield",
        "dividend_yield": "dividend_yield",
        "beta": "beta",
        "sharesoutstanding": "shares_outstanding",
        "ordinarysharesnumber": "shares_outstanding",
    }
    _date_keys = frozenset({"date", "periodend", "period_end", "index"})

    rows: list[tuple[Any, ...]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        vals_by_metric: dict[str, float] = {}
        as_of: datetime | None = None
        for k, v in rec.items():
            nk = _norm_key(str(k))
            if nk in _date_keys:
                as_of = _parse_ts(v)
                continue
            metric = _metric_keys.get(nk)
            if metric is None:
                continue
            try:
                fv = float(pd.to_numeric(v, errors="coerce"))
            except (TypeError, ValueError):
                fv = float("nan")
            if np.isfinite(fv):
                vals_by_metric[metric] = fv
        if as_of is None:
            continue

        def pick(name: str) -> float | None:
            return vals_by_metric.get(name)

        rows.append(
            (
                symbol_id,
                as_of,
                str(source),
                pick("market_cap"),
                pick("enterprise_value"),
                pick("trailing_pe"),
                pick("forward_pe"),
                pick("peg_ratio"),
                pick("price_to_book"),
                pick("dividend_yield"),
                pick("beta"),
                pick("shares_outstanding"),
            )
        )
    return rows


def yfinance_dividends_splits_to_corporate_actions(
    symbol_id: int,
    *,
    dividends: pd.Series | None,
    splits: pd.Series | None,
    source: str,
) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    if dividends is not None and len(dividends):
        for ts, amt in dividends.items():
            tdt = _parse_ts(ts)
            if tdt is None:
                continue
            fv = _float_opt(amt)
            if fv is None:
                continue
            raw = json.dumps({"amount": fv})
            rows.append((symbol_id, tdt, "dividend", fv, None, None, str(source), raw))
    if splits is not None and len(splits):
        for ts, ratio in splits.items():
            tdt = _parse_ts(ts)
            if tdt is None:
                continue
            fv = _float_opt(ratio)
            if fv is None:
                continue
            raw = json.dumps({"split_ratio": fv})
            rows.append((symbol_id, tdt, "split", None, fv, None, str(source), raw))
    return rows


def _fingerprint_insider_row(parts: Sequence[str]) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8"))
    return h.hexdigest()


def insider_table_to_rows(
    symbol_id: int,
    *,
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    source: str,
) -> list[tuple[Any, ...]]:
    """Map yfinance insider_transactions table payload to upsert tuples."""
    if not records:
        return []
    col_l = [str(c) for c in columns]
    norm_map = {_norm_key(c): c for c in col_l}

    def col(*candidates: str) -> str | None:
        for cand in candidates:
            k = _norm_key(cand)
            if k in norm_map:
                return norm_map[k]
        return None

    c_start = col("startdate", "transactionstartdate", "start")
    c_date = col("date", "reportdate", "positiondate")
    c_owner = col("insider", "name", "owner", "filername")
    c_type = col("transaction", "transactiontype", "type")
    c_shares = col("shares", "share")
    c_value = col("value", "securitiesowned", "positionvalue")
    c_pos = col("position", "positiontext", "filerrelation")

    rows: list[tuple[Any, ...]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue

        def get_cell(key: str | None) -> Any:
            if not key:
                return None
            return rec.get(key)

        start_raw = get_cell(c_start)
        date_raw = get_cell(c_date)
        owner = get_cell(c_owner)
        tx = get_cell(c_type)
        shares = _float_opt(get_cell(c_shares))
        value = _float_opt(get_cell(c_value))
        pos = get_cell(c_pos)
        owner_s = str(owner).strip() if owner is not None else ""
        tx_s = str(tx).strip() if tx is not None else ""
        sd = pd.Timestamp(start_raw).date() if start_raw is not None and not pd.isna(pd.Timestamp(start_raw)) else None
        rd = pd.Timestamp(date_raw).date() if date_raw is not None and not pd.isna(pd.Timestamp(date_raw)) else None
        parts = [
            str(rd or ""),
            str(sd or ""),
            owner_s,
            tx_s,
            "" if shares is None else f"{shares:.6g}",
            "" if value is None else f"{value:.6g}",
            str(pos or "").strip(),
        ]
        fp = _fingerprint_insider_row(parts)
        if rd is None and sd is None and not owner_s and not tx_s:
            continue
        rows.append(
            (
                symbol_id,
                rd,
                sd,
                owner_s or None,
                tx_s or None,
                shares,
                value,
                str(pos).strip() if pos is not None and str(pos).strip() else None,
                str(source),
                fp,
            )
        )
    return rows


def earnings_dates_dataframe_to_rows(
    symbol_id: int,
    df: pd.DataFrame | None,
    *,
    source: str,
) -> list[tuple[Any, ...]]:
    if df is None or df.empty:
        return []
    rows: list[tuple[Any, ...]] = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            return []
    cols_lower = {str(c).lower(): c for c in df.columns}
    def pick(*names: str) -> str | None:
        for n in names:
            if n.lower() in cols_lower:
                return str(cols_lower[n.lower()])
        return None
    c_est = pick("eps estimate", "eps average", "epsavg")
    c_rep = pick("reported eps", "eps actual")
    c_sur = pick("surprise(%)", "surprise %", "surprise")

    for ts, ser in df.iterrows():
        ed = pd.Timestamp(ts).date()
        est = _float_opt(ser.get(c_est)) if c_est else None
        rep = _float_opt(ser.get(c_rep)) if c_rep else None
        sur = _float_opt(ser.get(c_sur)) if c_sur else None
        rows.append((symbol_id, ed, str(source), est, rep, sur, None))
    return rows


def dataframe_to_columns_records(df: pd.DataFrame | None) -> tuple[list[str], list[dict[str, Any]]]:
    """Lightweight ``reset_index`` → (columns, records) for valuation / insider frames."""
    if df is None or df.empty:
        return [], []
    safe = df.reset_index()
    safe = safe.where(pd.notnull(safe), None)
    cols = [str(c) for c in safe.columns]
    records: list[dict[str, Any]] = []
    for _, row in safe.iterrows():
        rec = {c: row.get(c) for c in cols}
        records.append(rec)
    return cols, records


def calendar_dict_to_earnings_rows(
    symbol_id: int,
    data: Mapping[str, Any],
    *,
    source: str,
) -> list[tuple[Any, ...]]:
    """Best-effort parse of yfinance ``get_calendar()`` dict into ``earnings_dates`` rows."""
    rows: list[tuple[Any, ...]] = []
    if not data:
        return rows
    # Single upcoming earnings: string or timestamp
    ed_raw = data.get("Earnings Date") or data.get("earningsDate")
    if ed_raw is not None and not isinstance(ed_raw, (list, dict)):
        d = pd.Timestamp(ed_raw).date()
        est = _float_opt(data.get("Earnings Average") or data.get("earningsAverage"))
        low = _float_opt(data.get("Earnings Low") or data.get("earningsLow"))
        high = _float_opt(data.get("Earnings High") or data.get("earningsHigh"))
        q = data.get("Quarter") or data.get("quarter")
        q_lab = str(q) if q is not None else None
        if est is None and low is not None and high is not None:
            est = (low + high) / 2.0
        rows.append((symbol_id, d, str(source), est, None, None, q_lab))
    return rows


def ticker_info_to_daily_row(
    symbol_id: int,
    info: Mapping[str, Any],
    *,
    as_of: datetime,
    source: str,
) -> tuple[Any, ...]:
    """Single-row snapshot from ``ticker.info`` for daily fundamentals."""
    def g(*keys: str) -> float | None:
        for k in keys:
            if k in info:
                return _float_opt(info.get(k))
        return None

    return (
        symbol_id,
        as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc),
        str(source),
        g("marketCap"),
        g("enterpriseValue"),
        g("trailingPE"),
        g("forwardPE"),
        g("pegRatio"),
        g("priceToBook"),
        g("dividendYield"),
        g("beta"),
        g("sharesOutstanding"),
    )


__all__ = [
    "DAILY_FUNDAMENTAL_FIELDS",
    "DAILY_FIELD_TO_SQL",
    "FUNDAMENTAL_SQL_COLS",
    "SQL_TO_PY_PERIODIC",
    "calendar_dict_to_earnings_rows",
    "dataframe_to_columns_records",
    "earnings_dates_dataframe_to_rows",
    "insider_table_to_rows",
    "periodic_frame_to_wide_rows",
    "ticker_info_to_daily_row",
    "validate_daily_fundamental_fields",
    "valuation_measures_to_daily_rows",
    "yfinance_dividends_splits_to_corporate_actions",
]
