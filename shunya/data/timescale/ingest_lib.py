"""Upsert helpers for bootstrap CLIs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

_NAME_MAX = 2048


def ensure_symbols(
    cur: Any,
    tickers: Sequence[str],
    *,
    display_names: Mapping[str, str | None] | None = None,
) -> dict[str, int]:
    """Upsert tickers into ``symbols``; return mapping ``ticker -> symbol_id``.

    When ``display_names`` is set, non-empty values update ``symbols.name`` on insert/conflict.
    """
    out: dict[str, int] = {}
    dn = display_names or {}
    for t in tickers:
        tick = str(t)
        raw_nm = dn.get(tick)
        nm: str | None = None
        if raw_nm is not None:
            s = str(raw_nm).strip()
            if s:
                nm = s[:_NAME_MAX]
        if nm:
            cur.execute(
                """
                INSERT INTO symbols (ticker, name) VALUES (%s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    name = COALESCE(NULLIF(EXCLUDED.name, ''), symbols.name)
                RETURNING id
                """,
                (tick, nm),
            )
        else:
            cur.execute(
                """
                INSERT INTO symbols (ticker) VALUES (%s)
                ON CONFLICT (ticker) DO UPDATE SET ticker = EXCLUDED.ticker
                RETURNING id
                """,
                (tick,),
            )
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT id FROM symbols WHERE ticker = %s", (tick,))
            row = cur.fetchone()
        out[tick] = int(row[0])
    return out


def rows_from_provider_ohlcv(
    df: pd.DataFrame,
    ticker_to_id: Mapping[str, int],
    *,
    interval: str,
    source: str,
) -> list[tuple[Any, ...]]:
    """Flatten a ``MarketDataProvider`` OHLCV frame into DB rows."""
    rows: list[tuple[Any, ...]] = []
    if df.empty:
        return rows

    if isinstance(df.columns, pd.MultiIndex):
        tickers = [t for t in df.columns.get_level_values(0).unique() if str(t) in ticker_to_id]
        for t in tickers:
            part = df[str(t)].copy()
            _append_ticker_bars(rows, part, ticker_to_id[str(t)], interval, source)
    else:
        if len(ticker_to_id) != 1:
            raise ValueError("single-level OHLCV frame requires exactly one ticker mapping")
        tid = next(iter(ticker_to_id.values()))
        _append_ticker_bars(rows, df.copy(), tid, interval, source)
    return rows


def _append_ticker_bars(
    rows: list[tuple[Any, ...]],
    part: pd.DataFrame,
    symbol_id: int,
    interval: str,
    source: str,
) -> None:
    need = ("Open", "High", "Low", "Close", "Volume")
    miss = [c for c in need if c not in part.columns]
    if miss:
        raise KeyError(f"OHLCV frame missing columns {miss}")
    part = part.sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(part.index))
    for i in range(len(part)):
        ts = idx[i]
        o = float(pd.to_numeric(part["Open"].iloc[i], errors="coerce"))
        h = float(pd.to_numeric(part["High"].iloc[i], errors="coerce"))
        l = float(pd.to_numeric(part["Low"].iloc[i], errors="coerce"))
        c = float(pd.to_numeric(part["Close"].iloc[i], errors="coerce"))
        v = float(pd.to_numeric(part["Volume"].iloc[i], errors="coerce"))
        if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l) and np.isfinite(c) and np.isfinite(v)):
            continue
        if v < 0.0:
            continue
        rows.append((symbol_id, ts.to_pydatetime(), interval, o, h, l, c, v, source, None))


UPSERT_OHLCV_SQL = """
INSERT INTO ohlcv_bars (symbol_id, ts, interval, open, high, low, close, volume, source, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, ts, interval, source) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    ingested_at = now(),
    metadata = COALESCE(EXCLUDED.metadata, ohlcv_bars.metadata)
"""


def fundamentals_eav_rows(
    periodic: pd.DataFrame,
    ticker_to_id: Mapping[str, int],
    *,
    freq: str,
    source: str,
) -> list[tuple[Any, ...]]:
    """Explode a wide periodic ``(Ticker, Date)`` frame into EAV rows."""
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
        for field, val in ser.items():
            if pd.isna(val):
                continue
            fv = float(val)
            rows.append((sid, pe, str(freq), str(field), fv, str(source)))
    return rows


UPSERT_FUND_SQL = """
INSERT INTO fundamentals_field_values (symbol_id, period_end, freq, field, value, source)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, period_end, freq, field, source) DO UPDATE SET
    value = EXCLUDED.value,
    ingested_at = now()
"""

UPSERT_FUND_DAILY_SQL = """
INSERT INTO fundamentals_daily (
    symbol_id, as_of_ts, source,
    market_cap, enterprise_value, trailing_pe, forward_pe, peg_ratio, price_to_book,
    dividend_yield, beta, shares_outstanding
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, as_of_ts, source) DO UPDATE SET
    market_cap = EXCLUDED.market_cap,
    enterprise_value = EXCLUDED.enterprise_value,
    trailing_pe = EXCLUDED.trailing_pe,
    forward_pe = EXCLUDED.forward_pe,
    peg_ratio = EXCLUDED.peg_ratio,
    price_to_book = EXCLUDED.price_to_book,
    dividend_yield = EXCLUDED.dividend_yield,
    beta = EXCLUDED.beta,
    shares_outstanding = EXCLUDED.shares_outstanding,
    ingested_at = now()
"""

UPSERT_FUND_QUARTERLY_SQL = """
INSERT INTO fundamentals_quarterly (
    symbol_id, fiscal_period_end, source,
    revenue, net_income, eps_diluted, operating_cash_flow, free_cash_flow,
    total_assets, total_equity, total_debt, current_ratio,
    gross_margin, operating_margin, return_on_assets, return_on_equity,
    debt_to_equity, free_cash_flow_yield, price_to_earnings
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, fiscal_period_end, source) DO UPDATE SET
    revenue = EXCLUDED.revenue,
    net_income = EXCLUDED.net_income,
    eps_diluted = EXCLUDED.eps_diluted,
    operating_cash_flow = EXCLUDED.operating_cash_flow,
    free_cash_flow = EXCLUDED.free_cash_flow,
    total_assets = EXCLUDED.total_assets,
    total_equity = EXCLUDED.total_equity,
    total_debt = EXCLUDED.total_debt,
    current_ratio = EXCLUDED.current_ratio,
    gross_margin = EXCLUDED.gross_margin,
    operating_margin = EXCLUDED.operating_margin,
    return_on_assets = EXCLUDED.return_on_assets,
    return_on_equity = EXCLUDED.return_on_equity,
    debt_to_equity = EXCLUDED.debt_to_equity,
    free_cash_flow_yield = EXCLUDED.free_cash_flow_yield,
    price_to_earnings = EXCLUDED.price_to_earnings,
    ingested_at = now()
"""

UPSERT_FUND_ANNUAL_SQL = """
INSERT INTO fundamentals_annual (
    symbol_id, fiscal_period_end, source,
    revenue, net_income, eps_diluted, operating_cash_flow, free_cash_flow,
    total_assets, total_equity, total_debt, current_ratio,
    gross_margin, operating_margin, return_on_assets, return_on_equity,
    debt_to_equity, free_cash_flow_yield, price_to_earnings
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, fiscal_period_end, source) DO UPDATE SET
    revenue = EXCLUDED.revenue,
    net_income = EXCLUDED.net_income,
    eps_diluted = EXCLUDED.eps_diluted,
    operating_cash_flow = EXCLUDED.operating_cash_flow,
    free_cash_flow = EXCLUDED.free_cash_flow,
    total_assets = EXCLUDED.total_assets,
    total_equity = EXCLUDED.total_equity,
    total_debt = EXCLUDED.total_debt,
    current_ratio = EXCLUDED.current_ratio,
    gross_margin = EXCLUDED.gross_margin,
    operating_margin = EXCLUDED.operating_margin,
    return_on_assets = EXCLUDED.return_on_assets,
    return_on_equity = EXCLUDED.return_on_equity,
    debt_to_equity = EXCLUDED.debt_to_equity,
    free_cash_flow_yield = EXCLUDED.free_cash_flow_yield,
    price_to_earnings = EXCLUDED.price_to_earnings,
    ingested_at = now()
"""

UPSERT_CORPORATE_ACTIONS_SQL = """
INSERT INTO corporate_actions (
    symbol_id, action_ts, kind, amount, split_ratio, currency, source, raw
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (symbol_id, action_ts, kind, source) DO UPDATE SET
    amount = EXCLUDED.amount,
    split_ratio = EXCLUDED.split_ratio,
    currency = EXCLUDED.currency,
    raw = EXCLUDED.raw,
    ingested_at = now()
"""

UPSERT_INSIDER_TRANSACTIONS_SQL = """
INSERT INTO insider_transactions (
    symbol_id, report_date, transaction_start_date, owner_name, transaction_type,
    shares, value, position, source, row_fingerprint
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, row_fingerprint, source) DO UPDATE SET
    report_date = EXCLUDED.report_date,
    transaction_start_date = EXCLUDED.transaction_start_date,
    owner_name = EXCLUDED.owner_name,
    transaction_type = EXCLUDED.transaction_type,
    shares = EXCLUDED.shares,
    value = EXCLUDED.value,
    position = EXCLUDED.position,
    ingested_at = now()
"""

UPSERT_EARNINGS_DATES_SQL = """
INSERT INTO earnings_dates (
    symbol_id, earnings_date, source, eps_estimate, reported_eps, surprise_pct, quarter_label
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, earnings_date, source) DO UPDATE SET
    eps_estimate = EXCLUDED.eps_estimate,
    reported_eps = EXCLUDED.reported_eps,
    surprise_pct = EXCLUDED.surprise_pct,
    quarter_label = EXCLUDED.quarter_label,
    ingested_at = now()
"""

UPSERT_SYMBOL_CLASSIFICATIONS_SQL = """
INSERT INTO symbol_classifications (
    symbol_id, as_of, sector, industry, source,
    sector_disp, sector_key, industry_disp, industry_key,
    quote_type, type_disp, exchange, full_exchange_name,
    currency, region, market, country, state, city, zip,
    website, phone, ir_website, long_name, short_name,
    full_time_employees
)
VALUES (
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s
)
ON CONFLICT (symbol_id, source, as_of) DO UPDATE SET
    sector = EXCLUDED.sector,
    industry = EXCLUDED.industry,
    sector_disp = EXCLUDED.sector_disp,
    sector_key = EXCLUDED.sector_key,
    industry_disp = EXCLUDED.industry_disp,
    industry_key = EXCLUDED.industry_key,
    quote_type = EXCLUDED.quote_type,
    type_disp = EXCLUDED.type_disp,
    exchange = EXCLUDED.exchange,
    full_exchange_name = EXCLUDED.full_exchange_name,
    currency = EXCLUDED.currency,
    region = EXCLUDED.region,
    market = EXCLUDED.market,
    country = EXCLUDED.country,
    state = EXCLUDED.state,
    city = EXCLUDED.city,
    zip = EXCLUDED.zip,
    website = EXCLUDED.website,
    phone = EXCLUDED.phone,
    ir_website = EXCLUDED.ir_website,
    long_name = EXCLUDED.long_name,
    short_name = EXCLUDED.short_name,
    full_time_employees = EXCLUDED.full_time_employees,
    ingested_at = now()
"""


def symbol_classification_upsert_tuple(
    meta: Mapping[str, Any],
    symbol_id: int,
    as_of: object,
    source: str,
) -> tuple[Any, ...]:
    """Build bind tuple for :data:`UPSERT_SYMBOL_CLASSIFICATIONS_SQL` from :func:`~shunya.data.providers.extract_yfinance_classification_fields` output."""
    return (
        symbol_id,
        as_of,
        meta.get("sector"),
        meta.get("industry"),
        source,
        meta.get("sector_disp"),
        meta.get("sector_key"),
        meta.get("industry_disp"),
        meta.get("industry_key"),
        meta.get("quote_type"),
        meta.get("type_disp"),
        meta.get("exchange"),
        meta.get("full_exchange_name"),
        meta.get("currency"),
        meta.get("region"),
        meta.get("market"),
        meta.get("country"),
        meta.get("state"),
        meta.get("city"),
        meta.get("zip"),
        meta.get("website"),
        meta.get("phone"),
        meta.get("ir_website"),
        meta.get("long_name"),
        meta.get("short_name"),
        meta.get("full_time_employees"),
    )
