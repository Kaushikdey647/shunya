"""Timescale-backed market data cache: OHLCV refresh manifest + yfinance instrument JSON documents."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

__all__ = [
    "DOC_FINANCIALS_BALANCE",
    "DOC_FINANCIALS_CASHFLOW",
    "DOC_FINANCIALS_INCOME",
    "DOC_HOLDERS",
    "DOC_OPTION_CHAIN",
    "DOC_OPTION_EXPIRATIONS",
    "DOC_OPTION_IV_SURFACE",
    "DOC_OVERVIEW",
    "DOC_VALUATION_MEASURES",
    "DOC_ANALYST_PRICE_TARGETS",
    "DOC_EARNINGS_ESTIMATE",
    "DOC_REVENUE_ESTIMATE",
    "DOC_EARNINGS_HISTORY",
    "DOC_EPS_TREND",
    "DOC_EPS_REVISIONS",
    "DOC_GROWTH_ESTIMATES",
    "DOC_RECOMMENDATIONS",
    "DOC_RECOMMENDATIONS_SUMMARY",
    "DOC_UPGRADES_DOWNGRADES",
    "DOC_SUSTAINABILITY",
    "DOC_INSIDER_PURCHASES",
    "DOC_INSIDER_TRANSACTIONS",
    "DOC_INSIDER_ROSTER_HOLDERS",
    "DOC_MAJOR_HOLDERS",
    "DOC_CALENDAR",
    "DOC_SEC_FILINGS",
    "FINSTRUMENT_DOC_SOURCE_YFINANCE",
    "default_market_cache_ttl_days",
    "fetch_ohlcv_manifest_last_refresh_sync",
    "get_instrument_document_payload_if_fresh",
    "ohlcv_manifest_is_fresh",
    "ohlcv_manifests_all_fresh_for_universe_on_cursor",
    "touch_ohlcv_refresh_manifest_on_cursor",
    "touch_ohlcv_refresh_manifest_sync",
    "try_market_cache_dsn",
    "upsert_instrument_document_sync",
]

FINSTRUMENT_DOC_SOURCE_YFINANCE = "yfinance"

DOC_OVERVIEW = "overview"
DOC_VALUATION_MEASURES = "valuation_measures"
DOC_ANALYST_PRICE_TARGETS = "analyst_price_targets"
DOC_EARNINGS_ESTIMATE = "earnings_estimate"
DOC_REVENUE_ESTIMATE = "revenue_estimate"
DOC_EARNINGS_HISTORY = "earnings_history"
DOC_EPS_TREND = "eps_trend"
DOC_EPS_REVISIONS = "eps_revisions"
DOC_GROWTH_ESTIMATES = "growth_estimates"
DOC_RECOMMENDATIONS = "recommendations"
DOC_RECOMMENDATIONS_SUMMARY = "recommendations_summary"
DOC_UPGRADES_DOWNGRADES = "upgrades_downgrades"
DOC_SUSTAINABILITY = "sustainability"
DOC_INSIDER_PURCHASES = "insider_purchases"
DOC_INSIDER_TRANSACTIONS = "insider_transactions"
DOC_INSIDER_ROSTER_HOLDERS = "insider_roster_holders"
DOC_MAJOR_HOLDERS = "major_holders"
DOC_CALENDAR = "calendar"
DOC_SEC_FILINGS = "sec_filings"
DOC_FINANCIALS_INCOME = "financials_income"
DOC_FINANCIALS_BALANCE = "financials_balance"
DOC_FINANCIALS_CASHFLOW = "financials_cashflow"
DOC_HOLDERS = "holders"
DOC_OPTION_EXPIRATIONS = "option_expirations"
DOC_OPTION_CHAIN = "option_chain"
DOC_OPTION_IV_SURFACE = "option_iv_surface"


def default_market_cache_ttl_days() -> int:
    """TTL in days when ``api.settings`` is not loaded (e.g. CLI / pure shunya)."""
    for key in ("SHUNYA_API_MARKET_DATA_CACHE_TTL_DAYS", "SHUNYA_MARKET_DATA_CACHE_TTL_DAYS"):
        v = os.environ.get(key)
        if v is not None and str(v).strip():
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return 30


def try_market_cache_dsn() -> str | None:
    if not (os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")):
        return None
    try:
        from shunya.data.timescale.dbutil import get_database_url

        return get_database_url()
    except ValueError:
        return None


def ohlcv_manifests_all_fresh_for_universe_on_cursor(
    cur: Any,
    *,
    tickers: list[str],
    interval: str,
    source: str,
    ttl_days: int,
    now: datetime | None = None,
) -> bool:
    """Every ticker must exist in ``symbols`` and have a manifest row within ``ttl_days``."""
    now_ = now or datetime.now(timezone.utc)
    cur.execute(
        """
        WITH want AS (SELECT unnest(%s::text[]) AS ticker)
        SELECT w.ticker, r.last_refresh_at
        FROM want w
        JOIN symbols s ON s.ticker = w.ticker
        LEFT JOIN ohlcv_symbol_interval_refresh r
          ON r.symbol_id = s.id AND r.interval = %s AND r.source = %s
        """,
        (tickers, interval, source),
    )
    rows = cur.fetchall()
    if len(rows) != len(tickers):
        return False
    for _t, lr in rows:
        if not ohlcv_manifest_is_fresh(lr, ttl_days=ttl_days, now=now_):
            return False
    return True


def ohlcv_manifest_is_fresh(
    last_refresh_at: datetime | None,
    *,
    ttl_days: int,
    now: datetime | None = None,
) -> bool:
    if last_refresh_at is None:
        return False
    now_ = now or datetime.now(timezone.utc)
    lr = last_refresh_at
    if lr.tzinfo is None:
        lr = lr.replace(tzinfo=timezone.utc)
    else:
        lr = lr.astimezone(timezone.utc)
    return lr >= now_ - timedelta(days=max(1, int(ttl_days)))


def fetch_ohlcv_manifest_last_refresh_sync(
    dsn: str,
    *,
    ticker: str,
    interval: str,
    source: str,
) -> datetime | None:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.last_refresh_at
                FROM ohlcv_symbol_interval_refresh r
                JOIN symbols s ON s.id = r.symbol_id
                WHERE s.ticker = %s AND r.interval = %s AND r.source = %s
                """,
                (ticker, interval, source),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    ts: datetime = row[0]
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def touch_ohlcv_refresh_manifest_on_cursor(
    cur: Any,
    *,
    symbol_id: int,
    interval: str,
    source: str,
) -> None:
    cur.execute(
        """
        INSERT INTO ohlcv_symbol_interval_refresh (symbol_id, interval, source, last_refresh_at, last_error, updated_at)
        VALUES (%s, %s, %s, now(), NULL, now())
        ON CONFLICT (symbol_id, interval, source) DO UPDATE SET
            last_refresh_at = EXCLUDED.last_refresh_at,
            last_error = NULL,
            updated_at = now()
        """,
        (symbol_id, interval, source),
    )


def touch_ohlcv_refresh_manifest_sync(
    dsn: str,
    *,
    symbol: str,
    interval: str,
    source: str,
) -> None:
    import psycopg

    from shunya.data.timescale.ingest_lib import ensure_symbols

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            tmap = ensure_symbols(cur, [symbol])
            touch_ohlcv_refresh_manifest_on_cursor(
                cur, symbol_id=tmap[symbol], interval=interval, source=source
            )
        conn.commit()


def get_instrument_document_payload_if_fresh(
    dsn: str,
    *,
    ticker: str,
    source: str,
    resource_type: str,
    resource_key: str,
    ttl_days: int,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    import psycopg

    now_ = now or datetime.now(timezone.utc)
    cutoff = now_ - timedelta(days=max(1, int(ttl_days)))
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.payload
                FROM instrument_yfinance_documents d
                JOIN symbols s ON s.id = d.symbol_id
                WHERE s.ticker = %s AND d.source = %s AND d.resource_type = %s
                  AND d.resource_key = %s AND d.fetched_at >= %s
                """,
                (ticker, source, resource_type, resource_key, cutoff),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    pl = row[0]
    return pl if isinstance(pl, dict) else None


def upsert_instrument_document_sync(
    dsn: str,
    *,
    ticker: str,
    source: str,
    resource_type: str,
    resource_key: str,
    payload: Mapping[str, Any],
) -> None:
    import psycopg
    from psycopg.types.json import Json

    from shunya.data.timescale.ingest_lib import ensure_symbols

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            tmap = ensure_symbols(cur, [ticker])
            sid = tmap[ticker]
            cur.execute(
                """
                INSERT INTO instrument_yfinance_documents (
                    symbol_id, source, resource_type, resource_key, payload, fetched_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (symbol_id, source, resource_type, resource_key) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    fetched_at = EXCLUDED.fetched_at,
                    updated_at = now()
                """,
                (sid, source, resource_type, resource_key, Json(dict(payload))),
            )
        conn.commit()
