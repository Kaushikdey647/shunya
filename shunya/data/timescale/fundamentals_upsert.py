"""Upsert relational fundamentals / events after yfinance fetches (API or local cache)."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from .fundamentals_relational_lib import (
    calendar_dict_to_earnings_rows,
    insider_table_to_rows,
    valuation_measures_to_daily_rows,
)
from .ingest_lib import (
    UPSERT_EARNINGS_DATES_SQL,
    UPSERT_FUND_DAILY_SQL,
    UPSERT_INSIDER_TRANSACTIONS_SQL,
    ensure_symbols,
)
from .market_cache_lib import try_market_cache_dsn

_log = logging.getLogger(__name__)


def sync_valuation_measures_to_db(
    *,
    ticker: str,
    source: str,
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    dsn = try_market_cache_dsn()
    if not dsn or not records:
        return
    try:
        import psycopg
    except ModuleNotFoundError:
        return
    rows = None
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                tmap = ensure_symbols(cur, [str(ticker)])
                sid = tmap[str(ticker)]
                rows = valuation_measures_to_daily_rows(
                    sid, columns=list(columns), records=list(records), source=str(source)
                )
                if rows:
                    cur.executemany(UPSERT_FUND_DAILY_SQL, rows)
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("fundamentals_daily upsert failed %s: %s", ticker, exc)


def sync_insider_transactions_table_to_db(
    *,
    ticker: str,
    source: str,
    columns: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> None:
    dsn = try_market_cache_dsn()
    if not dsn or not records:
        return
    try:
        import psycopg
    except ModuleNotFoundError:
        return
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                tmap = ensure_symbols(cur, [str(ticker)])
                sid = tmap[str(ticker)]
                rows = insider_table_to_rows(
                    sid, columns=list(columns), records=list(records), source=str(source)
                )
                if rows:
                    cur.executemany(UPSERT_INSIDER_TRANSACTIONS_SQL, rows)
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("insider_transactions upsert failed %s: %s", ticker, exc)


def sync_calendar_dict_to_earnings_db(
    *,
    ticker: str,
    source: str,
    data: Mapping[str, Any],
) -> None:
    dsn = try_market_cache_dsn()
    if not dsn or not data:
        return
    try:
        import psycopg
    except ModuleNotFoundError:
        return
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                tmap = ensure_symbols(cur, [str(ticker)])
                sid = tmap[str(ticker)]
                rows = calendar_dict_to_earnings_rows(sid, data, source=str(source))
                if rows:
                    cur.executemany(UPSERT_EARNINGS_DATES_SQL, rows)
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("earnings_dates upsert failed %s: %s", ticker, exc)
