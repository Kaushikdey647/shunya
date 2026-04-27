"""Timescale queries for index constituents and OHLCV sanity checks."""

from __future__ import annotations

from typing import Any

import pandas as pd
from psycopg.rows import dict_row

from backtest_api.db import resolve_database_url
from backtest_api.index_catalog import RAW_INDEX_TICKER_BY_CODE, known_index_codes

# Matches TimescaleMarketDataProvider default daily bars.
_OHLCV_SOURCE = "yfinance"


def index_exists(index_code: str) -> bool:
    import psycopg

    code = index_code.strip().upper()
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM equity_indexes WHERE code = %s LIMIT 1",
                (code,),
            )
            return cur.fetchone() is not None


def constituent_tickers(index_code: str) -> list[str]:
    code = index_code.strip().upper()
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker
                FROM symbol_index_membership m
                JOIN symbols s ON s.id = m.symbol_id
                WHERE m.index_code = %s
                ORDER BY s.ticker
                """,
                (code,),
            )
            rows = cur.fetchall()
    return [str(r[0]) for r in rows]


def list_indices_for_api() -> list[dict[str, Any]]:
    """
    Rows for GET /indices: codes present in DB and in the raw-index ticker catalog,
    with member counts.
    """
    import psycopg

    codes = list(known_index_codes())
    if not codes:
        return []
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT e.code, e.display_name,
                       (SELECT COUNT(*)::int FROM symbol_index_membership m
                        WHERE m.index_code = e.code) AS member_count
                FROM equity_indexes e
                WHERE e.code = ANY(%s)
                ORDER BY e.display_name
                """,
                (codes,),
            )
            rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["benchmark_ticker"] = RAW_INDEX_TICKER_BY_CODE.get(str(d["code"]), "")
        out.append(d)
    return out


def ohlcv_bar_counts_in_window(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    interval: str,
    source: str = _OHLCV_SOURCE,
) -> dict[str, int]:
    """Return ``ticker -> bar count`` for ``[start_date, end_date)`` (``0`` if symbol or bars missing)."""
    if not tickers:
        return {}
    import psycopg

    t0 = pd.Timestamp(start_date).normalize()
    t1 = pd.Timestamp(end_date).normalize()
    if t1 <= t0:
        raise ValueError("end_date must be after start_date")

    out: dict[str, int] = {str(t): 0 for t in tickers}
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker,
                       COUNT(b.ts) FILTER (
                         WHERE b.open IS NOT NULL
                           AND b.high IS NOT NULL
                           AND b.low IS NOT NULL
                           AND b.close IS NOT NULL
                           AND b.volume IS NOT NULL
                           AND b.volume >= 0
                       )::bigint AS n
                FROM symbols s
                LEFT JOIN ohlcv_bars b ON b.symbol_id = s.id
                  AND b.interval = %s
                  AND b.source = %s
                  AND b.ts >= %s
                  AND b.ts < %s
                WHERE s.ticker = ANY(%s)
                GROUP BY s.ticker
                """,
                (interval, source, t0, t1, list(tickers)),
            )
            for r in cur.fetchall():
                out[str(r[0])] = int(r[1])
    return out


def tickers_with_ohlcv_in_window(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    interval: str,
    source: str = _OHLCV_SOURCE,
) -> list[str]:
    """Constituents that have at least one OHLCV bar in the window (stable sort)."""
    counts = ohlcv_bar_counts_in_window(
        tickers, start_date=start_date, end_date=end_date, interval=interval, source=source
    )
    return sorted(t for t in tickers if counts.get(str(t), 0) >= 1)


def validate_ohlcv_window(
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    interval: str,
    source: str = _OHLCV_SOURCE,
) -> None:
    """
    Ensure each ticker has at least one bar in [start, end).

    Raises ValueError with a short message if any ticker is missing data.
    """
    if not tickers:
        raise ValueError("empty ticker list for OHLCV validation")

    rows = ohlcv_bar_counts_in_window(
        tickers, start_date=start_date, end_date=end_date, interval=interval, source=source
    )
    missing = [t for t in tickers if rows.get(t, 0) < 1]
    if missing:
        sample = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ValueError(
            f"No OHLCV bars in range for {len(missing)} ticker(s) "
            f"(interval={interval!r}, source={source!r}): {sample}{more}"
        )
