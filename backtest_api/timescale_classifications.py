"""Load ``symbol_classifications`` from Timescale for finTs without Yahoo."""

from __future__ import annotations

from typing import Mapping

from backtest_api.db import resolve_database_url


def load_classifications_for_tickers(ticker_list: list[str]) -> Mapping[str, Mapping[str, str]]:
    """
    Latest ``as_of`` row per symbol (``source='yfinance'`` matches ingest CLI).

    Maps DB lowercase columns to finTs keys ``Sector``, ``Industry``, ``SubIndustry``.
    """
    if not ticker_list:
        return {}
    import psycopg

    out: dict[str, dict[str, str]] = {}
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (s.ticker)
                    s.ticker,
                    COALESCE(c.sector, '') AS sector,
                    COALESCE(c.industry, '') AS industry,
                    COALESCE(c.sub_industry, '') AS sub_industry
                FROM symbols s
                INNER JOIN symbol_classifications c ON c.symbol_id = s.id AND c.source = 'yfinance'
                WHERE s.ticker = ANY(%s)
                ORDER BY s.ticker, c.as_of DESC
                """,
                (list(str(t) for t in ticker_list),),
            )
            for row in cur.fetchall():
                t, sec, ind, sub = str(row[0]), str(row[1]), str(row[2]), str(row[3])
                out[t] = {
                    "Sector": sec or "UnknownSector",
                    "Industry": ind or "UnknownIndustry",
                    "SubIndustry": sub or "UnknownSubIndustry",
                }
    return out
