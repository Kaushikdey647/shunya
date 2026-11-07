"""Load SEC sector / industry (view ``symbol_reference``) from Timescale for finTs."""

from __future__ import annotations

from typing import Mapping

from api.db import resolve_database_url


def load_classifications_for_tickers(ticker_list: list[str]) -> Mapping[str, Mapping[str, str]]:
    """
    Sector is the SEC owner org, industry the SIC description (``sync-reference``).

    Maps DB lowercase columns to finTs keys ``Sector`` and ``Industry``.
    """
    if not ticker_list:
        return {}
    import psycopg

    out: dict[str, dict[str, str]] = {}
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, COALESCE(sector, '') AS sector, COALESCE(industry, '') AS industry
                FROM symbol_reference
                WHERE ticker = ANY(%s) AND sic IS NOT NULL
                """,
                (list(str(t) for t in ticker_list),),
            )
            for row in cur.fetchall():
                t, sec, ind = str(row[0]), str(row[1]), str(row[2])
                out[t] = {
                    "Sector": sec or "UnknownSector",
                    "Industry": ind or "UnknownIndustry",
                }
    return out
