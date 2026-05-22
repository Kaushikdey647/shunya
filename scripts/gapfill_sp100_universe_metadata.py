#!/usr/bin/env python3
"""
Backfill **symbol_classifications** and **fundamentals_daily** for SP100 members so the
saved universe **SP100** overview (sector/industry mix, aggregate fundamentals) has data.

The HTTP summary reads ``symbol_classifications`` (latest ``as_of``) and ``fundamentals_daily``
(latest ``as_of_ts`` per member); see ``api.repositories.universes.universe_summary``.

This script is idempotent (upserts). By default it runs Yahoo-backed ingests only when the
DB reports missing rows; use ``--force`` to always refresh.

Requires: ``uv sync --extra timescale`` (and ``examples/`` on ``PYTHONPATH`` for
``ingest-fundamentals``, same as ``shunya-timescale ingest-fundamentals``).

Example::

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
    uv run python scripts/gapfill_sp100_universe_metadata.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

INDEX_CODE = "SP100"
_TICKERS_EXCLUDED = frozenset({"GS-PK"})
_DEFAULT_OHLCV_START = date(2020, 1, 1)
_FUND_LOOKBACK_YEARS = 2


def _constituent_tickers_sp100(dsn: str) -> list[str]:
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker
                FROM symbol_index_membership m
                JOIN symbols s ON s.id = m.symbol_id
                WHERE m.index_code = %s
                ORDER BY s.ticker
                """,
                (INDEX_CODE,),
            )
            return [str(r[0]) for r in cur.fetchall()]


def _filter_constituents(tickers: list[str]) -> list[str]:
    return [t for t in tickers if t not in _TICKERS_EXCLUDED]


def _counts_missing(dsn: str) -> tuple[int, int]:
    """Return (missing_classifications, missing_fundamentals_daily_with_market_cap)."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH cons AS (
                    SELECT s.id AS symbol_id
                    FROM symbol_index_membership m
                    JOIN symbols s ON s.id = m.symbol_id
                    WHERE m.index_code = %s
                      AND NOT (s.ticker = ANY (%s))
                )
                SELECT
                    (SELECT COUNT(*)::int FROM cons c
                     WHERE NOT EXISTS (
                         SELECT 1 FROM symbol_classifications sc
                         WHERE sc.symbol_id = c.symbol_id
                     )),
                    (SELECT COUNT(*)::int FROM cons c
                     WHERE NOT EXISTS (
                         SELECT 1 FROM fundamentals_daily f
                         WHERE f.symbol_id = c.symbol_id
                           AND f.market_cap IS NOT NULL
                           AND f.market_cap > 0
                           AND lower(f.market_cap::text) <> 'nan'
                     ))
                """,
                (INDEX_CODE, list(_TICKERS_EXCLUDED)),
            )
            row = cur.fetchone()
            if not row:
                return 0, 0
            return int(row[0] or 0), int(row[1] or 0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: DATABASE_URL / SHUNYA_DATABASE_URL).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run ingests even when counts show no gaps (refresh from Yahoo).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gap counts and planned window only; no Yahoo or DB writes.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    durl = (args.database_url or os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL") or "").strip()
    if not durl:
        print("Set DATABASE_URL or SHUNYA_DATABASE_URL, or pass --database-url", file=sys.stderr)
        return 2
    os.environ["DATABASE_URL"] = durl

    from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
    from shunya.data.timescale.cli import cmd_ingest_classifications, cmd_ingest_fundamentals
    from shunya.data.timescale.dbutil import get_database_url

    dsn = get_database_url()
    raw = _constituent_tickers_sp100(dsn)
    constituents = _filter_constituents(raw)
    if not constituents:
        print("gapfill_sp100_universe_metadata: no SP100 constituents in DB; nothing to do.")
        return 0

    miss_c, miss_f = _counts_missing(dsn)
    end_d = date.today()
    fund_start_d = _DEFAULT_OHLCV_START - timedelta(days=365 * _FUND_LOOKBACK_YEARS + 5)
    fund_start_s = fund_start_d.isoformat()
    end_s = end_d.isoformat()

    print(
        f"gapfill_sp100_universe_metadata: {len(constituents)} constituents; "
        f"missing_classifications={miss_c}, missing_fundamentals_daily(market_cap)={miss_f}"
    )

    if args.dry_run:
        print(f"dry-run: would use fundamentals window [{fund_start_s}, {end_s})")
        return 0

    if miss_c == 0 and miss_f == 0 and not args.force:
        print("gapfill_sp100_universe_metadata: no gaps (--force to refresh anyway).")
        return 0

    sym_arg = " ".join(constituents)

    # TODO(market-data-router): Hardcoded classification/fundamental source strings; tie to shared SourceId / CLI defaults.
    ns_c = argparse.Namespace(symbols=sym_arg, source=STORED_OHLCV_DEFAULT_UPSTREAM_ID, insecure_curl=False)
    print("gapfill_sp100_universe_metadata: ingest-classifications …")
    rc = cmd_ingest_classifications(ns_c)
    if rc != 0:
        print(f"ingest-classifications failed with code {rc}", file=sys.stderr)
        return rc

    ns_f = argparse.Namespace(
        symbols=sym_arg,
        start=fund_start_s,
        end=end_s,
        source="yfinance_statements",
        provider="yfinance",
        yearly=False,
        fields="",
        insecure_curl=False,
    )
    print("gapfill_sp100_universe_metadata: ingest-fundamentals …")
    rc = cmd_ingest_fundamentals(ns_f)
    if rc != 0:
        print(f"ingest-fundamentals failed with code {rc}", file=sys.stderr)
        return rc

    print("gapfill_sp100_universe_metadata: done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
