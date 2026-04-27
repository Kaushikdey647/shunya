"""CLI: apply migrations and bootstrap TimescaleDB from live providers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import List, Sequence

from ..providers import YFinanceMarketDataProvider, fetch_yfinance_classifications
from ..timeframes import BarSpec, default_bar_index_policy, default_bar_spec
from .dbutil import apply_migrations, get_database_url
from .index_membership_sync import load_py_ticker_index_union, sync_symbol_index_memberships
from .ingest_lib import (
    UPSERT_FUND_SQL,
    UPSERT_OHLCV_SQL,
    UPSERT_SYMBOL_CLASSIFICATIONS_SQL,
    ensure_symbols,
    fundamentals_eav_rows,
    rows_from_provider_ohlcv,
    symbol_classification_upsert_tuple,
)
from .intervals import bar_spec_to_interval_key


def _parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()]


def cmd_migrate(_: argparse.Namespace) -> int:
    apply_migrations()
    return 0


def cmd_sync_index_memberships(_: argparse.Namespace) -> int:
    """Fill ``symbol_index_membership`` from PyTickerSymbols (no OHLCV download)."""
    import psycopg

    symbols, membership_sets, display_names = load_py_ticker_index_union()
    if not symbols:
        print("PyTickerSymbols returned no tickers", file=sys.stderr)
        return 1
    dsn = get_database_url()
    sync_symbol_index_memberships(psycopg, dsn, symbols, membership_sets, display_names)
    print(f"sync-index-memberships: upserted memberships for {len(symbols)} tickers")
    return 0


def cmd_ingest_ohlcv(args: argparse.Namespace) -> int:
    import psycopg

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        print("no symbols", file=sys.stderr)
        return 2

    session = None
    if args.insecure_curl:
        try:
            from curl_cffi import requests as curl_requests

            session = curl_requests.Session(impersonate="chrome", verify=False)
        except ImportError:
            print("curl_cffi not installed; omit --insecure-curl or pip install curl-cffi", file=sys.stderr)
            return 1

    spec = default_bar_spec()
    policy = default_bar_index_policy()
    if args.bar_unit is not None:
        from ..timeframes import BarUnit

        spec = BarSpec(BarUnit(args.bar_unit), int(args.bar_step))

    interval = bar_spec_to_interval_key(spec)
    source = str(args.source)

    prov = YFinanceMarketDataProvider(session=session)
    raw = prov.download(symbols, args.start, args.end, bar_spec=spec, bar_index_policy=policy)
    if raw.empty:
        print("provider returned empty frame", file=sys.stderr)
        return 1

    dsn = get_database_url()
    n = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES ('ingest_ohlcv', %s, %s, 'running')
                RETURNING id
                """,
                (source, json.dumps({"symbols": symbols, "start": args.start, "end": args.end, "interval": interval})),
            )
            run_id = int(cur.fetchone()[0])
            tmap = ensure_symbols(cur, symbols)
            rows = rows_from_provider_ohlcv(raw, tmap, interval=interval, source=source)
            for chunk_start in range(0, len(rows), 2000):
                chunk = rows[chunk_start : chunk_start + 2000]
                cur.executemany(UPSERT_OHLCV_SQL, chunk)
                n += len(chunk)
            cur.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = now(), rows_upserted = %s, status = 'ok'
                WHERE id = %s
                """,
                (n, run_id),
            )
        conn.commit()
    print(f"ingest_ohlcv: upserted {n} bars for {len(symbols)} symbols ({interval}, {source})")
    return 0


def cmd_ingest_fundamentals(args: argparse.Namespace) -> int:
    import psycopg

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        return 2

    session = None
    if args.insecure_curl:
        try:
            from curl_cffi import requests as curl_requests

            session = curl_requests.Session(impersonate="chrome", verify=False)
        except ImportError:
            print("curl_cffi required for --insecure-curl", file=sys.stderr)
            return 1

    if args.provider != "yfinance":
        print("only --provider yfinance is implemented (examples provider)", file=sys.stderr)
        return 2

    try:
        from examples.yfinance_fundamental_provider import YFinanceFundamentalDataProvider
    except ImportError:
        print(
            "Could not import examples.yfinance_fundamental_provider; run from repo with PYTHONPATH=.",
            file=sys.stderr,
        )
        return 1

    from shunya.data.fundamentals import FUNDAMENTAL_FIELDS

    fund = YFinanceFundamentalDataProvider(session=session, enable_fetch_cache=False)
    if args.fields and str(args.fields).strip():
        fields = [x.strip() for x in str(args.fields).replace(",", " ").split() if x.strip()]
    else:
        fields = list(FUNDAMENTAL_FIELDS)
    freq_q = not args.yearly
    periodic = fund.fetch(
        symbols,
        args.start,
        args.end,
        fields=fields,
        quarterly=freq_q,
        bar_spec=default_bar_spec(),
    )
    if periodic.empty:
        print("fundamentals fetch returned empty", file=sys.stderr)
        return 1

    freq = "quarterly" if freq_q else "yearly"
    source = str(args.source)
    dsn = get_database_url()
    n = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES ('ingest_fundamentals', %s, %s, 'running')
                RETURNING id
                """,
                (source, json.dumps({"symbols": symbols, "start": args.start, "end": args.end})),
            )
            run_id = int(cur.fetchone()[0])
            tmap = ensure_symbols(cur, symbols)
            rows = fundamentals_eav_rows(periodic, tmap, freq=freq, source=source)
            for chunk_start in range(0, len(rows), 5000):
                chunk = rows[chunk_start : chunk_start + 5000]
                cur.executemany(UPSERT_FUND_SQL, chunk)
                n += len(chunk)
            cur.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = now(), rows_upserted = %s, status = 'ok'
                WHERE id = %s
                """,
                (n, run_id),
            )
        conn.commit()
    print(f"ingest_fundamentals: upserted {n} EAV cells ({freq}, {source})")
    return 0


def cmd_ingest_classifications(args: argparse.Namespace) -> int:
    import psycopg

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        return 2

    session = None
    if args.insecure_curl:
        try:
            from curl_cffi import requests as curl_requests

            session = curl_requests.Session(impersonate="chrome", verify=False)
        except ImportError:
            print("curl_cffi required for --insecure-curl", file=sys.stderr)
            return 1

    cmap = fetch_yfinance_classifications(symbols, session=session)
    as_of = date.today()
    source = str(args.source)
    dsn = get_database_url()
    n = 0
    sql = UPSERT_SYMBOL_CLASSIFICATIONS_SQL
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES ('ingest_classifications', %s, %s, 'running')
                RETURNING id
                """,
                (source, json.dumps({"symbols": symbols})),
            )
            run_id = int(cur.fetchone()[0])
            tmap = ensure_symbols(cur, symbols)
            for t in symbols:
                sid = tmap.get(str(t))
                if sid is None:
                    continue
                meta = cmap.get(str(t), {})
                cur.execute(sql, symbol_classification_upsert_tuple(meta, sid, as_of, source))
                n += 1
            cur.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = now(), rows_upserted = %s, status = 'ok'
                WHERE id = %s
                """,
                (n, run_id),
            )
        conn.commit()
    print(f"ingest_classifications: upserted {n} rows ({source}, as_of={as_of})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m shunya.data.timescale.cli")
    p.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: DATABASE_URL or SHUNYA_DATABASE_URL)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("migrate", help="Apply packaged SQL migrations (shunya/data/timescale/migrations/*.sql)")

    sub.add_parser(
        "sync-index-memberships",
        help="Populate symbol_index_membership (+ symbols names) from PyTickerSymbols (SP100, SP500, …)",
    )

    p_ohlcv = sub.add_parser("ingest-ohlcv", help="Download OHLCV via yfinance and upsert ohlcv_bars")
    p_ohlcv.add_argument("--symbols", required=True, help="Space or comma separated tickers")
    p_ohlcv.add_argument("--start", required=True)
    p_ohlcv.add_argument("--end", required=True)
    p_ohlcv.add_argument("--source", default="yfinance")
    p_ohlcv.add_argument("--insecure-curl", action="store_true", help="Use curl_cffi session verify=False")
    p_ohlcv.add_argument("--bar-unit", default=None, help="Override BarUnit name e.g. DAYS")
    p_ohlcv.add_argument("--bar-step", type=int, default=1)

    p_f = sub.add_parser("ingest-fundamentals", help="Fetch fundamentals and upsert fundamentals_field_values")
    p_f.add_argument("--symbols", required=True)
    p_f.add_argument("--start", required=True)
    p_f.add_argument("--end", required=True)
    p_f.add_argument("--source", default="yfinance_statements")
    p_f.add_argument("--provider", default="yfinance", choices=["yfinance"])
    p_f.add_argument("--yearly", action="store_true", help="Use yearly statements instead of quarterly")
    p_f.add_argument("--fields", default="", help="Optional space-separated subset; default all FUNDAMENTAL_FIELDS")
    p_f.add_argument("--insecure-curl", action="store_true")

    p_c = sub.add_parser("ingest-classifications", help="Fetch yfinance sector/industry into symbol_classifications")
    p_c.add_argument("--symbols", required=True)
    p_c.add_argument("--source", default="yfinance")
    p_c.add_argument("--insecure-curl", action="store_true")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    durl = args.database_url or os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if not durl:
        print("Set DATABASE_URL or pass --database-url", file=sys.stderr)
        return 2
    os.environ["DATABASE_URL"] = durl

    if args.cmd == "migrate":
        return cmd_migrate(args)
    if args.cmd == "sync-index-memberships":
        return cmd_sync_index_memberships(args)
    if args.cmd == "ingest-ohlcv":
        return cmd_ingest_ohlcv(args)
    if args.cmd == "ingest-fundamentals":
        return cmd_ingest_fundamentals(args)
    if args.cmd == "ingest-classifications":
        return cmd_ingest_classifications(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
