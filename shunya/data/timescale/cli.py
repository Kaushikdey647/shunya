"""CLI: apply migrations and bootstrap TimescaleDB from live providers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from typing import List, Sequence

import pandas as pd

from ..providers import (
    AlphaVantageMarketDataProvider,
    TiingoMarketDataProvider,
    YFinanceMarketDataProvider,
    env_yfinance_repair_default,
    fetch_yfinance_classifications,
)
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
from .market_cache_lib import touch_ohlcv_refresh_manifest_on_cursor


def _parse_symbols(s: str) -> List[str]:
    return [x.strip() for x in s.replace(",", " ").split() if x.strip()]


def _tickers_from_symbols_table(*, limit: int | None, offset: int) -> List[str]:
    import psycopg

    dsn = get_database_url()
    parts = ["SELECT ticker FROM symbols ORDER BY ticker"]
    params: list[int] = []
    if limit is not None:
        parts.append("LIMIT %s")
        params.append(int(limit))
    if offset:
        parts.append("OFFSET %s")
        params.append(int(offset))
    sql = " ".join(parts)
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [str(r[0]) for r in cur.fetchall()]


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

    if args.symbols_from_db:
        symbols = _tickers_from_symbols_table(limit=args.db_limit, offset=int(args.db_offset or 0))
    else:
        symbols = _parse_symbols(args.symbols or "")
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
    if args.provider == "alphavantage" and source == "yfinance":
        source = "alphavantage"
    if args.provider == "tiingo" and source == "yfinance":
        source = "tiingo"

    if args.provider == "yfinance":
        prov = YFinanceMarketDataProvider(session=session, repair=env_yfinance_repair_default())
        raw = prov.download(
            symbols,
            args.start,
            args.end,
            bar_spec=spec,
            bar_index_policy=policy,
            repair=env_yfinance_repair_default(),
        )
    elif args.provider == "tiingo":
        prov = TiingoMarketDataProvider(
            inter_request_delay_seconds=float(args.tiingo_delay_seconds),
        )
        raw = prov.download(
            symbols,
            args.start,
            args.end,
            bar_spec=spec,
            bar_index_policy=policy,
        )
    else:
        prov = AlphaVantageMarketDataProvider(
            session=session,
            outputsize=args.av_outputsize,
            inter_request_delay_seconds=float(args.av_delay_seconds),
        )
        raw = prov.download(
            symbols,
            args.start,
            args.end,
            bar_spec=spec,
            bar_index_policy=policy,
        )
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
                (
                    source,
                    json.dumps(
                        {
                            "symbols": symbols,
                            "start": args.start,
                            "end": args.end,
                            "interval": interval,
                            "provider": args.provider,
                            "av_outputsize": args.av_outputsize,
                            "av_delay_seconds": args.av_delay_seconds,
                            "tiingo_delay_seconds": args.tiingo_delay_seconds,
                            "symbols_from_db": bool(args.symbols_from_db),
                            "db_limit": args.db_limit,
                            "db_offset": args.db_offset,
                        }
                    ),
                ),
            )
            run_id = int(cur.fetchone()[0])
            tmap = ensure_symbols(cur, symbols)
            rows = rows_from_provider_ohlcv(raw, tmap, interval=interval, source=source)
            for chunk_start in range(0, len(rows), 2000):
                chunk = rows[chunk_start : chunk_start + 2000]
                cur.executemany(UPSERT_OHLCV_SQL, chunk)
                n += len(chunk)
            for _sym_t, sym_id in tmap.items():
                touch_ohlcv_refresh_manifest_on_cursor(
                    cur, symbol_id=sym_id, interval=interval, source=source
                )
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

    from shunya.data.timescale.fundamentals_relational_lib import (
        dataframe_to_columns_records,
        earnings_dates_dataframe_to_rows,
        insider_table_to_rows,
        periodic_frame_to_wide_rows,
        ticker_info_to_daily_row,
        valuation_measures_to_daily_rows,
        yfinance_dividends_splits_to_corporate_actions,
    )
    from shunya.data.timescale.ingest_lib import (
        UPSERT_CORPORATE_ACTIONS_SQL,
        UPSERT_EARNINGS_DATES_SQL,
        UPSERT_FUND_ANNUAL_SQL,
        UPSERT_FUND_DAILY_SQL,
        UPSERT_FUND_QUARTERLY_SQL,
        UPSERT_INSIDER_TRANSACTIONS_SQL,
    )
    from shunya.data.yfinance_session import build_yfinance_session

    import yfinance as yf

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
    fields_full = list(FUNDAMENTAL_FIELDS)
    yf_sess = build_yfinance_session()
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

            if freq_q:
                wq = periodic_frame_to_wide_rows(periodic, tmap, source=source)
                if wq:
                    cur.executemany(UPSERT_FUND_QUARTERLY_SQL, wq)
                other = fund.fetch(
                    symbols,
                    args.start,
                    args.end,
                    fields=fields_full,
                    quarterly=False,
                    bar_spec=default_bar_spec(),
                )
                if not other.empty:
                    wa = periodic_frame_to_wide_rows(other, tmap, source=source)
                    if wa:
                        cur.executemany(UPSERT_FUND_ANNUAL_SQL, wa)
            else:
                wa = periodic_frame_to_wide_rows(periodic, tmap, source=source)
                if wa:
                    cur.executemany(UPSERT_FUND_ANNUAL_SQL, wa)
                other = fund.fetch(
                    symbols,
                    args.start,
                    args.end,
                    fields=fields_full,
                    quarterly=True,
                    bar_spec=default_bar_spec(),
                )
                if not other.empty:
                    wq = periodic_frame_to_wide_rows(other, tmap, source=source)
                    if wq:
                        cur.executemany(UPSERT_FUND_QUARTERLY_SQL, wq)

            as_of_info = datetime.now(timezone.utc)
            for sym in symbols:
                sid = tmap[str(sym)]
                t = yf.Ticker(str(sym), session=yf_sess)
                try:
                    info = t.info or {}
                except Exception:
                    info = {}
                if isinstance(info, Mapping) and info:
                    try:
                        drow = ticker_info_to_daily_row(sid, info, as_of=as_of_info, source=source)
                        cur.execute(UPSERT_FUND_DAILY_SQL, drow)
                    except Exception:
                        pass
                try:
                    vm = t.get_valuation_measures()
                except Exception:
                    vm = None
                cols, recs = dataframe_to_columns_records(vm if isinstance(vm, pd.DataFrame) else None)
                if recs:
                    drows = valuation_measures_to_daily_rows(
                        sid, columns=cols, records=recs, source=source
                    )
                    if drows:
                        cur.executemany(UPSERT_FUND_DAILY_SQL, drows)
                try:
                    ca = yfinance_dividends_splits_to_corporate_actions(
                        sid, dividends=t.dividends, splits=t.splits, source=source
                    )
                    if ca:
                        cur.executemany(UPSERT_CORPORATE_ACTIONS_SQL, ca)
                except Exception:
                    pass
                try:
                    ins = t.get_insider_transactions()
                except Exception:
                    ins = None
                icols, irecs = dataframe_to_columns_records(ins if isinstance(ins, pd.DataFrame) else None)
                if irecs:
                    irows = insider_table_to_rows(sid, columns=icols, records=irecs, source=source)
                    if irows:
                        cur.executemany(UPSERT_INSIDER_TRANSACTIONS_SQL, irows)
                try:
                    edf = t.earnings_dates
                except Exception:
                    edf = None
                erows = earnings_dates_dataframe_to_rows(sid, edf, source=source)
                if erows:
                    cur.executemany(UPSERT_EARNINGS_DATES_SQL, erows)

            cur.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = now(), rows_upserted = %s, status = 'ok'
                WHERE id = %s
                """,
                (n, run_id),
            )
        conn.commit()
    print(f"ingest_fundamentals: upserted {n} EAV cells ({freq}, {source}); wide + events synced")
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

    p_ohlcv = sub.add_parser(
        "ingest-ohlcv",
        help="Download OHLCV (yfinance, Alpha Vantage, or Tiingo) and upsert ohlcv_bars",
    )
    sym_grp = p_ohlcv.add_mutually_exclusive_group(required=True)
    sym_grp.add_argument(
        "--symbols",
        default=None,
        help="Space or comma separated tickers (mutually exclusive with --symbols-from-db)",
    )
    sym_grp.add_argument(
        "--symbols-from-db",
        action="store_true",
        help="Ingest all tickers present in the symbols table (see --db-limit / --db-offset)",
    )
    p_ohlcv.add_argument("--start", required=True)
    p_ohlcv.add_argument("--end", required=True)
    p_ohlcv.add_argument(
        "--source",
        default="yfinance",
        help="Stored in ohlcv_bars.source; default becomes alphavantage when --provider alphavantage",
    )
    p_ohlcv.add_argument(
        "--provider",
        choices=["yfinance", "alphavantage", "tiingo"],
        default="yfinance",
        help="Market data API (alphavantage/tiingo: daily only; set API keys per provider)",
    )
    p_ohlcv.add_argument(
        "--av-outputsize",
        choices=["compact", "full"],
        default="compact",
        help="Alpha Vantage TIME_SERIES_DAILY outputsize; full requires premium per AV docs",
    )
    p_ohlcv.add_argument(
        "--av-delay-seconds",
        type=float,
        default=12.0,
        help="Pause between Alpha Vantage symbol requests (default 12s ~ 5/min free tier)",
    )
    p_ohlcv.add_argument(
        "--tiingo-delay-seconds",
        type=float,
        default=0.0,
        help="Pause between Tiingo symbol requests (default 0; use for large batches under quotas)",
    )
    p_ohlcv.add_argument(
        "--db-limit",
        type=int,
        default=None,
        help="With --symbols-from-db: max number of tickers to process",
    )
    p_ohlcv.add_argument(
        "--db-offset",
        type=int,
        default=0,
        help="With --symbols-from-db: skip first N tickers (ordered)",
    )
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
