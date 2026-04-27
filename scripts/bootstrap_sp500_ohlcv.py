#!/usr/bin/env python3
"""
Bootstrap Timescale ``ohlcv_bars`` with daily OHLCV for the S&P 500 from Yahoo Finance.

Uses :class:`pytickersymbols.PyTickerSymbols` (same universe pattern as
``examples/alpha_benchmark_oex.ipynb``) for index ``S&P 500``, resolves each
name to a **USD Yahoo** ticker when available, then downloads via
:class:`~shunya.data.providers.YFinanceMarketDataProvider` and upserts like
``shunya-timescale ingest-ohlcv``.

Symbols with no usable bars from yfinance are **skipped** (logged, no failure).

Requires: ``uv sync --extra timescale``, ``DATABASE_URL``, migrations applied
(``shunya-timescale migrate``).

HTTP/TLS matches the rest of Shunya: :func:`~shunya.data.yfinance_session.build_yfinance_session`
(curl_cffi Chrome, ``verify=False`` by default). Set ``YFINANCE_TLS_VERIFY=1`` for strict certificate
verification.

Example::

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
    uv run python scripts/bootstrap_sp500_ohlcv.py --start 2015-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Sequence

import pandas as pd

from shunya.data.providers import YFinanceMarketDataProvider
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy
from shunya.data.yfinance_session import build_yfinance_session
from shunya.data.timescale.dbutil import get_database_url
from shunya.data.timescale.ingest_lib import UPSERT_OHLCV_SQL, ensure_symbols, rows_from_provider_ohlcv
from shunya.data.timescale.intervals import bar_spec_to_interval_key

_LOG = logging.getLogger(__name__)


def _yahoo_usd_symbol(stock: dict[str, Any]) -> str | None:
    """Prefer Yahoo USD listing from ``symbols[]``; fall back to top-level ``symbol``."""
    raw_syms = stock.get("symbols")
    if isinstance(raw_syms, list):
        for entry in raw_syms:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("currency", "")).upper() != "USD":
                continue
            y = entry.get("yahoo")
            if isinstance(y, str) and y.strip():
                return y.strip()
    sym = stock.get("symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip()
    return None


def _parse_sp500_yahoo_symbols() -> list[str]:
    from pytickersymbols import PyTickerSymbols

    pts = PyTickerSymbols()
    stocks = list(pts.get_stocks_by_index("S&P 500"))
    out: list[str] = []
    seen: set[str] = set()
    for stock in stocks:
        y = _yahoo_usd_symbol(stock if isinstance(stock, dict) else {})
        if not y:
            continue
        u = y.upper()
        if u in seen:
            continue
        seen.add(u)
        out.append(y)
    out.sort(key=str.upper)
    return out


def _ticker_frame_has_close_bars(part: pd.DataFrame) -> bool:
    if part is None or part.empty:
        return False
    if "Close" not in part.columns:
        return False
    return bool(part["Close"].notna().any())


def _slice_ohlcv_for_tickers(raw: pd.DataFrame, requested: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep only requested tickers that have at least one non-null Close.
    Returns (frame slice, tickers skipped because empty).
    """
    skipped_empty: list[str] = []
    if raw.empty:
        return raw, list(requested)

    if not isinstance(raw.columns, pd.MultiIndex):
        if len(requested) == 1 and _ticker_frame_has_close_bars(raw):
            return raw.copy(), []
        return pd.DataFrame(), list(requested)

    keep: list[str] = []
    for t in requested:
        try:
            part = raw[str(t)]
        except (KeyError, TypeError):
            skipped_empty.append(str(t))
            continue
        if _ticker_frame_has_close_bars(part):
            keep.append(str(t))
        else:
            skipped_empty.append(str(t))

    if not keep:
        return pd.DataFrame(), list(requested)
    return raw[keep].copy(), skipped_empty


def _chunked(seq: Sequence[str], size: int) -> list[list[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _upsert_batch(
    psycopg_module: Any,
    dsn: str,
    df: pd.DataFrame,
    interval_key: str,
    source: str,
    *,
    flat_symbol: str | None,
) -> int:
    """Upsert one batch; for a flat single-ticker frame pass ``flat_symbol``."""
    if df.empty:
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        symbols = [str(t) for t in df.columns.get_level_values(0).unique()]
    else:
        if not flat_symbol:
            raise ValueError("flat OHLCV requires flat_symbol")
        symbols = [flat_symbol]

    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES ('bootstrap_sp500_ohlcv', %s, %s, 'running')
                RETURNING id
                """,
                (source, json.dumps({"tickers": symbols, "interval": interval_key})),
            )
            run_id = int(cur.fetchone()[0])
            tmap = ensure_symbols(cur, symbols)
            rows = rows_from_provider_ohlcv(df, tmap, interval=interval_key, source=source)
            n = 0
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
    return n


def _download_one(
    prov: YFinanceMarketDataProvider,
    symbol: str,
    start: str,
    end: str,
    spec: BarSpec,
    policy: Any,
) -> pd.DataFrame:
    return prov.download([symbol], start, end, bar_spec=spec, bar_index_policy=policy)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="Exclusive end date (YYYY-MM-DD, yfinance-style)")
    p.add_argument("--database-url", default=None, help="Postgres URL (default: DATABASE_URL / SHUNYA_DATABASE_URL)")
    p.add_argument("--source", default="yfinance", help="Stored ohlcv_bars.source tag")
    p.add_argument("--batch-size", type=int, default=40, help="Tickers per yfinance download batch")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between batches (rate limit)")
    p.add_argument("--dry-run", action="store_true", help="Resolve symbols and print counts only; no DB writes")
    args = p.parse_args(argv)

    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2

    durl = args.database_url or os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if not args.dry_run:
        if not durl:
            print("Set DATABASE_URL or pass --database-url", file=sys.stderr)
            return 2
        os.environ["DATABASE_URL"] = durl

    symbols = _parse_sp500_yahoo_symbols()
    if not symbols:
        print("No S&P 500 symbols resolved", file=sys.stderr)
        return 1

    _LOG.info("Resolved %d Yahoo tickers for S&P 500 (USD)", len(symbols))

    spec = BarSpec(BarUnit.DAYS, 1)
    policy = default_bar_index_policy()
    interval_key = bar_spec_to_interval_key(spec)

    prov = YFinanceMarketDataProvider(session=build_yfinance_session())

    if args.dry_run:
        print(f"dry-run: would ingest {len(symbols)} symbols, interval={interval_key}, {args.start} .. {args.end}")
        print("first 15:", ", ".join(symbols[:15]))
        return 0

    try:
        import psycopg
    except ImportError:
        print("Install timescale extra: pip install 'shunya-py[timescale]'", file=sys.stderr)
        return 1

    dsn = get_database_url()
    batches = _chunked(symbols, args.batch_size)
    total_rows = 0
    skipped: list[str] = []

    for bi, batch in enumerate(batches):
        _LOG.info("Batch %d/%d (%d tickers)", bi + 1, len(batches), len(batch))
        try:
            raw = prov.download(
                list(batch),
                args.start,
                args.end,
                bar_spec=spec,
                bar_index_policy=policy,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("batch download failed (%s); retrying per symbol", exc)
            raw = pd.DataFrame()

        if raw.empty:
            for t in batch:
                try:
                    one = _download_one(prov, t, args.start, args.end, spec, policy)
                    sliced, miss = _slice_ohlcv_for_tickers(one, [t])
                    if sliced.empty:
                        skipped.extend(miss if miss else [t])
                        continue
                    n = _upsert_batch(
                        psycopg,
                        dsn,
                        sliced,
                        interval_key,
                        args.source,
                        flat_symbol=t,
                    )
                    total_rows += n
                except Exception as exc2:  # noqa: BLE001
                    _LOG.debug("skip %s: %s", t, exc2)
                    skipped.append(t)
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        present: set[str] = set()
        if isinstance(raw.columns, pd.MultiIndex):
            present = {str(x) for x in raw.columns.get_level_values(0).unique()}
        elif len(batch) == 1:
            present = {batch[0]}

        for t in batch:
            if t not in present:
                skipped.append(t)

        sliced, miss_empty = _slice_ohlcv_for_tickers(raw, batch)
        for t in miss_empty:
            if t not in skipped:
                skipped.append(t)

        if sliced.empty:
            if args.sleep > 0:
                time.sleep(args.sleep)
            continue

        if isinstance(sliced.columns, pd.MultiIndex):
            n = _upsert_batch(psycopg, dsn, sliced, interval_key, args.source, flat_symbol=None)
        else:
            if len(batch) != 1:
                _LOG.error("unexpected flat multi-ticker frame")
                skipped.extend(batch)
                continue
            n = _upsert_batch(psycopg, dsn, sliced, interval_key, args.source, flat_symbol=batch[0])
        total_rows += n

        if args.sleep > 0:
            time.sleep(args.sleep)

    _LOG.info("Upsert finished: %d row touches (approx)", total_rows)
    uniq_skip = sorted({s for s in skipped})
    if uniq_skip:
        _LOG.warning("Skipped %d symbols (no data or error): %s", len(uniq_skip), ", ".join(uniq_skip[:50]))
        if len(uniq_skip) > 50:
            _LOG.warning("... and %d more", len(uniq_skip) - 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
