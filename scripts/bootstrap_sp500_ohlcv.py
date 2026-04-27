#!/usr/bin/env python3
"""
Bootstrap Timescale ``ohlcv_bars`` with daily OHLCV from Yahoo Finance for the union of all
indices exposed by :class:`pytickersymbols.PyTickerSymbols` (see ``INDEX_SOURCE_TO_CODE``).

Symbols resolve to a Yahoo ticker when possible (USD preferred, then any Yahoo listing in
``symbols[]``, else native ``symbol``) so US and international listings are included.

After OHLCV upsert, pulls quarterly fundamentals and yfinance classifications for tickers
that return statement data, then stores index membership (``symbol_index_membership``).

Symbols with no usable bars from yfinance are **skipped** (logged, no failure).

Requires: ``uv sync --extra timescale``, ``DATABASE_URL``, migrations applied
(``shunya-timescale migrate``).

HTTP/TLS matches the rest of Shunya: :func:`~shunya.data.yfinance_session.build_yfinance_session`
(curl_cffi Chrome, ``verify=False`` by default). Set ``YFINANCE_TLS_VERIFY=1`` for strict certificate
verification.

Example::

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
    # Defaults: 2020-01-01 .. 2027-01-01 (exclusive), incremental OHLCV skip from DB coverage
    uv run python scripts/bootstrap_sp500_ohlcv.py

    Full re-download (no DB coverage skip)::

    uv run python scripts/bootstrap_sp500_ohlcv.py --full

    **Incremental Yahoo usage (default):** OHLCV ``download`` runs only for tickers that fail DB
    coverage checks for ``[--start, --end)`` (unless ``--full``). After OHLCV, ``ticker.info`` and
    the fundamentals provider are skipped for tickers that already have recent
    ``symbol_classifications`` / in-window ``fundamentals_field_values`` rows (see
    ``--classification-refresh-days`` and ``--always-fetch-fundamentals``).

    Default: multiple passes (up to 30) until OHLCV coverage heuristics are satisfied; use
    ``--single-pass`` for one pass only. Company rows (``symbol_classifications``) are written
    for every ticker that receives OHLCV; fundamentals when Yahoo statements return data.
    Index memberships (SP100, SP500, …) are synced for the full union at the end of the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from shunya.data.fundamentals import FUNDAMENTAL_FIELDS
from shunya.data.providers import YFinanceMarketDataProvider, fetch_yfinance_classifications
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy, default_bar_spec
from shunya.data.yfinance_session import build_yfinance_session
from shunya.data.timescale.dbutil import get_database_url
from shunya.data.timescale.ingest_lib import (
    UPSERT_FUND_SQL,
    UPSERT_OHLCV_SQL,
    UPSERT_SYMBOL_CLASSIFICATIONS_SQL,
    ensure_symbols,
    fundamentals_eav_rows,
    rows_from_provider_ohlcv,
    symbol_classification_upsert_tuple,
)
from shunya.data.timescale.index_membership_sync import (
    INDEX_SOURCE_TO_CODE,
    load_py_ticker_index_union,
    sync_symbol_index_memberships,
)
from shunya.data.timescale.intervals import bar_spec_to_interval_key

_LOG = logging.getLogger(__name__)


def _parse_index_union_symbols() -> tuple[list[str], dict[str, set[str]], dict[str, str]]:
    """Union of Yahoo tickers + membership map + display names (see ``load_py_ticker_index_union``)."""
    return load_py_ticker_index_union()


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


def _as_date(d: Any) -> date | None:
    if d is None:
        return None
    return pd.Timestamp(d).date()


@dataclass(frozen=True)
class _OhlcvCoverage:
    min_ts: date | None
    max_ts: date | None
    bar_count: int


def _load_ohlcv_coverage(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    *,
    interval_key: str,
    source: str,
    start_d: date,
    end_exclusive_d: date,
) -> dict[str, _OhlcvCoverage]:
    """Per-ticker MIN/MAX ts and bar count for ``[start_d, end_exclusive_d)`` (matches ingest window)."""
    if not tickers:
        return {}
    out: dict[str, _OhlcvCoverage] = {str(t): _OhlcvCoverage(None, None, 0) for t in tickers}
    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker,
                       MIN(b.ts) FILTER (
                         WHERE b.open IS NOT NULL AND b.high IS NOT NULL
                           AND b.low IS NOT NULL AND b.close IS NOT NULL
                           AND b.volume IS NOT NULL AND b.volume >= 0
                       )::date AS min_ts,
                       MAX(b.ts) FILTER (
                         WHERE b.open IS NOT NULL AND b.high IS NOT NULL
                           AND b.low IS NOT NULL AND b.close IS NOT NULL
                           AND b.volume IS NOT NULL AND b.volume >= 0
                       )::date AS max_ts,
                       COUNT(b.ts) FILTER (
                         WHERE b.open IS NOT NULL AND b.high IS NOT NULL
                           AND b.low IS NOT NULL AND b.close IS NOT NULL
                           AND b.volume IS NOT NULL AND b.volume >= 0
                       )::bigint AS n
                FROM symbols s
                LEFT JOIN ohlcv_bars b
                  ON b.symbol_id = s.id
                 AND b.interval = %s
                 AND b.source = %s
                 AND b.ts >= %s::date
                 AND b.ts < %s::date
                WHERE s.ticker = ANY(%s)
                GROUP BY s.id, s.ticker
                """,
                (interval_key, source, start_d, end_exclusive_d, list(tickers)),
            )
            for row in cur.fetchall():
                t, mn, mx, n = row
                out[str(t)] = _OhlcvCoverage(_as_date(mn), _as_date(mx), int(n or 0))
    return out


def _reference_last_bar_date(end_exclusive: date) -> date:
    """Last calendar day we expect bars toward (exclusive end, capped at today)."""
    last_window = end_exclusive - timedelta(days=1)
    return min(date.today(), last_window)


def _business_days_inclusive(a: date, b: date) -> int:
    if a > b:
        return 0
    return len(
        pd.bdate_range(pd.Timestamp(a), pd.Timestamp(b) + timedelta(days=1), inclusive="left")
    )


def _ticker_needs_ohlcv_fetch(
    cov: _OhlcvCoverage,
    start_d: date,
    end_exclusive_d: date,
    *,
    strict_coverage: bool,
) -> bool:
    """
    Decide whether to call Yahoo for this ticker again.

    - Always fetch when there are no rows in the window.
    - Refresh when the latest bar is older than ~7 NYSE sessions before the reference date
      (``min(today, end_exclusive-1d)``), so the DB stays current through the requested window.
    - Optional ``strict_coverage``: bar count vs business days in ``[min_ts, ref]`` is below ~0.76
      (gaps / partial history), ignored when ``min_ts`` is missing.
    """
    if cov.bar_count == 0 or cov.max_ts is None:
        return True
    ref = _reference_last_bar_date(end_exclusive_d)
    if ref < start_d:
        return False
    ref_ts = pd.Timestamp(ref)
    if pd.Timestamp(cov.max_ts) < ref_ts - pd.offsets.BDay(7):
        return True
    if strict_coverage and cov.min_ts is not None:
        span_bd = _business_days_inclusive(cov.min_ts, ref)
        if span_bd > 30 and cov.bar_count < int(0.76 * span_bd):
            return True
    return False


def _filter_tickers_by_db_coverage(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    *,
    interval_key: str,
    source: str,
    start_d: date,
    end_exclusive_d: date,
    strict_coverage: bool,
) -> tuple[list[str], dict[str, _OhlcvCoverage]]:
    cov = _load_ohlcv_coverage(
        psycopg_module,
        dsn,
        tickers,
        interval_key=interval_key,
        source=source,
        start_d=start_d,
        end_exclusive_d=end_exclusive_d,
    )
    need = [
        str(t)
        for t in tickers
        if _ticker_needs_ohlcv_fetch(cov[str(t)], start_d, end_exclusive_d, strict_coverage=strict_coverage)
    ]
    return need, cov


def _tickers_needing_yfinance_classifications(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    *,
    classifications_source: str,
    refresh_if_older_than_days: int,
) -> list[str]:
    """
    Tickers that should trigger ``fetch_yfinance_classifications`` (yfinance ``ticker.info``).

    When ``refresh_if_older_than_days`` > 0, skip tickers that already have a
    ``symbol_classifications`` row for ``classifications_source`` with ``as_of`` on or after
    ``today - refresh_if_older_than_days``. Set to 0 to always refresh everyone in ``tickers``.
    """
    tickers_list = [str(t) for t in tickers if str(t)]
    if not tickers_list:
        return []
    if refresh_if_older_than_days <= 0:
        return tickers_list
    cutoff = date.today() - timedelta(days=refresh_if_older_than_days)
    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.ticker
                FROM unnest(%s::text[]) AS u(ticker)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM symbols s
                    JOIN symbol_classifications c
                      ON c.symbol_id = s.id
                     AND c.source = %s
                     AND c.as_of >= %s::date
                    WHERE s.ticker = u.ticker
                )
                """,
                (tickers_list, classifications_source, cutoff),
            )
            return [str(r[0]) for r in cur.fetchall()]


def _tickers_needing_fundamentals_fetch(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    *,
    fundamentals_source: str,
    window_start: date,
    window_end_exclusive: date,
) -> list[str]:
    """
    Tickers with no quarterly fundamentals rows in ``[window_start, window_end_exclusive)``
    for ``fundamentals_source`` — safe to omit from the statement-provider batch fetch.
    """
    tickers_list = [str(t) for t in tickers if str(t)]
    if not tickers_list:
        return []
    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.ticker
                FROM unnest(%s::text[]) AS u(ticker)
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM symbols s
                    JOIN fundamentals_field_values f
                      ON f.symbol_id = s.id
                     AND f.source = %s
                     AND f.freq = 'quarterly'
                     AND f.period_end >= %s::date
                     AND f.period_end < %s::date
                    WHERE s.ticker = u.ticker
                )
                """,
                (tickers_list, fundamentals_source, window_start, window_end_exclusive),
            )
            return [str(r[0]) for r in cur.fetchall()]


def _ohlcv_tickers_from_frame(sliced: pd.DataFrame, flat_symbol: str | None) -> list[str]:
    if sliced.empty:
        return []
    if isinstance(sliced.columns, pd.MultiIndex):
        return [str(x) for x in sliced.columns.get_level_values(0).unique()]
    if flat_symbol:
        return [flat_symbol]
    return []


def _periodic_for_tickers(periodic: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    if periodic.empty or not tickers:
        return periodic.iloc[0:0]
    want = {str(t) for t in tickers}
    mask = periodic.index.get_level_values(0).astype(str).isin(want)
    return periodic.loc[mask]


def _run_batch_followup(
    psycopg_module: Any,
    dsn: str,
    *,
    ohlcv_tickers: list[str],
    start: str,
    end: str,
    ticker_display_names: dict[str, str],
    classifications_source: str,
    fundamentals_source: str,
    fund_provider: Any,
    fund_bar_spec: BarSpec,
    yfinance_session: Any,
    classification_refresh_days: int,
    skip_fundamentals_if_window_full: bool,
) -> None:
    """
    For every ticker that received OHLCV in this batch:

    - Upsert yfinance company / classification fields (``symbol_classifications``), skipping
      ``ticker.info`` when the DB already has a recent row (see ``classification_refresh_days``).
    - Upsert quarterly fundamentals when the statement provider returns rows; optionally
      omit tickers that already have quarterly fundamentals in the ingest window.
    """
    if not ohlcv_tickers:
        return

    tickers = [str(t) for t in ohlcv_tickers]
    cls_fetch = _tickers_needing_yfinance_classifications(
        psycopg_module,
        dsn,
        tickers,
        classifications_source=classifications_source,
        refresh_if_older_than_days=classification_refresh_days,
    )
    cmap: dict[str, dict[str, Any]] = {}
    if cls_fetch:
        cmap = fetch_yfinance_classifications(cls_fetch, session=yfinance_session)
    elif tickers:
        _LOG.info(
            "Skipping yfinance classifications for all %d tickers (recent symbol_classifications in DB)",
            len(tickers),
        )
    if cls_fetch and len(cls_fetch) < len(tickers):
        _LOG.info(
            "Partial classification refresh: %d/%d tickers need yfinance ticker.info",
            len(cls_fetch),
            len(tickers),
        )

    as_of = date.today()
    fund_window_start = date.fromisoformat(str(start)[:10])
    fund_window_end_excl = date.fromisoformat(str(end)[:10])
    if skip_fundamentals_if_window_full:
        fund_tickers = _tickers_needing_fundamentals_fetch(
            psycopg_module,
            dsn,
            tickers,
            fundamentals_source=fundamentals_source,
            window_start=fund_window_start,
            window_end_exclusive=fund_window_end_excl,
        )
    else:
        fund_tickers = list(tickers)
    if not fund_tickers:
        periodic = pd.DataFrame()
        if tickers and skip_fundamentals_if_window_full:
            _LOG.info(
                "Skipping fundamentals provider for all %d tickers (quarterly rows already in ingest window)",
                len(tickers),
            )
    else:
        if skip_fundamentals_if_window_full and len(fund_tickers) < len(tickers):
            _LOG.info(
                "Partial fundamentals fetch: %d/%d tickers need statement provider",
                len(fund_tickers),
                len(tickers),
            )
        periodic = fund_provider.fetch(
            fund_tickers,
            start,
            end,
            fields=list(FUNDAMENTAL_FIELDS),
            quarterly=True,
            bar_spec=fund_bar_spec,
        )
    fund_syms: set[str] = set()
    if not periodic.empty:
        fund_syms = set(periodic.index.unique(level=0).astype(str))
    complete = [t for t in tickers if t in fund_syms]
    periodic_sub = _periodic_for_tickers(periodic, complete) if complete else periodic.iloc[0:0]

    n_cls = 0
    n_fund = 0
    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES ('bootstrap_sp500_followup', %s, %s, 'running')
                RETURNING id
                """,
                (
                    fundamentals_source,
                    json.dumps({"tickers": tickers, "start": start, "end": end}),
                ),
            )
            run_id = int(cur.fetchone()[0])
            tmap = ensure_symbols(
                cur,
                tickers,
                display_names={t: ticker_display_names.get(t) for t in tickers},
            )
            cls_need = set(cls_fetch)
            for t in tickers:
                if str(t) not in cls_need:
                    continue
                meta = cmap.get(str(t), {})
                cur.execute(
                    UPSERT_SYMBOL_CLASSIFICATIONS_SQL,
                    symbol_classification_upsert_tuple(
                        meta,
                        tmap[str(t)],
                        as_of,
                        classifications_source,
                    ),
                )
                n_cls += 1
            if not periodic_sub.empty:
                fund_rows = fundamentals_eav_rows(
                    periodic_sub,
                    tmap,
                    freq="quarterly",
                    source=fundamentals_source,
                )
                for chunk_start in range(0, len(fund_rows), 5000):
                    chunk = fund_rows[chunk_start : chunk_start + 5000]
                    cur.executemany(UPSERT_FUND_SQL, chunk)
                    n_fund += len(chunk)
            cur.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = now(), rows_upserted = %s, status = 'ok'
                WHERE id = %s
                """,
                (n_fund + n_cls, run_id),
            )
        conn.commit()

    if periodic.empty:
        _LOG.debug("no fundamentals rows for batch subset (%d tickers)", len(tickers))


def _upsert_classifications_only(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    ticker_display_names: dict[str, str],
    classifications_source: str,
    yfinance_session: Any,
    *,
    classification_refresh_days: int,
) -> None:
    """yfinance company / sector fields for ``tickers`` (no fundamentals)."""
    tickers_list = [str(t) for t in tickers if str(t)]
    if not tickers_list:
        return
    need = _tickers_needing_yfinance_classifications(
        psycopg_module,
        dsn,
        tickers_list,
        classifications_source=classifications_source,
        refresh_if_older_than_days=classification_refresh_days,
    )
    if not need:
        _LOG.debug(
            "Skipping yfinance classifications for %d-ticker chunk (all fresh in DB)",
            len(tickers_list),
        )
        return
    if len(need) < len(tickers_list):
        _LOG.info(
            "Constituent metadata: fetching yfinance for %d/%d tickers in chunk (others recent in DB)",
            len(need),
            len(tickers_list),
        )
    cmap = fetch_yfinance_classifications(need, session=yfinance_session)
    as_of = date.today()
    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            tmap = ensure_symbols(
                cur,
                need,
                display_names={t: ticker_display_names.get(str(t)) for t in need},
            )
            for t in need:
                meta = cmap.get(str(t), {})
                cur.execute(
                    UPSERT_SYMBOL_CLASSIFICATIONS_SQL,
                    symbol_classification_upsert_tuple(
                        meta,
                        tmap[str(t)],
                        as_of,
                        classifications_source,
                    ),
                )
        conn.commit()
    _LOG.info("Upserted symbol_classifications for %d tickers (metadata-only pass)", len(need))


def _sync_all_index_memberships(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    membership_sets: dict[str, set[str]],
    ticker_display_names: dict[str, str],
) -> None:
    """Ensure every union constituent has ``symbol_index_membership`` rows (SP500, SP100, …)."""
    sync_symbol_index_memberships(
        psycopg_module,
        dsn,
        tickers,
        membership_sets,
        ticker_display_names,
    )


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
    p.add_argument(
        "--start",
        default="2020-01-01",
        help="Inclusive start date (YYYY-MM-DD); default 2020-01-01",
    )
    p.add_argument(
        "--end",
        default="2027-01-01",
        help="Exclusive end date (YYYY-MM-DD, yfinance-style); default 2027-01-01 (through calendar 2026)",
    )
    p.add_argument("--database-url", default=None, help="Postgres URL (default: DATABASE_URL / SHUNYA_DATABASE_URL)")
    p.add_argument("--source", default="yfinance", help="Stored ohlcv_bars.source tag")
    p.add_argument("--classifications-source", default="yfinance", help="symbol_classifications.source")
    p.add_argument("--fundamentals-source", default="yfinance_statements", help="fundamentals_field_values.source")
    p.add_argument("--batch-size", type=int, default=40, help="Tickers per yfinance download batch")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between batches (rate limit)")
    p.add_argument(
        "--full",
        action="store_true",
        help="Ignore DB coverage and download all tickers (full refresh for the date window)",
    )
    p.add_argument(
        "--no-strict-coverage",
        action="store_true",
        help="Disable bar-count vs business-day density check (default: strict on)",
    )
    p.add_argument(
        "--single-pass",
        action="store_true",
        help="Run at most one OHLCV pass (default: repeat passes until coverage or --max-passes)",
    )
    p.add_argument(
        "--max-passes",
        type=int,
        default=30,
        help="Maximum OHLCV passes when not using --single-pass (default 30)",
    )
    p.add_argument(
        "--pass-sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between OHLCV passes when using multi-pass (default 0.25)",
    )
    p.add_argument(
        "--skip-constituent-classifications",
        action="store_true",
        help="Skip the final yfinance metadata pass for the full union (faster; OHLCV batches still classify)",
    )
    p.add_argument(
        "--classification-refresh-days",
        type=int,
        default=7,
        help="Skip yfinance ticker.info when symbol_classifications for this source has as_of within N days (0=always fetch)",
    )
    p.add_argument(
        "--always-fetch-fundamentals",
        action="store_true",
        help="Always call the fundamentals provider for every OHLCV batch (default: omit tickers with quarterly fundamentals in the ingest window)",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve symbols and print counts only; no DB writes")
    args = p.parse_args(argv)

    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.max_passes < 1:
        print("--max-passes must be >= 1", file=sys.stderr)
        return 2
    try:
        start_d = date.fromisoformat(args.start)
        end_exclusive_d = date.fromisoformat(args.end)
    except ValueError as exc:
        print(f"Invalid --start/--end (use YYYY-MM-DD): {exc}", file=sys.stderr)
        return 2
    if start_d >= end_exclusive_d:
        print("--start must be strictly before --end (end is exclusive)", file=sys.stderr)
        return 2
    if args.classification_refresh_days < 0:
        print("--classification-refresh-days must be >= 0", file=sys.stderr)
        return 2

    durl = args.database_url or os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if not args.dry_run:
        if not durl:
            print("Set DATABASE_URL or pass --database-url", file=sys.stderr)
            return 2
        os.environ["DATABASE_URL"] = durl

    symbols, membership_sets, ticker_display_names = _parse_index_union_symbols()
    strict_coverage = not args.no_strict_coverage
    if not symbols:
        print("No index symbols resolved", file=sys.stderr)
        return 1

    _LOG.info(
        "Resolved %d Yahoo tickers (union of %d PyTickerSymbols indices)",
        len(symbols),
        len(INDEX_SOURCE_TO_CODE),
    )

    spec = BarSpec(BarUnit.DAYS, 1)
    policy = default_bar_index_policy()
    interval_key = bar_spec_to_interval_key(spec)
    fund_bar_spec = default_bar_spec()

    yfinance_session = build_yfinance_session()
    prov = YFinanceMarketDataProvider(session=yfinance_session)

    try:
        from examples.yfinance_fundamental_provider import YFinanceFundamentalDataProvider
    except ImportError:
        YFinanceFundamentalDataProvider = None  # type: ignore[misc, assignment]

    fund_provider: Any = None
    if YFinanceFundamentalDataProvider is not None:
        fund_provider = YFinanceFundamentalDataProvider(
            session=yfinance_session,
            thread_local_session=False,
            enable_fetch_cache=False,
        )

    if args.dry_run:
        n_multi = sum(1 for _u, idxs in membership_sets.items() if len(idxs) > 1)
        print(
            f"dry-run: would ingest {len(symbols)} symbols, interval={interval_key}, {args.start} .. {args.end}"
        )
        print(f"tickers in multiple indices (membership sets): {n_multi}")
        print("first 15:", ", ".join(symbols[:15]))
        return 0

    try:
        import psycopg
    except ImportError:
        print("Install timescale extra: pip install 'shunya-py[timescale]'", file=sys.stderr)
        return 1

    if fund_provider is None:
        print(
            "Could not import examples.yfinance_fundamental_provider; run from repo root with PYTHONPATH=.",
            file=sys.stderr,
        )
        return 1

    dsn = get_database_url()
    total_rows_all = 0
    skipped_all: list[str] = []
    max_passes_eff = args.max_passes if not args.single_pass else 1

    def run_download_pass(symbols_work: list[str]) -> tuple[int, list[str]]:
        batches_local = _chunked(symbols_work, args.batch_size)
        total = 0
        skipped_local: list[str] = []
        for bi, batch in enumerate(batches_local):
            _LOG.info("Batch %d/%d (%d tickers)", bi + 1, len(batches_local), len(batch))
            flat_sym: str | None = None
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
                            skipped_local.extend(miss if miss else [t])
                            continue
                        n = _upsert_batch(
                            psycopg,
                            dsn,
                            sliced,
                            interval_key,
                            args.source,
                            flat_symbol=t,
                        )
                        total += n
                        _run_batch_followup(
                            psycopg,
                            dsn,
                            ohlcv_tickers=[t],
                            start=args.start,
                            end=args.end,
                            ticker_display_names=ticker_display_names,
                            classifications_source=args.classifications_source,
                            fundamentals_source=args.fundamentals_source,
                            fund_provider=fund_provider,
                            fund_bar_spec=fund_bar_spec,
                            yfinance_session=yfinance_session,
                            classification_refresh_days=args.classification_refresh_days,
                            skip_fundamentals_if_window_full=not args.always_fetch_fundamentals,
                        )
                    except Exception as exc2:  # noqa: BLE001
                        _LOG.debug("skip %s: %s", t, exc2)
                        skipped_local.append(t)
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
                    skipped_local.append(t)

            sliced, miss_empty = _slice_ohlcv_for_tickers(raw, batch)
            for t in miss_empty:
                if t not in skipped_local:
                    skipped_local.append(t)

            if sliced.empty:
                if args.sleep > 0:
                    time.sleep(args.sleep)
                continue

            if isinstance(sliced.columns, pd.MultiIndex):
                n = _upsert_batch(psycopg, dsn, sliced, interval_key, args.source, flat_symbol=None)
            else:
                if len(batch) != 1:
                    _LOG.error("unexpected flat multi-ticker frame")
                    skipped_local.extend(batch)
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    continue
                n = _upsert_batch(psycopg, dsn, sliced, interval_key, args.source, flat_symbol=batch[0])
                flat_sym = batch[0]
            total += n

            ohlcv_ok = _ohlcv_tickers_from_frame(sliced, flat_sym)
            _run_batch_followup(
                psycopg,
                dsn,
                ohlcv_tickers=ohlcv_ok,
                start=args.start,
                end=args.end,
                ticker_display_names=ticker_display_names,
                classifications_source=args.classifications_source,
                fundamentals_source=args.fundamentals_source,
                fund_provider=fund_provider,
                fund_bar_spec=fund_bar_spec,
                yfinance_session=yfinance_session,
                classification_refresh_days=args.classification_refresh_days,
                skip_fundamentals_if_window_full=not args.always_fetch_fundamentals,
            )

            if args.sleep > 0:
                time.sleep(args.sleep)
        return total, skipped_local

    pass_idx = 0
    while True:
        pass_idx += 1
        if args.full:
            symbols_work = list(symbols)
        else:
            symbols_work, _ = _filter_tickers_by_db_coverage(
                psycopg,
                dsn,
                symbols,
                interval_key=interval_key,
                source=args.source,
                start_d=start_d,
                end_exclusive_d=end_exclusive_d,
                strict_coverage=strict_coverage,
            )
        if not symbols_work:
            if pass_idx == 1:
                _LOG.info(
                    "No OHLCV refresh needed: %d tickers already satisfy coverage for %s .. %s",
                    len(symbols),
                    start_d,
                    end_exclusive_d,
                )
            break
        if pass_idx > max_passes_eff:
            _LOG.warning(
                "Stopping after %d pass(es): %d tickers still flagged incomplete; sample: %s",
                max_passes_eff,
                len(symbols_work),
                ", ".join(symbols_work[:30]) + (" ..." if len(symbols_work) > 30 else ""),
            )
            break
        _LOG.info(
            "Pass %d / %d: fetching %d of %d tickers",
            pass_idx,
            max_passes_eff,
            len(symbols_work),
            len(symbols),
        )
        rows_pass, skip_pass = run_download_pass(symbols_work)
        total_rows_all += rows_pass
        skipped_all.extend(skip_pass)
        _LOG.info("Pass %d finished (~%d OHLCV row touches)", pass_idx, rows_pass)
        if args.pass_sleep > 0 and not args.single_pass and not args.full:
            time.sleep(args.pass_sleep)
        if args.single_pass or args.full:
            break

    _sync_all_index_memberships(
        psycopg,
        dsn,
        symbols,
        membership_sets,
        ticker_display_names,
    )

    if not args.skip_constituent_classifications:
        cls_chunk = 20
        for ci, chunk in enumerate(_chunked(symbols, cls_chunk)):
            _LOG.info(
                "Constituent company metadata %d-%d / %d",
                ci * cls_chunk + 1,
                min((ci + 1) * cls_chunk, len(symbols)),
                len(symbols),
            )
            _upsert_classifications_only(
                psycopg,
                dsn,
                chunk,
                ticker_display_names,
                args.classifications_source,
                yfinance_session,
                classification_refresh_days=args.classification_refresh_days,
            )
            if len(symbols) > cls_chunk:
                time.sleep(max(args.sleep * 0.25, 0.08))

    _LOG.info("Upsert finished: %d row touches (approx)", total_rows_all)
    uniq_skip = sorted({s for s in skipped_all})
    if uniq_skip:
        _LOG.warning("Skipped %d symbols (no data or error): %s", len(uniq_skip), ", ".join(uniq_skip[:50]))
        if len(uniq_skip) > 50:
            _LOG.warning("... and %d more", len(uniq_skip) - 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
