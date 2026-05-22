#!/usr/bin/env python3
"""
Bootstrap TimescaleDB for **S&P 100** index backtests (``index_code`` = ``SP100``).

Steps:
  1. Apply SQL migrations (unless ``--skip-migrate``).
  2. Sync ``symbol_index_membership`` from PyTickerSymbols (all catalog indices; lightweight).
  3. Load daily OHLCV for **SP100 constituents + benchmark** ``^OEX`` into ``ohlcv_bars``
     (default window: **2020-01-01 through today**, ``[start, end)`` half-open; override with
     ``--start`` / ``--end``). ``^OEX`` is always downloaded with constituents. ``GS-PK`` is
     skipped (not in yfinance OHLCV set for this bootstrap).
  4. Ingest quarterly/wide fundamentals from Yahoo over a **longer** window (OHLCV start minus
     two years) so prior quarters exist for forward-fill at the start of the bar window.
  5. Ingest yfinance sector/industry classifications for constituents.
  6. Verify OHLCV coverage (benchmark must have bars; constituents warn or fail with ``--strict``).

**Forward-fill:** Fundamentals are stored as sparse periodic rows. When you attach fundamentals to
daily bars (``finTs`` / :func:`~shunya.data.fundamentals.align_fundamental_panel_to_panel_index`),
values are forward-filled onto each bar date — not expanded as a dense daily table in Postgres.

Requires:
  - ``uv sync --extra timescale`` and ``DATABASE_URL`` (or ``SHUNYA_DATABASE_URL``).
    The timescale extra includes **Rich** for colored output, progress bars, and spinners.
  - Run from a **clone** of this repo (``examples/yfinance_fundamental_provider`` for fundamentals).

Example::

    export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shunya
    uv run python scripts/bootstrap_sp100_timescale.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID
from shunya.data.providers import YFinanceMarketDataProvider
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy, default_bar_spec
from shunya.data.timescale.cli import (
    cmd_ingest_classifications,
    cmd_ingest_fundamentals,
    cmd_sync_index_memberships,
)
from shunya.data.timescale.dbutil import apply_migrations, get_database_url
from shunya.data.timescale.ingest_lib import (
    UPSERT_OHLCV_SQL,
    ensure_symbols,
    rows_from_provider_ohlcv,
)
from shunya.data.timescale.intervals import bar_spec_to_interval_key
from shunya.data.timescale.market_cache_lib import touch_ohlcv_refresh_manifest_on_cursor
from shunya.data.yfinance_session import build_yfinance_session

_LOG = logging.getLogger(__name__)

_T = TypeVar("_T")


def _try_console() -> tuple[Any | None, bool]:
    try:
        from rich.console import Console
        from rich.theme import Theme

        theme = Theme(
            {
                "info": "cyan",
                "warn": "yellow",
                "err": "bold red",
                "ok": "bold green",
                "accent": "bold cyan",
                "muted": "dim white",
            }
        )
        return Console(theme=theme, highlight=False, soft_wrap=True), True
    except ImportError:
        return None, False


def _install_logging(console: Any | None) -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    if console is not None:
        from rich.logging import RichHandler

        root.addHandler(
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                markup=True,
                show_time=True,
            )
        )
        root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _print_banner(
    console: Any | None,
    *,
    start_s: str,
    end_s: str,
    fund_start_s: str,
    dry_run: bool,
) -> None:
    if console is None:
        print(f"\n=== SP100 Timescale bootstrap ===\nOHLCV [{start_s}, {end_s})\n", file=sys.stderr)
        return
    from rich.panel import Panel
    from rich.table import Table

    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(justify="right", style="muted")
    tbl.add_column(style="white")
    tbl.add_row("OHLCV window", f"[accent]{start_s}[/] → [accent]{end_s}[/] [dim](end exclusive)[/]")
    tbl.add_row("Fundamentals from", f"[accent]{fund_start_s}[/]")
    tbl.add_row("Benchmark", f"[bold]{BENCHMARK_TICKER}[/]")
    excl = ", ".join(sorted(_TICKERS_EXCLUDED_FROM_BOOTSTRAP)) or "—"
    tbl.add_row("Excluded tickers", f"[warn]{excl}[/]")
    tbl.add_row("Mode", "[yellow]dry-run[/]" if dry_run else "[green]live[/]")
    console.print()
    console.print(
        Panel.fit(
            tbl,
            title="[accent]Shunya[/] · [bold white]SP100 × Timescale[/]",
            subtitle="[dim]yfinance → Postgres / TimescaleDB[/]",
            border_style="cyan",
        )
    )
    console.print()


def _emit(console: Any | None, msg: str) -> None:
    if console is None:
        print(msg, file=sys.stderr)
    else:
        console.print(msg)


def _emit_error(console: Any | None, msg: str) -> None:
    if console is None:
        print(f"ERROR: {msg}", file=sys.stderr)
    else:
        console.print(f"[err]✗[/] {msg}")


def _emit_warn_box(console: Any | None, title: str, body: str) -> None:
    if console is None:
        print(f"WARNING: {title}\n{body}", file=sys.stderr)
        return
    from rich.panel import Panel

    console.print(Panel(body, title=f"[warn]{title}[/]", border_style="yellow"))


def _emit_success_panel(console: Any | None, title: str, lines: list[tuple[str, str]]) -> None:
    if console is None:
        print(f"OK: {title}", file=sys.stderr)
        for k, v in lines:
            print(f"  {k}: {v}", file=sys.stderr)
        return
    from rich.panel import Panel
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column(style="muted", justify="right")
    t.add_column(style="white")
    for k, v in lines:
        t.add_row(k, v)
    console.print()
    console.print(Panel(t, title=f"[ok]{title}[/]", border_style="green"))


def _with_status(console: Any | None, message: str, fn: Callable[[], _T]) -> _T:
    if console is None:
        return fn()
    with console.status(message, spinner="dots12", spinner_style="cyan"):
        return fn()


def _coverage_summary(
    console: Any | None,
    counts: dict[str, int],
    *,
    constituents: list[str],
    benchmark: str,
) -> None:
    """Benchmark + constituent coverage without printing 100+ tickers."""
    bench_n = counts.get(benchmark, 0)
    missing = [t for t in constituents if counts.get(t, 0) < 1]
    ok_n = len(constituents) - len(missing)
    if console is None:
        print(
            f"Benchmark {benchmark}: {bench_n} bars; constituents with bars: {ok_n}/{len(constituents)}",
            file=sys.stderr,
        )
        if missing:
            print(f"Missing OHLCV: {', '.join(missing[:25])}", file=sys.stderr)
        return
    from rich.panel import Panel
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 1))
    t.add_column(style="cyan", justify="right")
    t.add_column(style="white")
    t.add_row("Benchmark", f"[bold]{benchmark}[/]  [accent]{bench_n:,}[/] bars")
    t.add_row("Constituents", f"[ok]{ok_n}[/] / {len(constituents)} with ≥1 bar")
    if missing:
        tail = ", ".join(missing[:20]) + (" …" if len(missing) > 20 else "")
        t.add_row("[warn]Missing[/]", tail)
    console.print(Panel(t, title="[accent]OHLCV coverage[/]", border_style="blue"))


INDEX_CODE = "SP100"
# Must match ``api.index_catalog.RAW_INDEX_TICKER_BY_CODE["SP100"]``.
BENCHMARK_TICKER = "^OEX"
# Yahoo OHLCV rows in ``ohlcv_bars`` use ``STORED_OHLCV_DEFAULT_UPSTREAM_ID`` (see ``api/index_universe``).
_DEFAULT_CHUNK = 35
_FUND_LOOKBACK_YEARS = 2
# Default OHLCV window: full history from 2020 through today (index backtests / ^OEX alignment).
_DEFAULT_OHLCV_START = date(2020, 1, 1)
# Membership rows to skip (e.g. preferred series not available on Yahoo OHLCV for this job).
_TICKERS_EXCLUDED_FROM_BOOTSTRAP = frozenset({"GS-PK"})


def _chunked(seq: Sequence[str], size: int) -> list[list[str]]:
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def _ticker_frame_has_close_bars(part: pd.DataFrame) -> bool:
    if part is None or part.empty:
        return False
    if "Close" not in part.columns:
        return False
    return bool(part["Close"].notna().any())


def _slice_ohlcv_for_tickers(raw: pd.DataFrame, requested: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
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


def _filter_bootstrap_constituents(tickers: Sequence[str]) -> list[str]:
    return [str(t) for t in tickers if str(t) not in _TICKERS_EXCLUDED_FROM_BOOTSTRAP]


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


def _ohlcv_bar_counts_in_window(
    dsn: str,
    tickers: list[str],
    *,
    start_date: str,
    end_date: str,
    interval: str,
    source: str = STORED_OHLCV_DEFAULT_UPSTREAM_ID,
) -> dict[str, int]:
    import psycopg

    t0 = pd.Timestamp(start_date).normalize()
    t1 = pd.Timestamp(end_date).normalize()
    out: dict[str, int] = {str(t): 0 for t in tickers}
    if not tickers:
        return out
    with psycopg.connect(dsn) as conn:
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


def _upsert_ohlcv_batch(
    dsn: str,
    df: pd.DataFrame,
    interval_key: str,
    source: str,
    *,
    flat_symbol: str | None,
) -> int:
    import psycopg

    if df.empty:
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        symbols = [str(t) for t in df.columns.get_level_values(0).unique()]
    else:
        if not flat_symbol:
            raise ValueError("flat OHLCV requires flat_symbol")
        symbols = [flat_symbol]

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES ('bootstrap_sp100_ohlcv', %s, %s, 'running')
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
            for _sym_t, sym_id in tmap.items():
                touch_ohlcv_refresh_manifest_on_cursor(
                    cur, symbol_id=sym_id, interval=interval_key, source=source
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
    return n


def _download_one(
    prov: YFinanceMarketDataProvider,
    symbol: str,
    start: str,
    end: str,
    spec: BarSpec,
    policy: object,
) -> pd.DataFrame:
    return prov.download([symbol], start, end, bar_spec=spec, bar_index_policy=policy)


def _ingest_ohlcv_chunked(
    dsn: str,
    symbols: list[str],
    start: str,
    end: str,
    *,
    chunk_size: int,
    sleep_s: float,
    repair: bool,
    console: Any | None = None,
) -> tuple[int, list[str]]:
    spec = BarSpec(BarUnit.DAYS, 1)
    policy = default_bar_index_policy()
    interval_key = bar_spec_to_interval_key(spec)
    session = build_yfinance_session()
    prov = YFinanceMarketDataProvider(session=session, repair=repair)
    total = 0
    skipped_all: list[str] = []
    batches = _chunked(symbols, chunk_size)
    batch_iter: Any = enumerate(batches)
    if console is not None:
        try:
            from rich.progress import track

            batch_iter = enumerate(
                track(
                    batches,
                    description="[bold cyan]OHLCV batches[/]",
                    console=console,
                    transient=False,
                )
            )
        except ImportError:
            batch_iter = enumerate(batches)

    for bi, batch in batch_iter:
        _LOG.info("OHLCV batch %d/%d (%d tickers)", bi + 1, len(batches), len(batch))
        flat_sym: str | None = None
        try:
            raw = prov.download(
                list(batch),
                start,
                end,
                bar_spec=spec,
                bar_index_policy=policy,
                repair=repair,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("batch download failed (%s); retrying per symbol", exc)
            raw = pd.DataFrame()

        if raw.empty:
            for t in batch:
                try:
                    one = _download_one(prov, t, start, end, spec, policy)
                    sliced, miss = _slice_ohlcv_for_tickers(one, [t])
                    skipped_all.extend(miss)
                    if sliced.empty:
                        continue
                    n = _upsert_ohlcv_batch(
                        dsn,
                        sliced,
                        interval_key,
                        STORED_OHLCV_DEFAULT_UPSTREAM_ID,
                        flat_symbol=t if not isinstance(sliced.columns, pd.MultiIndex) else None,
                    )
                    total += n
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("single-ticker download failed %s: %s", t, exc)
                    skipped_all.append(str(t))
        else:
            sliced, miss = _slice_ohlcv_for_tickers(raw, batch)
            skipped_all.extend(miss)
            if not sliced.empty:
                if isinstance(sliced.columns, pd.MultiIndex):
                    flat_sym = None
                elif len(batch) == 1:
                    flat_sym = str(batch[0])
                else:
                    flat_sym = None
                total += _upsert_ohlcv_batch(dsn, sliced, interval_key, STORED_OHLCV_DEFAULT_UPSTREAM_ID, flat_symbol=flat_sym)

        if sleep_s > 0 and bi + 1 < len(batches):
            time.sleep(sleep_s)

    return total, skipped_all


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--database-url", default=None, help="Postgres URL (default: DATABASE_URL / SHUNYA_DATABASE_URL)")
    p.add_argument("--skip-migrate", action="store_true", help="Do not run SQL migrations")
    p.add_argument(
        "--start",
        default=None,
        help=f"Inclusive OHLCV start YYYY-MM-DD (default: {_DEFAULT_OHLCV_START.isoformat()})",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Exclusive OHLCV end YYYY-MM-DD (default: today)",
    )
    p.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK, help="Tickers per yfinance OHLCV batch")
    p.add_argument("--sleep", type=float, default=0.2, help="Seconds between OHLCV batches")
    p.add_argument("--skip-fundamentals", action="store_true")
    p.add_argument("--skip-classifications", action="store_true")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any SP100 constituent has no OHLCV in the window (benchmark always required)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print plan only; no DB writes")
    p.add_argument(
        "--no-yfinance-repair",
        action="store_true",
        help="Disable yfinance price repair (overrides env defaults)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    console, has_rich = _try_console()
    _install_logging(console)
    if not has_rich:
        print(
            "Tip: uv sync --extra timescale — includes Rich for colors, spinners, and progress bars.",
            file=sys.stderr,
        )

    if args.chunk_size < 1:
        _emit_error(console, "--chunk-size must be >= 1")
        return 2

    durl = args.database_url or os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if not args.dry_run and not durl:
        _emit_error(console, "Set DATABASE_URL or SHUNYA_DATABASE_URL, or pass --database-url")
        return 2
    if durl:
        os.environ["DATABASE_URL"] = durl

    try:
        end_d = date.fromisoformat(args.end) if args.end else date.today()
        if args.start:
            start_d = date.fromisoformat(args.start)
        else:
            start_d = _DEFAULT_OHLCV_START
    except ValueError as exc:
        _emit_error(console, f"Invalid --start/--end (use YYYY-MM-DD): {exc}")
        return 2

    if start_d >= end_d:
        _emit_error(console, "--start must be strictly before --end (end is exclusive)")
        return 2

    start_s = start_d.isoformat()
    end_s = end_d.isoformat()
    fund_start_d = start_d - timedelta(days=365 * _FUND_LOOKBACK_YEARS + 5)
    fund_start_s = fund_start_d.isoformat()

    dsn = durl or ""
    if not args.dry_run:
        dsn = get_database_url()

    _print_banner(console, start_s=start_s, end_s=end_s, fund_start_s=fund_start_s, dry_run=args.dry_run)

    if args.dry_run:
        if console is not None:
            from rich.panel import Panel

            lines = [
                f"[bold]OHLCV[/]  [accent]{start_s}[/] → [accent]{end_s}[/]  [dim](end exclusive)[/]",
                f"[bold]Fundamentals[/]  [accent]{fund_start_s}[/] → [accent]{end_s}[/]",
            ]
            if durl:
                try:
                    cons_raw = _constituent_tickers_sp100(dsn)
                    cons = _filter_bootstrap_constituents(cons_raw)
                    ohlcv_syms = list(dict.fromkeys([*cons, BENCHMARK_TICKER]))
                    lines.append(
                        f"[bold]Symbols[/]  [ok]{len(ohlcv_syms)}[/] OHLCV targets "
                        f"([dim]{len(cons_raw)} DB members → {len(cons)} after exclusions[/])"
                    )
                    if cons:
                        lines.append(f"[bold]Sample[/]  [dim]{', '.join(cons[:12])}[/]")
                except Exception as exc:  # noqa: BLE001
                    lines.append(f"[warn]Could not read membership:[/] {exc}")
            else:
                lines.append("[warn]Set DATABASE_URL[/] to preview membership from the DB.")
            console.print(Panel("\n".join(lines), title="[yellow]Dry run[/]", border_style="yellow"))
        else:
            _LOG.info("dry-run: OHLCV [%s, %s); fundamentals [%s, %s)", start_s, end_s, fund_start_s, end_s)
            if durl:
                try:
                    cons_raw = _constituent_tickers_sp100(dsn)
                    cons = _filter_bootstrap_constituents(cons_raw)
                    ohlcv_syms = list(dict.fromkeys([*cons, BENCHMARK_TICKER]))
                    _LOG.info(
                        "dry-run: %d DB members, %d after exclusions, %d OHLCV symbols",
                        len(cons_raw),
                        len(cons),
                        len(ohlcv_syms),
                    )
                except Exception as exc:  # noqa: BLE001
                    _LOG.info("dry-run: membership query failed: %s", exc)
        return 0

    repair = not args.no_yfinance_repair

    if not args.skip_migrate:

        def _migrate() -> None:
            apply_migrations(dsn=dsn)

        _with_status(console, "[bold cyan]Applying SQL migrations…[/]", _migrate)
        _LOG.info("Migrations applied")

    def _sync() -> int:
        return cmd_sync_index_memberships(argparse.Namespace())

    rc_sync = _with_status(console, "[bold cyan]Syncing index memberships (PyTickerSymbols)…[/]", _sync)
    if rc_sync != 0:
        _emit_error(console, f"sync-index-memberships exited with code {rc_sync}")
        return rc_sync
    _LOG.info("Index memberships synced")

    constituents_raw = _constituent_tickers_sp100(dsn)
    if not constituents_raw:
        _emit_error(
            console,
            f"No symbol_index_membership rows for {INDEX_CODE}. "
            "Sync ran but the index is empty — check PyTickerSymbols / DB.",
        )
        return 1
    constituents = _filter_bootstrap_constituents(constituents_raw)
    skipped_members = [t for t in constituents_raw if t not in constituents]
    if skipped_members:
        body = ", ".join(skipped_members)
        if console is not None:
            _emit_warn_box(console, "Excluded from bootstrap", body)
        else:
            _LOG.info("Excluded ticker(s): %s", body)
    _LOG.info("SP100 constituents: %d from DB → %d after exclusions", len(constituents_raw), len(constituents))

    ohlcv_symbols = list(dict.fromkeys([*constituents, BENCHMARK_TICKER]))
    _LOG.info("OHLCV download: %d symbols in [%s .. %s)", len(ohlcv_symbols), start_s, end_s)

    total_bars, skipped = _ingest_ohlcv_chunked(
        dsn,
        ohlcv_symbols,
        start_s,
        end_s,
        chunk_size=int(args.chunk_size),
        sleep_s=float(args.sleep),
        repair=repair,
        console=console,
    )
    if skipped:
        _emit_warn_box(
            console,
            "Tickers with no usable Close",
            ", ".join(skipped[:40]) + (" …" if len(skipped) > 40 else ""),
        )
    _LOG.info("OHLCV upsert finished (%d bar rows upserted this run)", total_bars)

    if not args.skip_fundamentals:
        ns_f = argparse.Namespace(
            symbols=" ".join(constituents),
            start=fund_start_s,
            end=end_s,
            source="yfinance_statements",
            provider="yfinance",
            yearly=False,
            fields="",
            insecure_curl=False,
        )

        def _fund() -> int:
            return cmd_ingest_fundamentals(ns_f)

        rc = _with_status(
            console,
            f"[bold cyan]Ingesting fundamentals[/] [dim]({fund_start_s} → {end_s})[/]…",
            _fund,
        )
        if rc != 0:
            _emit_error(console, f"ingest-fundamentals exited with code {rc}")
            return rc

    if not args.skip_classifications:
        ns_c = argparse.Namespace(
            symbols=" ".join(constituents),
            source="yfinance",
            insecure_curl=False,
        )

        def _cls() -> int:
            return cmd_ingest_classifications(ns_c)

        rc = _with_status(console, "[bold cyan]Ingesting sector / industry classifications…[/]", _cls)
        if rc != 0:
            _emit_error(console, f"ingest-classifications exited with code {rc}")
            return rc

    interval_key = bar_spec_to_interval_key(default_bar_spec())
    counts = _ohlcv_bar_counts_in_window(
        dsn,
        ohlcv_symbols,
        start_date=start_s,
        end_date=end_s,
        interval=interval_key,
        source=STORED_OHLCV_DEFAULT_UPSTREAM_ID,
    )
    _coverage_summary(console, counts, constituents=constituents, benchmark=BENCHMARK_TICKER)

    bench_n = counts.get(BENCHMARK_TICKER, 0)
    if bench_n < 1:
        _emit_error(
            console,
            f"Benchmark {BENCHMARK_TICKER} has no OHLCV in [{start_s}, {end_s}); index backtests will fail.",
        )
        return 1

    missing_c = [t for t in constituents if counts.get(t, 0) < 1]
    if missing_c:
        msg = (
            f"{len(missing_c)} constituent(s) have no OHLCV in window (interval={interval_key!r}): "
            f"{', '.join(missing_c[:12])}"
        )
        if args.strict:
            _emit_error(console, msg)
            return 1
        _emit_warn_box(console, "Coverage gaps (non-strict)", msg)

    _emit_success_panel(
        console,
        "Bootstrap complete",
        [
            ("Universe", f"SP100 + {BENCHMARK_TICKER}"),
            ("OHLCV window", f"{start_s} → {end_s} (exclusive end)"),
            ("Bars upserted (this run)", f"{total_bars:,}"),
            ("Constituents w/ bars", f"{len(constituents) - len(missing_c)} / {len(constituents)}"),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
