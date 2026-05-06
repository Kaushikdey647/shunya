"""Upsert Yahoo Finance daily OHLCV into Timescale for index backtest recovery."""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import pandas as pd
from api.settings import env_yfinance_repair_default
from shunya.data.providers import YFinanceMarketDataProvider
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy
from shunya.data.timescale.ingest_lib import UPSERT_OHLCV_SQL, ensure_symbols, rows_from_provider_ohlcv
from shunya.data.timescale.intervals import bar_spec_to_interval_key

from api.db import resolve_database_url

_log = logging.getLogger(__name__)

_DAILY = BarSpec(BarUnit.DAYS, 1)


def tickers_for_ohlcv_backfill(payload: dict[str, Any]) -> list[str]:
    """Constituents + benchmark from stored job payload (post index resolution)."""
    ft = payload.get("fin_ts") or {}
    raw = ft.get("ticker_list") or []
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        s = str(t).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    bench = payload.get("benchmark_ticker")
    if bench is not None:
        b = str(bench).strip()
        if b and b not in seen:
            out.append(b)
    return out


def payload_has_index_code(payload: dict[str, Any]) -> bool:
    ic = payload.get("index_code")
    return isinstance(ic, str) and bool(ic.strip())


def backfill_ohlcv_from_yfinance(
    tickers: Sequence[str],
    *,
    start_date: str,
    end_date_exclusive: str,
    source: str = "yfinance",
    batch_size: int = 40,
) -> tuple[int, str | None]:
    """
    Download daily OHLCV via yfinance and upsert into ``ohlcv_bars``.

    Returns ``(rows_upserted, error_message)`` where ``error_message`` is set only when
    the overall operation fails (partial upserts may have committed before failure).
    """
    import psycopg

    syms = [str(t).strip() for t in tickers if str(t).strip()]
    if not syms:
        return 0, None

    dsn = resolve_database_url()
    interval_key = bar_spec_to_interval_key(_DAILY)
    policy = default_bar_index_policy()
    prov = YFinanceMarketDataProvider(repair=env_yfinance_repair_default())
    total = 0
    run_id: int | None = None

    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_runs (job, provider, params, status)
                    VALUES ('backtest_index_ohlcv_backfill', %s, %s, 'running')
                    RETURNING id
                    """,
                    (
                        source,
                        json.dumps(
                            {
                                "n_tickers": len(syms),
                                "start": start_date,
                                "end_exclusive": end_date_exclusive,
                            }
                        ),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    return 0, "ingestion_runs insert returned no id"
                run_id = int(row[0])
            conn.commit()

        for i in range(0, len(syms), max(1, int(batch_size))):
            batch = syms[i : i + max(1, int(batch_size))]
            try:
                df = prov.download(
                    list(batch),
                    start_date,
                    end_date_exclusive,
                    bar_spec=_DAILY,
                    bar_index_policy=policy,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("yfinance backfill download failed batch=%s: %s", batch[:5], exc)
                with psycopg.connect(dsn) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE ingestion_runs
                            SET finished_at = now(), status = 'failed', error = %s
                            WHERE id = %s
                            """,
                            (str(exc)[:8192], run_id),
                        )
                    conn.commit()
                return total, f"yfinance download failed: {exc!s}"

            if df.empty:
                _log.info("yfinance backfill: empty frame for batch starting %s", batch[:3])
                continue

            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    if isinstance(df.columns, pd.MultiIndex):
                        symbols = [str(t) for t in df.columns.get_level_values(0).unique()]
                    else:
                        if len(batch) != 1:
                            return total, "single-level OHLCV from yfinance requires batch size 1"
                        symbols = [batch[0]]

                    tmap = ensure_symbols(cur, symbols)
                    rows = rows_from_provider_ohlcv(df, tmap, interval=interval_key, source=source)
                    n = 0
                    for chunk_start in range(0, len(rows), 2000):
                        chunk = rows[chunk_start : chunk_start + 2000]
                        cur.executemany(UPSERT_OHLCV_SQL, chunk)
                        n += len(chunk)
                    total += n
                conn.commit()

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingestion_runs
                    SET finished_at = now(), rows_upserted = %s, status = 'ok', error = NULL
                    WHERE id = %s
                    """,
                    (total, run_id),
                )
            conn.commit()
        _log.info("yfinance OHLCV backfill finished: %d rows upserted", total)
        return total, None
    except Exception as exc:  # noqa: BLE001
        _log.exception("yfinance OHLCV backfill failed")
        if run_id is not None:
            try:
                with psycopg.connect(dsn) as conn2:
                    with conn2.cursor() as cur2:
                        cur2.execute(
                            """
                            UPDATE ingestion_runs
                            SET finished_at = now(), status = 'failed', error = %s
                            WHERE id = %s
                            """,
                            (str(exc)[:8192], run_id),
                        )
                    conn2.commit()
            except Exception:  # noqa: BLE001
                pass
        return total, str(exc)
