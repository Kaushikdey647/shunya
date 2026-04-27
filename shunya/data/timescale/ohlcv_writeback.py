"""Replace OHLCV rows in a UTC window after a yfinance refresh (instrument API path)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from .ingest_lib import UPSERT_OHLCV_SQL, ensure_symbols, rows_from_provider_ohlcv

_log = logging.getLogger(__name__)

DELETE_OHLCV_RANGE_SQL = """
DELETE FROM ohlcv_bars
WHERE symbol_id = %s AND interval = %s AND source = %s AND ts >= %s AND ts < %s
"""


def replace_ohlcv_range_sync(
    dsn: str,
    *,
    symbol: str,
    interval_key: str,
    source: str,
    start_inclusive: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    ohlcv_df: pd.DataFrame,
    job: str = "api_ohlcv_replace",
    ingestion_run_id: int | None = None,
) -> tuple[int, int]:
    """
    Delete bars in ``[start_inclusive, end_exclusive)`` then upsert from ``ohlcv_df``.

    If ``ingestion_run_id`` is set, that row must already exist with status ``running`` (deferred path).
    Otherwise a new ``ingestion_runs`` row is created.

    Returns ``(ingestion_run_id, rows_upserted)``.
    """
    import psycopg

    t0 = pd.Timestamp(start_inclusive)
    t1 = pd.Timestamp(end_exclusive)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    else:
        t1 = t1.tz_convert("UTC")

    params = json.dumps(
        {
            "symbol": symbol,
            "interval": interval_key,
            "start": t0.isoformat(),
            "end_exclusive": t1.isoformat(),
        }
    )

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            if ingestion_run_id is None:
                cur.execute(
                    """
                    INSERT INTO ingestion_runs (job, provider, params, status)
                    VALUES (%s, %s, %s, 'running')
                    RETURNING id
                    """,
                    (job, source, params),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("ingestion_runs insert returned no id")
                run_id = int(row[0])
            else:
                run_id = ingestion_run_id

            try:
                tmap = ensure_symbols(cur, [symbol])
                cur.execute(
                    DELETE_OHLCV_RANGE_SQL,
                    (tmap[symbol], interval_key, source, t0.to_pydatetime(), t1.to_pydatetime()),
                )
                rows = rows_from_provider_ohlcv(ohlcv_df, tmap, interval=interval_key, source=source)
                n = 0
                for chunk_start in range(0, len(rows), 2000):
                    chunk = rows[chunk_start : chunk_start + 2000]
                    cur.executemany(UPSERT_OHLCV_SQL, chunk)
                    n += len(chunk)
                cur.execute(
                    """
                    UPDATE ingestion_runs
                    SET finished_at = now(), rows_upserted = %s, status = 'ok', error = NULL
                    WHERE id = %s
                    """,
                    (n, run_id),
                )
                conn.commit()
                return run_id, n
            except Exception as exc:  # noqa: BLE001
                _log.warning("ohlcv writeback failed run_id=%s: %s", run_id, exc)
                cur.execute(
                    """
                    UPDATE ingestion_runs
                    SET finished_at = now(), status = 'failed', error = %s
                    WHERE id = %s
                    """,
                    (str(exc)[:8192], run_id),
                )
                conn.commit()
                raise


def get_ingestion_run_sync(dsn: str, run_id: int) -> dict[str, Any] | None:
    """Return one ``ingestion_runs`` row as a dict, or ``None`` if missing."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, job, started_at, finished_at, provider, params, rows_upserted, status, error
                FROM ingestion_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            r = cur.fetchone()
            if r is None:
                return None
            return {
                "id": r[0],
                "job": r[1],
                "started_at": r[2].isoformat() if r[2] is not None else None,
                "finished_at": r[3].isoformat() if r[3] is not None else None,
                "provider": r[4],
                "params": r[5],
                "rows_upserted": r[6],
                "status": r[7],
                "error": r[8],
            }


def create_deferred_ingestion_run_sync(dsn: str, *, source: str, job: str, params: dict[str, Any]) -> int:
    """Insert ``running`` ingestion row; caller schedules :func:`replace_ohlcv_range_sync` with this id."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (job, provider, params, status)
                VALUES (%s, %s, %s, 'running')
                RETURNING id
                """,
                (job, source, json.dumps(params)),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("ingestion_runs insert returned no id")
            run_id = int(row[0])
            conn.commit()
            return run_id
