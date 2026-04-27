from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from psycopg.rows import dict_row

from backtest_api.db import resolve_database_url
from backtest_api.schemas.models import BacktestCreate, BacktestJobOut


def _job_row_to_out(row: dict[str, Any]) -> BacktestJobOut:
    ic = row.get("index_code")
    if ic is not None and isinstance(ic, str) and not ic.strip():
        ic = None
    inc = row.get("include_test_period_in_results")
    if isinstance(inc, bool):
        include_test = inc
    elif inc is None:
        include_test = False
    else:
        include_test = str(inc).lower() in ("true", "1", "t", "yes")
    return BacktestJobOut(
        id=str(row["id"]),
        alpha_id=str(row["alpha_id"]),
        alpha_name=row.get("alpha_name"),
        index_code=str(ic) if ic is not None else None,
        include_test_period_in_results=include_test,
        status=row["status"],
        error_message=row.get("error_message"),
        result_summary=row.get("result_summary") if row.get("result_summary") is not None else None,
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )


def insert_job(payload: BacktestCreate) -> BacktestJobOut:
    import psycopg

    raw = payload.model_dump(mode="json")
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO api_backtest_jobs (alpha_id, status, request_payload)
                VALUES (%s, 'queued', %s::jsonb)
                RETURNING id
                """,
                (payload.alpha_id, json.dumps(raw)),
            )
            ins = cur.fetchone()
        conn.commit()
    if ins is None:
        raise RuntimeError("insert_job: INSERT did not return id")
    job_id = str(ins["id"])
    enriched = get_job(job_id)
    if enriched is None:
        raise RuntimeError("insert_job: row missing after insert")
    return enriched


def list_jobs(
    *,
    limit: int = 50,
    offset: int = 0,
    alpha_id: Optional[str] = None,
    status: Optional[str] = None,
) -> list[BacktestJobOut]:
    import psycopg

    where: list[str] = []
    params: list[Any] = []
    if alpha_id:
        where.append("j.alpha_id = %s")
        params.append(alpha_id)
    if status:
        where.append("j.status = %s")
        params.append(status)
    wh = ("WHERE " + " AND ".join(where)) if where else ""
    params.extend([limit, offset])
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT j.id, j.alpha_id, j.status, j.error_message, j.result_summary,
                       j.created_at, j.started_at, j.finished_at,
                       a.name AS alpha_name,
                       NULLIF(j.request_payload->>'index_code', '') AS index_code,
                       COALESCE((j.request_payload->>'include_test_period_in_results')::boolean, false)
                         AS include_test_period_in_results
                FROM api_backtest_jobs j
                LEFT JOIN api_alphas a ON a.id = j.alpha_id
                {wh}
                ORDER BY j.created_at DESC
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
    return [_job_row_to_out(r) for r in rows]


def get_job(job_id: str) -> Optional[BacktestJobOut]:
    import psycopg

    try:
        UUID(job_id)
    except ValueError:
        return None
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT j.id, j.alpha_id, j.status, j.error_message, j.result_summary,
                       j.created_at, j.started_at, j.finished_at,
                       a.name AS alpha_name,
                       NULLIF(j.request_payload->>'index_code', '') AS index_code,
                       COALESCE((j.request_payload->>'include_test_period_in_results')::boolean, false)
                         AS include_test_period_in_results
                FROM api_backtest_jobs j
                LEFT JOIN api_alphas a ON a.id = j.alpha_id
                WHERE j.id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
    return _job_row_to_out(row) if row else None


def claim_next_queued_job() -> Optional[tuple[str, dict[str, Any]]]:
    """Atomically set one queued job to running; return (job_id, request_payload) or None."""
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                WITH c AS (
                    SELECT id FROM api_backtest_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE api_backtest_jobs j
                SET status = 'running', started_at = now()
                FROM c
                WHERE j.id = c.id
                RETURNING j.id, j.request_payload
                """
            )
            row = cur.fetchone()
        conn.commit()
    if not row:
        return None
    payload = row["request_payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return str(row["id"]), dict(payload)


def mark_job_succeeded(job_id: str, result: dict[str, Any], summary: dict[str, Any]) -> None:
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_backtest_jobs
                SET status = 'succeeded', result_payload = %s::jsonb, result_summary = %s::jsonb,
                    error_message = NULL, finished_at = now()
                WHERE id = %s
                """,
                (json.dumps(result), json.dumps(summary), job_id),
            )
        conn.commit()


def mark_job_failed(job_id: str, message: str) -> None:
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_backtest_jobs
                SET status = 'failed', error_message = %s, finished_at = now()
                WHERE id = %s
                """,
                (message[:8000], job_id),
            )
        conn.commit()


def get_result_payload(job_id: str) -> Optional[dict[str, Any]]:
    import psycopg

    try:
        UUID(job_id)
    except ValueError:
        return None
    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status, result_payload FROM api_backtest_jobs WHERE id = %s",
                (job_id,),
            )
            row = cur.fetchone()
    if not row or row["status"] != "succeeded":
        return None
    rp = row.get("result_payload")
    if isinstance(rp, str):
        return json.loads(rp)
    return dict(rp) if rp else None


def reconcile_stale_running_jobs() -> None:
    import psycopg

    with psycopg.connect(resolve_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_backtest_jobs
                SET status = 'failed',
                    error_message = 'Server restarted while job was running.',
                    finished_at = now()
                WHERE status = 'running'
                """
            )
        conn.commit()
