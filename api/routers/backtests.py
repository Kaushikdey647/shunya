from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query, status

from api.backtest_resolve import resolve_index_backtest_if_needed
from api.backtest_windows import normalize_backtest_create
from api.repositories import alphas as alphas_repo
from api.repositories import backtests as jobs_repo
from api.schemas.models import (
    BacktestCreate,
    BacktestJobOut,
    BacktestJobsDeleteBatchOut,
    BacktestJobsDeleteBatchRequest,
    BacktestLogLineOut,
)
from shunya.errors import ErrorCode, ShunyaError

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=BacktestJobOut, status_code=status.HTTP_201_CREATED)
def enqueue_backtest(body: BacktestCreate) -> BacktestJobOut:
    if alphas_repo.get_alpha_raw(body.alpha_id) is None:
        raise ShunyaError("Alpha not found.", code=ErrorCode.ALPHA_NOT_FOUND, http_status=404)
    body = normalize_backtest_create(body)
    resolved = resolve_index_backtest_if_needed(body)
    return jobs_repo.insert_job(resolved)


@router.get("", response_model=list[BacktestJobOut])
def list_backtests(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    alpha_id: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status"),
) -> list[BacktestJobOut]:
    if status_filter is not None and status_filter not in ("queued", "running", "succeeded", "failed"):
        raise ShunyaError(
            "Invalid status filter.",
            code=ErrorCode.INVALID_STATUS_FILTER,
            http_status=400,
        )
    return jobs_repo.list_jobs(limit=limit, offset=offset, alpha_id=alpha_id, status=status_filter)


@router.post("/delete-batch", response_model=BacktestJobsDeleteBatchOut)
def delete_backtests_batch(body: BacktestJobsDeleteBatchRequest) -> BacktestJobsDeleteBatchOut:
    try:
        deleted = jobs_repo.delete_jobs_by_ids(body.ids)
    except ValueError as exc:
        if str(exc) == "delete_batch_too_large":
            raise ShunyaError(
                "Too many job ids after deduplication (max 200 per request).",
                code=ErrorCode.DELETE_BATCH_TOO_LARGE,
                http_status=400,
            ) from exc
        raise
    return BacktestJobsDeleteBatchOut(deleted=deleted)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_backtest(job_id: str) -> None:
    ok = jobs_repo.delete_job(job_id)
    if not ok:
        raise ShunyaError("Job not found.", code=ErrorCode.BACKTEST_JOB_NOT_FOUND, http_status=404)


@router.get("/{job_id}", response_model=BacktestJobOut)
def get_backtest(job_id: str) -> BacktestJobOut:
    row = jobs_repo.get_job(job_id)
    if row is None:
        raise ShunyaError("Job not found.", code=ErrorCode.BACKTEST_JOB_NOT_FOUND, http_status=404)
    return row


@router.get("/{job_id}/logs", response_model=list[BacktestLogLineOut])
def get_backtest_logs(job_id: str) -> list[BacktestLogLineOut]:
    if jobs_repo.get_job(job_id) is None:
        raise ShunyaError("Job not found.", code=ErrorCode.BACKTEST_JOB_NOT_FOUND, http_status=404)
    rows = jobs_repo.get_job_logs(job_id)
    return [BacktestLogLineOut(**r) for r in rows]


@router.get("/{job_id}/result")
def get_backtest_result(job_id: str) -> dict[str, Any]:
    job = jobs_repo.get_job(job_id)
    if job is None:
        raise ShunyaError("Job not found.", code=ErrorCode.BACKTEST_JOB_NOT_FOUND, http_status=404)
    if job.status != "succeeded":
        raise ShunyaError(
            f"Job is {job.status}; result available only when succeeded.",
            code=ErrorCode.BACKTEST_RESULT_NOT_READY,
            http_status=409,
        )
    payload = jobs_repo.get_result_payload(job_id)
    if not payload:
        raise ShunyaError(
            "Result payload missing.",
            code=ErrorCode.BACKTEST_RESULT_MISSING,
            http_status=404,
        )
    return {"job_id": job_id, **payload}
