from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status

from backtest_api.backtest_resolve import resolve_index_backtest_if_needed
from backtest_api.backtest_windows import normalize_backtest_create
from backtest_api.repositories import alphas as alphas_repo
from backtest_api.repositories import backtests as jobs_repo
from backtest_api.schemas.models import BacktestCreate, BacktestJobOut

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=BacktestJobOut, status_code=status.HTTP_201_CREATED)
def enqueue_backtest(body: BacktestCreate) -> BacktestJobOut:
    if alphas_repo.get_alpha_raw(body.alpha_id) is None:
        raise HTTPException(status_code=404, detail="Alpha not found.")
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
        raise HTTPException(status_code=400, detail="Invalid status filter.")
    return jobs_repo.list_jobs(limit=limit, offset=offset, alpha_id=alpha_id, status=status_filter)


@router.get("/{job_id}", response_model=BacktestJobOut)
def get_backtest(job_id: str) -> BacktestJobOut:
    row = jobs_repo.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return row


@router.get("/{job_id}/result")
def get_backtest_result(job_id: str) -> dict[str, Any]:
    job = jobs_repo.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job.status}; result available only when succeeded.",
        )
    payload = jobs_repo.get_result_payload(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Result payload missing.")
    return {"job_id": job_id, **payload}
