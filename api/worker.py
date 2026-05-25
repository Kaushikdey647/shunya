from __future__ import annotations

import asyncio
import logging

from api.repositories import backtests as jobs_repo
from api.services.notification_hub import publish_notification
from api.tunable_config import get_effective_tunables
from api.worker_job import execute_claimed_backtest_job

_log = logging.getLogger(__name__)


async def backtest_worker_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        interval = max(0.2, float(get_effective_tunables().worker_poll_interval_seconds))
        row = await asyncio.to_thread(jobs_repo.claim_next_queued_job)
        if row is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
            continue
        job_id, payload = row
        alpha_id = str(payload.get("alpha_id", "") or "")
        await publish_notification(
            level="info",
            title="Backtest started",
            message=f"Worker started job {job_id} for alpha {alpha_id or '(unknown)'}."[:500],
            code="backtest.started",
            context={"job_id": job_id, "alpha_id": alpha_id[:64]},
        )
        err, err_code, serialized, summary = await asyncio.to_thread(
            execute_claimed_backtest_job, job_id, payload
        )
        if err is not None:
            _log.error("backtest job %s failed", job_id)
            await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, err, err_code)
            preview = (err or "").strip().split("\n", 1)[0][:400]
            await publish_notification(
                level="error",
                message=preview or "Backtest job failed.",
                code=err_code,
                context={"job_id": job_id},
            )
        else:
            assert serialized is not None and summary is not None
            await asyncio.to_thread(jobs_repo.mark_job_succeeded, job_id, serialized, summary)
            await publish_notification(
                level="info",
                title="Backtest finished",
                message=f"Backtest job {job_id} completed successfully.",
                code="backtest.succeeded",
                context={"job_id": job_id, "alpha_id": alpha_id[:64]},
            )
