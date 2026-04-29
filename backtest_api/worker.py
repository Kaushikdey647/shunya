from __future__ import annotations

import asyncio
import logging

from backtest_api.repositories import backtests as jobs_repo
from backtest_api.settings import get_settings
from backtest_api.worker_job import execute_claimed_backtest_job

_log = logging.getLogger(__name__)


async def backtest_worker_loop(stop: asyncio.Event) -> None:
    settings = get_settings()
    interval = max(0.2, float(settings.worker_poll_interval_seconds))
    while not stop.is_set():
        row = await asyncio.to_thread(jobs_repo.claim_next_queued_job)
        if row is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
            continue
        job_id, payload = row
        err, serialized, summary = await asyncio.to_thread(
            execute_claimed_backtest_job, job_id, payload
        )
        if err is not None:
            _log.error("backtest job %s failed", job_id)
            await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, err)
        else:
            assert serialized is not None and summary is not None
            await asyncio.to_thread(jobs_repo.mark_job_succeeded, job_id, serialized, summary)
