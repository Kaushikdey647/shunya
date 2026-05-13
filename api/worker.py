from __future__ import annotations

import asyncio
import logging

from api.repositories import backtests as jobs_repo
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
        err, err_code, serialized, summary = await asyncio.to_thread(
            execute_claimed_backtest_job, job_id, payload
        )
        if err is not None:
            _log.error("backtest job %s failed", job_id)
            await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, err, err_code)
        else:
            assert serialized is not None and summary is not None
            await asyncio.to_thread(jobs_repo.mark_job_succeeded, job_id, serialized, summary)
