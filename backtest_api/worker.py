from __future__ import annotations

import asyncio
import logging
import traceback

from backtest_api.repositories import alphas as alphas_repo
from backtest_api.repositories import backtests as jobs_repo
from backtest_api.runner import run_backtest_from_payload
from backtest_api.settings import get_settings

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
        try:
            alpha_id = str(payload["alpha_id"])
            ar = await asyncio.to_thread(alphas_repo.get_alpha_raw, alpha_id)
            if ar is None:
                await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, "Alpha not found.")
                continue
            serialized, summary = await asyncio.to_thread(
                run_backtest_from_payload,
                payload,
                str(ar["import_ref"]),
                dict(ar["finstrat_config"]),
            )
            await asyncio.to_thread(jobs_repo.mark_job_succeeded, job_id, serialized, summary)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            _log.exception("backtest job %s failed", job_id)
            msg = f"{exc!s}\n{tb}"[:8000]
            await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, msg)
