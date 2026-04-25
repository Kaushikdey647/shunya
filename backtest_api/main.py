from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backtest_api.repositories import backtests as jobs_repo
from backtest_api.routers import alphas, backtests, data
from backtest_api.worker import backtest_worker_loop

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = asyncio.Event()
    try:
        jobs_repo.reconcile_stale_running_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("reconcile stale jobs skipped: %s", exc)
    task = asyncio.create_task(backtest_worker_loop(stop))
    yield
    stop.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    app = FastAPI(title="Shunya backtest API", version="0.1.0", lifespan=lifespan)
    app.include_router(alphas.router)
    app.include_router(backtests.router)
    app.include_router(data.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


def run() -> None:
    import uvicorn

    host = os.environ.get("SHUNYA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("SHUNYA_API_PORT", "8000"))
    uvicorn.run("backtest_api.main:app", host=host, port=port, reload=False)
