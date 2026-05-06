from __future__ import annotations

import asyncio
import importlib
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.health_checks import collect_health
from api.repositories import backtests as jobs_repo
from api.routers import alphas, backtests, data, indices, instruments, market
from api.schemas.models import HealthResponseModel

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = asyncio.Event()
    try:
        jobs_repo.reconcile_stale_running_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("reconcile stale jobs skipped: %s", exc)
    # Resolve at startup so tests can monkeypatch ``api.main.backtest_worker_loop``.
    _main = importlib.import_module("api.main")
    task = asyncio.create_task(_main.backtest_worker_loop(stop))
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
    app.include_router(indices.router)
    app.include_router(backtests.router)
    app.include_router(data.router)
    app.include_router(market.router)
    app.include_router(instruments.router)

    @app.get("/health", response_model=HealthResponseModel)
    def health() -> HealthResponseModel:
        return collect_health()

    return app


def backtest_worker_loop(stop: asyncio.Event):
    """Default async worker; tests may replace this name on ``api.main``."""
    from api.worker import backtest_worker_loop as _default_loop

    return _default_loop(stop)


app = create_app()


def run() -> None:
    import uvicorn

    host = os.environ.get("SHUNYA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("SHUNYA_API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
