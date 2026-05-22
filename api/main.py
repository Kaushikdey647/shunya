"""FastAPI application factory and ASGI entrypoint for the Shunya HTTP API.

``create_app()`` wires routers, CORS, health routes, and a lifespan that starts the
in-process async job worker loop (backtest queue). The module-level ``app`` is the default ASGI app for
``uvicorn api.main:app``. See ``api/README.md`` and the published docs under **Reference**
for route semantics and environment variables.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.exception_handlers import register_exception_handlers
from api.health_checks import collect_health
from api.repositories import backtests as jobs_repo
from api.routers import (
    alphas,
    app_settings,
    backtests,
    data,
    indices,
    instrument_l1_stream,
    instrument_stream,
    instruments,
    market,
    trade_desk,
    universes,
)
from api.schemas.models import HealthResponseModel

_log = logging.getLogger(__name__)


def _parse_cors_allow_origins() -> list[str]:
    """Comma-separated browser origins for CORS (e.g. ``https://app.vercel.app``)."""
    raw = os.environ.get("SHUNYA_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop = asyncio.Event()
    try:
        jobs_repo.reconcile_stale_running_jobs()
    except Exception as exc:  # noqa: BLE001
        _log.warning("reconcile stale jobs skipped: %s", exc)

    from api.settings import get_settings
    from api.trade_desk_runtime import build_trade_desk_runtime
    from shunya.integration.alpaca_settings import try_load_alpaca_settings_from_env

    api_settings = get_settings()
    if api_settings.alpaca_enabled:
        alp = try_load_alpaca_settings_from_env()
        if alp is None:
            raise RuntimeError(
                "SHUNYA_API_ALPACA_ENABLED is set but Alpaca keys are missing. "
                "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (or SHUNYA_ALPACA_* aliases)."
            )
        app.state.trade_desk_runtime = build_trade_desk_runtime(alp)
        _log.info("Trade desk runtime initialized (paper=%s)", alp.paper)
    else:
        app.state.trade_desk_runtime = None

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
    app = FastAPI(title="Shunya HTTP API", version="0.1.0", lifespan=lifespan)

    cors_origins = _parse_cors_allow_origins()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(alphas.router)
    app.include_router(app_settings.router)
    app.include_router(indices.router)
    app.include_router(universes.router)
    app.include_router(backtests.router)
    app.include_router(data.router)
    app.include_router(market.router)
    app.include_router(instruments.router)
    app.include_router(instrument_stream.router)
    app.include_router(instrument_l1_stream.router)
    app.include_router(trade_desk.router)

    register_exception_handlers(app)

    @app.get("/health", response_model=HealthResponseModel)
    def health() -> HealthResponseModel:
        return collect_health()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Fast liveness probe (no DB / Yahoo). Use for Railway and load balancers."""

        return {"status": "ok"}

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
