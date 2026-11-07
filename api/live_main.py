"""Hosted live-runner ASGI app (``uvicorn api.live_main:app``).

Only the live router and ``/healthz``: no backtest worker, no Timescale. The cron service
runs ``shunya-live tick`` separately against the same Postgres.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import live

_log = logging.getLogger(__name__)


def create_live_app(*, store: Optional[Any] = None, broker: Optional[Any] = None) -> FastAPI:
    """Build the hosted app; ``store`` / ``broker`` injection is for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if store is not None or broker is not None:
            app.state.live_store, app.state.live_broker = store, broker
        else:
            from shunya.integration.alpaca_settings import build_trading_client, load_alpaca_settings_from_env
            from shunya.live.broker import AlpacaBroker
            from shunya.live.control import read_control
            from shunya.live.store import PostgresLiveStore, apply_live_migrations

            if os.environ.get("RUN_MIGRATIONS", "1") == "1":
                apply_live_migrations()
            s = PostgresLiveStore()
            settings = load_alpaca_settings_from_env()
            if not settings.paper and os.environ.get("SHUNYA_LIVE_ALLOW_REAL_MONEY") != "1":
                raise RuntimeError("Refusing live (real-money) Alpaca keys; set SHUNYA_LIVE_ALLOW_REAL_MONEY=1.")
            app.state.live_store = s
            app.state.live_broker = AlpacaBroker(build_trading_client(settings), lambda: read_control(s))
            _log.info("live app ready (paper=%s)", settings.paper)
        yield

    app = FastAPI(title="Shunya live runner", version="0.1.0", lifespan=lifespan)
    origins = [o.strip() for o in os.environ.get("SHUNYA_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.include_router(live.router)

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness probe for Railway (no DB / broker calls)."""
        return {"ok": True}

    return app


app = create_live_app()
