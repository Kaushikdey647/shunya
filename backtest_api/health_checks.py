from __future__ import annotations

import logging
import time
from datetime import timezone
from typing import Literal

import pandas as pd
import psycopg

from backtest_api.schemas.models import HealthComponentModel, HealthResponseModel
from backtest_api.settings import get_settings
from shunya.data.providers import YFinanceMarketDataProvider
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_index_policy
from shunya.data.yfinance_session import build_yfinance_session

_log = logging.getLogger(__name__)

HealthStatus = Literal["ok", "error"]


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def check_backend() -> HealthComponentModel:
    t0 = time.perf_counter()
    try:
        get_settings()
        return HealthComponentModel(status="ok", latency_ms=_elapsed_ms(t0))
    except Exception as exc:  # noqa: BLE001
        _log.warning("backend health check failed: %s", exc)
        return HealthComponentModel(status="error", latency_ms=_elapsed_ms(t0))


def check_database() -> HealthComponentModel:
    t0 = time.perf_counter()
    try:
        from backtest_api.db import resolve_database_url

        with psycopg.connect(resolve_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return HealthComponentModel(status="ok", latency_ms=_elapsed_ms(t0))
    except Exception as exc:  # noqa: BLE001
        _log.warning("database health check failed: %s", exc)
        return HealthComponentModel(status="error", latency_ms=_elapsed_ms(t0))


def check_yfinance() -> HealthComponentModel:
    """Lightweight OHLCV fetch using the same stack as ``/instruments/.../ohlcv``."""
    t0 = time.perf_counter()
    try:
        end = pd.Timestamp.now(tz=timezone.utc)
        start = end - pd.Timedelta(days=7)
        session = build_yfinance_session()
        policy = default_bar_index_policy()
        spec = BarSpec(BarUnit.DAYS, 1)
        prov = YFinanceMarketDataProvider(session=session)
        df = prov.download(["SPY"], start, end, bar_spec=spec, bar_index_policy=policy)
        if df is None or df.empty:
            return HealthComponentModel(status="error", latency_ms=_elapsed_ms(t0))
        return HealthComponentModel(status="ok", latency_ms=_elapsed_ms(t0))
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance health check failed: %s", exc)
        return HealthComponentModel(status="error", latency_ms=_elapsed_ms(t0))


def collect_health() -> HealthResponseModel:
    backend = check_backend()
    database = check_database()
    yfinance = check_yfinance()

    if backend.status == "error" or database.status == "error":
        overall: Literal["ok", "degraded", "error"] = "error"
    elif yfinance.status == "error":
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponseModel(
        status=overall,
        backend=backend,
        database=database,
        yfinance=yfinance,
    )
