"""Health endpoint: aggregate status and per-component latencies."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from backtest_api.schemas.models import HealthComponentModel


async def _worker_no_db(stop: asyncio.Event) -> None:
    """Avoid lifespan worker touching Postgres when DATABASE_URL is unset."""
    await stop.wait()


def test_health_ok_all_components(monkeypatch) -> None:
    from backtest_api import health_checks
    from backtest_api.main import create_app

    monkeypatch.setattr("backtest_api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setattr(
        health_checks,
        "check_backend",
        lambda: HealthComponentModel(status="ok", latency_ms=0.01),
    )
    monkeypatch.setattr(
        health_checks,
        "check_database",
        lambda: HealthComponentModel(status="ok", latency_ms=2.0),
    )
    monkeypatch.setattr(
        health_checks,
        "check_yfinance",
        lambda: HealthComponentModel(status="ok", latency_ms=50.0),
    )

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"]["status"] == "ok"
    assert body["backend"]["latency_ms"] == 0.01
    assert body["database"]["status"] == "ok"
    assert body["yfinance"]["status"] == "ok"


def test_health_degraded_when_yfinance_fails(monkeypatch) -> None:
    from backtest_api import health_checks
    from backtest_api.main import create_app

    monkeypatch.setattr("backtest_api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setattr(
        health_checks,
        "check_backend",
        lambda: HealthComponentModel(status="ok", latency_ms=0.01),
    )
    monkeypatch.setattr(
        health_checks,
        "check_database",
        lambda: HealthComponentModel(status="ok", latency_ms=2.0),
    )
    monkeypatch.setattr(
        health_checks,
        "check_yfinance",
        lambda: HealthComponentModel(status="error", latency_ms=100.0),
    )

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_health_error_when_database_fails(monkeypatch) -> None:
    from backtest_api import health_checks
    from backtest_api.main import create_app

    monkeypatch.setattr("backtest_api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setattr(
        health_checks,
        "check_backend",
        lambda: HealthComponentModel(status="ok", latency_ms=0.01),
    )
    monkeypatch.setattr(
        health_checks,
        "check_database",
        lambda: HealthComponentModel(status="error", latency_ms=1.0),
    )
    monkeypatch.setattr(
        health_checks,
        "check_yfinance",
        lambda: HealthComponentModel(status="ok", latency_ms=50.0),
    )

    with TestClient(create_app()) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "error"
