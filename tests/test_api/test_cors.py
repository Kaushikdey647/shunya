"""CORS allowlist via SHUNYA_CORS_ORIGINS."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient


async def _worker_no_db(stop: asyncio.Event) -> None:
    await stop.wait()


def test_no_cors_headers_when_origins_unset(monkeypatch) -> None:
    monkeypatch.delenv("SHUNYA_CORS_ORIGINS", raising=False)
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    from api.main import create_app

    with TestClient(create_app()) as client:
        r = client.get("/healthz", headers={"Origin": "https://app.example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_reflects_configured_origin(monkeypatch) -> None:
    monkeypatch.setenv("SHUNYA_CORS_ORIGINS", "https://shunya-ui.vercel.app, https://preview.vercel.app")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    from api.main import create_app

    with TestClient(create_app()) as client:
        r = client.get("/healthz", headers={"Origin": "https://shunya-ui.vercel.app"})
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://shunya-ui.vercel.app"


def test_cors_preflight_options(monkeypatch) -> None:
    monkeypatch.setenv("SHUNYA_CORS_ORIGINS", "https://app.example.com")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    from api.main import create_app

    with TestClient(create_app()) as client:
        r = client.options(
            "/healthz",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"
