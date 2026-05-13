"""GET/PATCH ``/settings/app`` and effective tunables merge."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from api.settings import get_settings
from api.tunable_config import clear_tunables_cache


@pytest.fixture(autouse=True)
def _clear_settings_and_tunables() -> None:
    get_settings.cache_clear()
    clear_tunables_cache()
    yield
    get_settings.cache_clear()
    clear_tunables_cache()


async def _worker_no_db(stop: asyncio.Event) -> None:
    await stop.wait()


def test_get_app_settings_returns_runtime_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setattr("api.repositories.runtime_config.fetch_runtime_payload", lambda: None)
    monkeypatch.setenv("SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS", "2.5")
    get_settings.cache_clear()
    clear_tunables_cache()

    with TestClient(create_app()) as client:
        r = client.get("/settings/app")
    assert r.status_code == 200
    body = r.json()
    assert body["runtime"]["worker_poll_interval_seconds"] == 2.5
    assert body["sources"]["worker_poll_interval_seconds"] == "environment"
    assert "database_configured" in body["environment"]


def test_get_effective_tunables_overlay_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.tunable_config import get_effective_tunables

    monkeypatch.setattr(
        "api.repositories.runtime_config.fetch_runtime_payload",
        lambda: {"worker_poll_interval_seconds": 42.0},
    )
    monkeypatch.setenv("SHUNYA_API_WORKER_POLL_INTERVAL_SECONDS", "1.0")
    get_settings.cache_clear()
    clear_tunables_cache()

    eff = get_effective_tunables(force_refresh=True)
    assert eff.worker_poll_interval_seconds == 42.0


def test_patch_app_settings_503_when_trade_desk_token_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.delenv("SHUNYA_API_TRADE_DESK_TOKEN", raising=False)
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        r = client.patch("/settings/app", json={"ollama_model": "x"})
    assert r.status_code == 503


def test_patch_app_settings_401_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "expected")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        r = client.patch(
            "/settings/app",
            headers={"X-Shunya-Trade-Desk-Token": "wrong"},
            json={"ollama_model": "m"},
        )
    assert r.status_code == 401


def test_patch_app_settings_503_when_database_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.delenv("SHUNYA_API_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SHUNYA_DATABASE_URL", raising=False)
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        r = client.patch(
            "/settings/app",
            headers={"X-Shunya-Trade-Desk-Token": "tok"},
            json={"ollama_model": "m"},
        )
    assert r.status_code == 503


def test_patch_app_settings_503_when_runtime_row_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_DATABASE_URL", "postgresql://unused:unused@127.0.0.1:9/unused")
    monkeypatch.setattr("api.repositories.runtime_config.fetch_runtime_payload", lambda: None)
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        r = client.patch(
            "/settings/app",
            headers={"X-Shunya-Trade-Desk-Token": "tok"},
            json={"ollama_model": "m"},
        )
    assert r.status_code == 503


def test_patch_app_settings_persists_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_DATABASE_URL", "postgresql://unused:unused@127.0.0.1:9/unused")

    store: dict[str, object] = {"worker_poll_interval_seconds": 3.0}

    def fetch() -> dict[str, object]:
        return dict(store)

    def save(payload: object) -> None:
        if isinstance(payload, dict):
            store.clear()
            store.update(payload)

    monkeypatch.setattr("api.repositories.runtime_config.fetch_runtime_payload", fetch)
    monkeypatch.setattr("api.repositories.runtime_config.save_runtime_payload", save)
    get_settings.cache_clear()
    clear_tunables_cache()

    with TestClient(create_app()) as client:
        r = client.patch(
            "/settings/app",
            headers={"X-Shunya-Trade-Desk-Token": "tok"},
            json={"ollama_model": "custom-model"},
        )
    assert r.status_code == 200
    assert r.json()["runtime"]["ollama_model"] == "custom-model"
    assert r.json()["sources"]["ollama_model"] == "database"
    assert store.get("ollama_model") == "custom-model"
    assert store.get("worker_poll_interval_seconds") == 3.0


@pytest.mark.timescale
def test_settings_patch_roundtrip_database(api_database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from shunya.data.timescale.dbutil import apply_migrations

    monkeypatch.setenv("DATABASE_URL", api_database_url)
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "integration-secret")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    get_settings.cache_clear()
    clear_tunables_cache()

    apply_migrations()

    from api.main import create_app

    with TestClient(create_app()) as client:
        g = client.get("/settings/app")
        assert g.status_code == 200
        before_model = g.json()["runtime"]["ollama_model"]

        p = client.patch(
            "/settings/app",
            headers={"X-Shunya-Trade-Desk-Token": "integration-secret"},
            json={"ollama_model": "integration-test-model"},
        )
        assert p.status_code == 200
        assert p.json()["runtime"]["ollama_model"] == "integration-test-model"
        assert p.json()["sources"]["ollama_model"] == "database"

        p2 = client.patch(
            "/settings/app",
            headers={"X-Shunya-Trade-Desk-Token": "integration-secret"},
            json={"ollama_model": before_model},
        )
        assert p2.status_code == 200
