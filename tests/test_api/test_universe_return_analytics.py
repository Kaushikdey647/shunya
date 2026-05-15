"""Universe return-analytics API."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


def test_return_analytics_invalid_universe_id() -> None:
    client = TestClient(create_app())
    r = client.get(
        "/universes/not-a-uuid/return-analytics",
        params={"period": "1y", "interval": "1d", "source": "yfinance"},
    )
    assert r.status_code == 400
    body = r.json()
    assert body.get("detail", {}).get("code") == "VALIDATION_ERROR"


def test_return_analytics_invalid_period() -> None:
    client = TestClient(create_app())
    uid = str(uuid.uuid4())
    r = client.get(
        f"/universes/{uid}/return-analytics",
        params={"period": "bogus", "interval": "1d", "source": "yfinance"},
    )
    assert r.status_code == 400
    assert r.json().get("detail", {}).get("code") == "VALIDATION_ERROR"


@pytest.mark.timescale
def test_return_analytics_not_found(api_database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", api_database_url)
    from shunya.data.timescale.dbutil import apply_migrations

    apply_migrations()
    client = TestClient(create_app())
    uid = "00000000-0000-4000-8000-00000000dead"
    r = client.get(
        f"/universes/{uid}/return-analytics",
        params={"period": "1y", "interval": "1d", "source": "yfinance"},
    )
    assert r.status_code == 404
    assert r.json().get("detail", {}).get("code") == "UNIVERSE_NOT_FOUND"
