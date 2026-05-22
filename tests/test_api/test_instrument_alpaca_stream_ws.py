"""Instrument Alpaca live WebSocket: availability and validation (no real Alpaca)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from shunya.integration.alpaca_settings import AlpacaRuntimeSettings


def test_alpaca_bars_ws_disabled_returns_error_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_stream as sm

    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=False))

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/AAPL/stream/alpaca-bars") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "error"
        assert raw["code"] == "alpaca_disabled"


def test_alpaca_bars_ws_invalid_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_stream as sm

    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=True))
    monkeypatch.setattr(
        sm,
        "try_load_alpaca_settings_from_env",
        lambda: AlpacaRuntimeSettings(api_key_id="dummy", secret_key="dummy", paper=True),
    )

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/@@@/stream/alpaca-bars") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "error"
        assert raw["code"] == "invalid_symbol"


def test_alpaca_bars_ws_deprecated_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_stream as sm

    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=True))
    monkeypatch.setattr(
        sm,
        "try_load_alpaca_settings_from_env",
        lambda: AlpacaRuntimeSettings(api_key_id="dummy", secret_key="dummy", paper=True),
    )

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/AAPL/stream/alpaca-bars") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "error"
        assert raw["code"] == "deprecated_stream"
        assert "alpaca-l1" in raw.get("replacement_path", "")
