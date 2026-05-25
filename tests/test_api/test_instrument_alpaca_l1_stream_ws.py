"""Instrument Alpaca L1 WebSocket: validation and hello (no real Alpaca connection)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from shunya.integration.alpaca_settings import AlpacaRuntimeSettings


@pytest.fixture(autouse=True)
def _ignore_us_rth_for_l1_ws_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hello-path tests run without a real DB; keep L1 WebSocket allowed outside US RTH."""
    monkeypatch.setenv("SHUNYA_ALPACA_L1_IGNORE_US_RTH", "1")


class _FakeAlpacaStream:
    def __init__(self) -> None:
        self.subscribed_quotes = False
        self.subscribed_trades = False

    async def _run_forever(self) -> None:  # noqa: SLF001
        await asyncio.sleep(3600)

    async def stop_ws(self) -> None:
        return

    def subscribe_quotes(self, _handler: Any, *_symbols: str) -> None:
        self.subscribed_quotes = True

    def subscribe_trades(self, _handler: Any, *_symbols: str) -> None:
        self.subscribed_trades = True

    def unsubscribe_quotes(self, *_symbols: str) -> None:
        return

    def unsubscribe_trades(self, *_symbols: str) -> None:
        return

    def register_trade_corrections(self, _handler: Any) -> None:
        return

    def register_trade_cancels(self, _handler: Any) -> None:
        return


def test_alpaca_l1_ws_us_rth_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_l1_stream as sm

    monkeypatch.delenv("SHUNYA_ALPACA_L1_IGNORE_US_RTH", raising=False)
    monkeypatch.setattr(sm, "alpaca_l1_us_equities_stream_allowed", lambda: False)
    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=True))
    monkeypatch.setattr(
        sm,
        "try_load_alpaca_settings_from_env",
        lambda: AlpacaRuntimeSettings(api_key_id="dummy", secret_key="dummy", paper=True),
    )

    published: list[dict[str, Any]] = []

    async def capture_publish(**kwargs: Any) -> None:
        published.append(kwargs)

    monkeypatch.setattr(sm, "publish_notification", capture_publish)

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/AAPL/stream/alpaca-l1") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "error"
        assert raw["code"] == "us_rth_closed"
    assert len(published) == 1
    assert published[0]["code"] == "us_rth_closed"
    assert published[0]["level"] == "warning"
    assert published[0]["context"]["symbol"] == "AAPL"


def test_alpaca_l1_ws_disabled_returns_error_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_l1_stream as sm

    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=False))

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/AAPL/stream/alpaca-l1") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "error"
        assert raw["code"] == "alpaca_disabled"


def test_alpaca_l1_ws_invalid_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_l1_stream as sm

    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=True))
    monkeypatch.setattr(
        sm,
        "try_load_alpaca_settings_from_env",
        lambda: AlpacaRuntimeSettings(api_key_id="dummy", secret_key="dummy", paper=True),
    )

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/@@@/stream/alpaca-l1") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "error"
        assert raw["code"] == "invalid_symbol"


def _clear_l1_hubs() -> None:
    import api.services.alpaca_l1_feed_hub as hubm

    with hubm._hubs_lock:
        hubm._hubs.clear()


def test_alpaca_l1_ws_hello_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.routers import instrument_l1_stream as sm

    import api.services.alpaca_l1_feed_hub as hubm

    _clear_l1_hubs()
    monkeypatch.setattr(sm, "get_settings", lambda: Settings(alpaca_enabled=True))
    monkeypatch.setattr(
        sm,
        "try_load_alpaca_settings_from_env",
        lambda: AlpacaRuntimeSettings(api_key_id="dummy", secret_key="dummy", paper=True),
    )
    fake = _FakeAlpacaStream()
    monkeypatch.setattr(hubm, "build_stock_data_stream", lambda *_a, **_k: fake)

    async def fake_to_thread(_fn, *_args, **_kwargs):
        return SimpleNamespace(instrument_kind="equity")

    monkeypatch.setattr(sm.asyncio, "to_thread", fake_to_thread)

    client = TestClient(create_app())
    with client.websocket_connect("/instruments/AAPL/stream/alpaca-l1") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "hello"
        assert raw["schema"] == 1
        assert raw["symbol"] == "AAPL"
        assert raw["feed"] == "iex"
        assert raw["channels"] == ["quotes", "trades"]
    assert fake.subscribed_quotes is True
    assert fake.subscribed_trades is True
    _clear_l1_hubs()
