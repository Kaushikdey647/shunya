"""Instruments API: mocked yfinance / OHLCV resolver."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backtest_api.main import create_app
from backtest_api.schemas.models import (
    InstrumentOhlcvResponse,
    InstrumentSearchNewsItem,
    InstrumentSearchQuote,
    InstrumentSearchResponse,
    OhlcvBar,
)
from backtest_api.services.instrument_ohlcv import InstrumentOhlcvResult


def test_search_requires_q() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/search")
    assert r.status_code == 422


def test_search_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(q: str) -> InstrumentSearchResponse:
        return InstrumentSearchResponse(
            quotes=[
                InstrumentSearchQuote(symbol="TEST", shortname="Test Co", exchange="NMS"),
            ],
            news=[InstrumentSearchNewsItem(title="Hello", link="https://example.com/n", publisher="X")],
            nav_links=[],
        )

    monkeypatch.setattr("backtest_api.routers.instruments._run_search", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/search?q=te")
    assert r.status_code == 200
    data = r.json()
    assert data["quotes"][0]["symbol"] == "TEST"
    assert data["news"][0]["title"] == "Hello"


def test_ohlcv_invalid_symbol() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/bad%20sym/ohlcv")
    assert r.status_code == 400


def test_ohlcv_returns_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(sym: str, interval: str, period: str, *, defer_storage: bool = False) -> InstrumentOhlcvResult:
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=sym,
                interval=interval,
                period=period,
                bars=[
                    OhlcvBar(
                        time="2024-01-02T00:00:00+00:00",
                        open=1.0,
                        high=2.0,
                        low=0.5,
                        close=1.5,
                        volume=100.0,
                    ),
                ],
                data_source="yfinance",
                storage_status="none",
            )
        )

    monkeypatch.setattr("backtest_api.routers.instruments.resolve_instrument_ohlcv_sync", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/ohlcv?interval=1d&period=1mo")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    assert len(data["bars"]) == 1
    assert data["bars"][0]["close"] == 1.5
    assert data["data_source"] == "yfinance"


def test_ingestion_run_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SHUNYA_DATABASE_URL", raising=False)
    client = TestClient(create_app())
    r = client.get("/instruments/ingestion-runs/1")
    assert r.status_code == 503
