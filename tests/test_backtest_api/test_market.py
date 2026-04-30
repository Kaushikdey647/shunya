"""Market dashboard API and services (mocked yfinance)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backtest_api.main import create_app
from backtest_api.schemas.models import (
    MarketHeadlineItem,
    MarketMoverRow,
    MarketSnapshotRow,
    MarketSnapshotResponse,
)
from backtest_api.services.market_exceptions import MarketProviderError
from backtest_api.services.market_headlines import fetch_market_headlines
from backtest_api.services.market_movers import fetch_movers
from backtest_api.services.market_snapshot import build_snapshot
from backtest_api.services.market_symbols import normalize_market_symbol


def test_normalize_market_symbol() -> None:
    assert normalize_market_symbol(" spy ") == "SPY"
    assert normalize_market_symbol("^VIX") == "^VIX"
    with pytest.raises(ValueError):
        normalize_market_symbol("bad sym!")


def test_build_snapshot_single_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")]
    )
    df = pd.DataFrame(
        {
            "Open": [10.0, 10.5],
            "High": [10.5, 11.0],
            "Low": [9.8, 10.2],
            "Close": [10.2, 10.5],
            "Volume": [1e6, 1.1e6],
        },
        index=idx,
    )

    def _dl(**_kwargs: object) -> pd.DataFrame:
        return df

    monkeypatch.setattr("backtest_api.services.market_snapshot.yf.download", _dl)
    rows = build_snapshot(["SPY"])
    assert len(rows) == 1
    assert rows[0].symbol == "SPY"
    assert rows[0].last == pytest.approx(10.5)
    assert rows[0].pct_change_1d is not None
    assert pytest.approx(rows[0].pct_change_1d, rel=1e-6) == (10.5 - 10.2) / 10.2 * 100.0


def test_build_snapshot_raises_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "backtest_api.services.market_snapshot.yf.download",
        lambda **_kw: pd.DataFrame(),
    )
    with pytest.raises(MarketProviderError):
        build_snapshot(["SPY"])


def test_fetch_movers_maps_quotes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _screen(_key: str, *, count: int = 25, session=None):  # noqa: ANN001
        return {
            "quotes": [
                {
                    "symbol": "AAA",
                    "regularMarketPrice": 12.5,
                    "regularMarketChangePercent": 2.0,
                    "regularMarketVolume": 1_000_000.0,
                }
            ]
        }

    monkeypatch.setattr("backtest_api.services.market_movers.yf.screen", _screen)
    rows = fetch_movers("gainers", 10)
    assert len(rows) == 1
    assert rows[0] == MarketMoverRow(
        ticker="AAA",
        price=12.5,
        pct_change=2.0,
        volume=1_000_000.0,
    )


def test_fetch_headlines(monkeypatch: pytest.MonkeyPatch) -> None:
    class _News:
        news = [{"title": "Hi", "publisher": "Reuters", "link": "https://x.test/a"}]

    def _search(*_a, **_kw):  # noqa: ANN001
        return _News()

    monkeypatch.setattr("backtest_api.services.market_headlines.yf.Search", _search)
    items = fetch_market_headlines(5)
    assert len(items) == 1
    assert items[0].title == "Hi"


def test_post_snapshot_router(monkeypatch: pytest.MonkeyPatch) -> None:
    def _snap(symbols: list[str]) -> list[MarketSnapshotRow]:
        assert symbols == ["SPY"]
        return [
            MarketSnapshotRow(
                symbol="SPY",
                last=100.0,
                pct_change_1d=1.0,
                volume=1e6,
                sparkline_close=[99.0, 100.0],
            )
        ]

    monkeypatch.setattr("backtest_api.routers.market.build_snapshot", _snap)
    client = TestClient(create_app())
    r = client.post("/market/snapshot", json={"symbols": ["spy"]})
    assert r.status_code == 200
    body = r.json()
    assert body["rows"][0]["symbol"] == "SPY"
    assert body["rows"][0]["last"] == 100.0


def test_post_snapshot_invalid_symbol() -> None:
    client = TestClient(create_app())
    r = client.post("/market/snapshot", json={"symbols": ["bad symbol"]})
    assert r.status_code == 400


def test_get_movers_router(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mov(kind: str, limit: int) -> list[MarketMoverRow]:  # noqa: ARG001
        return [MarketMoverRow(ticker="ZZ", price=1.0)]

    monkeypatch.setattr("backtest_api.routers.market.fetch_movers", _mov)
    client = TestClient(create_app())
    r = client.get("/market/movers?kind=losers&limit=5")
    assert r.status_code == 200
    assert r.json()["kind"] == "losers"


def test_get_headlines_router(monkeypatch: pytest.MonkeyPatch) -> None:
    def _head(limit: int) -> list[MarketHeadlineItem]:
        assert limit == 10
        return [MarketHeadlineItem(title="W")]

    monkeypatch.setattr("backtest_api.routers.market.fetch_market_headlines", _head)
    client = TestClient(create_app())
    r = client.get("/market/headlines?limit=10")
    assert r.status_code == 200
    assert r.json()["headlines"][0]["title"] == "W"


def test_snapshot_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_symbols: list[str]) -> None:
        raise MarketProviderError("x")

    monkeypatch.setattr("backtest_api.routers.market.build_snapshot", _boom)
    client = TestClient(create_app())
    r = client.post("/market/snapshot", json={"symbols": ["SPY"]})
    assert r.status_code == 502
