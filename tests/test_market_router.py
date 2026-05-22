"""Pure market route resolution and Alpaca feed wiring."""

from __future__ import annotations

import pandas as pd
import pytest
from alpaca.data.enums import DataFeed

from shunya.data.market_data.context import MarketDataRouteContext
from shunya.data.market_data.errors import (
    MARKET_ROUTE_NO_CREDENTIALS,
    MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN,
    MarketRouteError,
)
from shunya.data.market_data.resolve import resolve_market_route
from shunya.data.providers import AlpacaHistoricalMarketDataProvider
from shunya.data.timeframes import BarSpec, BarUnit


def _ctx_daily() -> MarketDataRouteContext:
    return MarketDataRouteContext(symbols=("SPY",), bar_spec=BarSpec(BarUnit.DAYS, 1), demo_relaxed=False)


def _ctx_1m() -> MarketDataRouteContext:
    return MarketDataRouteContext(symbols=("SPY",), bar_spec=BarSpec(BarUnit.MINUTES, 1), demo_relaxed=False)


def test_resolve_auto_daily_prefers_yfinance_upstream() -> None:
    d = resolve_market_route(_ctx_daily(), "auto")
    assert d.primary_upstream == "yfinance"
    assert d.cache_policy == "prefer_timescale"
    assert d.timescale_upstream_attempts == ("yfinance",)
    assert d.rule_id == "auto_daily_yfinance"


def test_resolve_explicit_yfinance_intraday_forbidden_without_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_MARKET_DATA_DEMO_RELAXED", raising=False)
    with pytest.raises(MarketRouteError) as ei:
        resolve_market_route(_ctx_1m(), "yfinance")
    assert ei.value.code == MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN
    assert ei.value.rule_id == "explicit_yfinance_intraday"


def test_resolve_explicit_yfinance_intraday_allowed_when_demo_relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_MARKET_DATA_DEMO_RELAXED", "1")
    d = resolve_market_route(_ctx_1m(), "yfinance")
    assert d.primary_upstream == "yfinance"


def test_resolve_auto_intraday_requires_alpaca_without_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_MARKET_DATA_DEMO_RELAXED", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(MarketRouteError) as ei:
        resolve_market_route(_ctx_1m(), "auto")
    assert ei.value.code == MARKET_ROUTE_NO_CREDENTIALS


def test_resolve_explicit_alpaca_requires_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(MarketRouteError) as ei:
        resolve_market_route(_ctx_1m(), "alpaca_sip")
    assert ei.value.code == MARKET_ROUTE_NO_CREDENTIALS


def test_alpaca_provider_passes_feed_matching_bar_feed_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    class _B:
        __slots__ = ("open", "high", "low", "close", "volume", "timestamp")

        def __init__(self) -> None:
            self.open = 100.0
            self.high = 101.0
            self.low = 99.0
            self.close = 100.5
            self.volume = 1_000_000.0
            self.timestamp = pd.Timestamp("2024-01-02 21:00:00+00:00")

    class _BarSet:
        def __init__(self, data):
            self.data = data

    class _Inner:
        def get_stock_bars(self, req: object) -> object:
            captured.append(req)
            return _BarSet({"AAPL": [_B()]})

    class _Factory:
        def get_stock_bars(self, req: object) -> object:
            return _Inner().get_stock_bars(req)

    monkeypatch.setattr(
        "shunya.data.providers.build_stock_historical_data_client",
        lambda _settings: _Factory(),
    )
    p = AlpacaHistoricalMarketDataProvider(api_key="k", secret_key="s", bar_feed_upstream="alpaca_iex")
    p.download(["AAPL"], "2024-01-01", "2024-01-10")
    assert len(captured) == 1
    assert captured[0].feed == DataFeed.IEX
