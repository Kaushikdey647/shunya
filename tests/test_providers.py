from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import pytest

from shunya.data.providers import (
    AlpacaHistoricalMarketDataProvider,
    YFinanceMarketDataProvider,
)
from shunya.data.timeframes import BarIndexPolicy, BarSpec, BarUnit


@dataclass
class _Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: pd.Timestamp


class _BarSet:
    def __init__(self, data):
        self.data = data


class _ClientWithData:
    def __init__(self, data):
        self._data = data

    def get_stock_bars(self, _req):
        return _BarSet(self._data)


def test_alpaca_provider_multi_ticker_shape_and_normalized_index(monkeypatch):
    class _FakeClientFactory:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._client = _ClientWithData(
                {
                    "AAPL": [
                        _Bar(
                            100.0,
                            101.0,
                            99.0,
                            100.5,
                            1_000_000,
                            pd.Timestamp(datetime(2024, 1, 2, 21, 0, tzinfo=timezone.utc)),
                        )
                    ],
                    "MSFT": [
                        _Bar(
                            200.0,
                            202.0,
                            199.0,
                            201.0,
                            2_000_000,
                            pd.Timestamp(datetime(2024, 1, 2, 21, 0, tzinfo=timezone.utc)),
                        )
                    ],
                }
            )

        def get_stock_bars(self, req):
            return self._client.get_stock_bars(req)

    monkeypatch.setattr(
        "shunya.data.providers.build_stock_historical_data_client",
        lambda _settings: _FakeClientFactory(),
    )
    p = AlpacaHistoricalMarketDataProvider(api_key="k", secret_key="s")
    out = p.download(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")

    assert isinstance(out.columns, pd.MultiIndex)
    assert ("AAPL", "Open") in out.columns
    assert ("MSFT", "Close") in out.columns
    assert out.index.name == "Date"
    assert out.index.tz is None
    assert out.index[0] == pd.Timestamp("2024-01-02")


def test_alpaca_provider_single_ticker_shape(monkeypatch):
    class _FakeClientFactory:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._client = _ClientWithData(
                {
                    "AAPL": [
                        _Bar(
                            100.0,
                            101.0,
                            99.0,
                            100.5,
                            1_000_000,
                            pd.Timestamp("2024-01-02 21:00:00+00:00"),
                        )
                    ]
                }
            )

        def get_stock_bars(self, req):
            return self._client.get_stock_bars(req)

    monkeypatch.setattr(
        "shunya.data.providers.build_stock_historical_data_client",
        lambda _settings: _FakeClientFactory(),
    )
    p = AlpacaHistoricalMarketDataProvider(api_key="k", secret_key="s")
    out = p.download(["AAPL"], "2024-01-01", "2024-01-10")
    assert not isinstance(out.columns, pd.MultiIndex)
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(set(out.columns))
    assert out.index[0] == pd.Timestamp("2024-01-02")


def test_alpaca_provider_raises_when_symbols_missing(monkeypatch):
    class _FakeClientFactory:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._client = _ClientWithData(
                {
                    "AAPL": [
                        _Bar(
                            100.0,
                            101.0,
                            99.0,
                            100.5,
                            1_000_000,
                            pd.Timestamp("2024-01-02 21:00:00+00:00"),
                        )
                    ]
                }
            )

        def get_stock_bars(self, req):
            return self._client.get_stock_bars(req)

    monkeypatch.setattr(
        "shunya.data.providers.build_stock_historical_data_client",
        lambda _settings: _FakeClientFactory(),
    )
    p = AlpacaHistoricalMarketDataProvider(api_key="k", secret_key="s")
    with pytest.raises(ValueError, match="MSFT"):
        p.download(["AAPL", "MSFT"], "2024-01-01", "2024-01-10")


def test_alpaca_provider_raises_on_empty_response(monkeypatch):
    class _FakeClientFactory:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._client = _ClientWithData({})

        def get_stock_bars(self, req):
            return self._client.get_stock_bars(req)

    monkeypatch.setattr(
        "shunya.data.providers.build_stock_historical_data_client",
        lambda _settings: _FakeClientFactory(),
    )
    p = AlpacaHistoricalMarketDataProvider(api_key="k", secret_key="s")
    with pytest.raises(ValueError, match="AAPL"):
        p.download(["AAPL"], "2024-01-01", "2024-01-10")


def test_yfinance_provider_normalizes_index(monkeypatch):
    def _fake_download(*args, **kwargs):
        assert kwargs.get("repair") is True
        del args
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02 16:00:00-05:00"), pd.Timestamp("2024-01-03 16:00:00-05:00")]
        )
        return pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [11.0, 12.0],
                "Low": [9.0, 10.0],
                "Close": [10.5, 11.5],
                "Volume": [100.0, 101.0],
            },
            index=idx,
        )

    monkeypatch.setattr("shunya.data.providers.yf.download", _fake_download)
    p = YFinanceMarketDataProvider()
    out = p.download(["AAPL"], "2024-01-01", "2024-01-10")
    assert out.index.name == "Date"
    assert out.index.tz is None
    assert out.index[0] == pd.Timestamp("2024-01-02")


def test_yfinance_provider_intraday_preserves_time_and_sets_interval(monkeypatch):
    seen: dict = {}

    def _fake_download_tickers(tickers, *args, **kwargs):
        assert kwargs.get("repair") is True
        seen["interval"] = kwargs.get("interval")
        del tickers, args
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02 14:30:00+00:00")])
        return pd.DataFrame(
            {
                "Open": [10.0],
                "High": [11.0],
                "Low": [9.0],
                "Close": [10.5],
                "Volume": [100.0],
            },
            index=idx,
        )

    monkeypatch.setattr("shunya.data.providers.yf.download", _fake_download_tickers)
    p = YFinanceMarketDataProvider()
    out = p.download(
        ["AAPL"], "2024-01-01", "2024-01-10", bar_spec=BarSpec(BarUnit.MINUTES, 5)
    )
    assert seen["interval"] == "5m"
    assert out.index[0] == pd.Timestamp("2024-01-02 09:30:00")

    out_utc = p.download(
        ["AAPL"],
        "2024-01-01",
        "2024-01-10",
        bar_spec=BarSpec(BarUnit.MINUTES, 5),
        bar_index_policy=BarIndexPolicy(timezone="UTC"),
    )
    assert out_utc.index[0] == pd.Timestamp("2024-01-02 14:30:00")
