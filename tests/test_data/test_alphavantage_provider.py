"""Tests for Alpha Vantage OHLCV parsing and provider."""

from __future__ import annotations

import pandas as pd
import pytest

from shunya.data.providers import (
    AlphaVantageMarketDataProvider,
    alphavantage_daily_payload_to_ohlcv,
    alphavantage_resolve_api_key,
)
from shunya.data.timeframes import BarSpec, BarUnit


def _sample_daily_payload() -> dict:
    return {
        "Meta Data": {"1. Information": "Daily Prices"},
        "Time Series (Daily)": {
            "2024-01-03": {
                "1. open": "100.0",
                "2. high": "101.5",
                "3. low": "99.0",
                "4. close": "100.5",
                "5. volume": "1000000",
            },
            "2024-01-04": {
                "1. open": "100.5",
                "2. high": "102.0",
                "3. low": "100.0",
                "4. close": "101.0",
                "5. volume": "1100000",
            },
        },
    }


def test_alphavantage_daily_payload_to_ohlcv_parses_rows() -> None:
    df = alphavantage_daily_payload_to_ohlcv(_sample_daily_payload(), symbol="IBM")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2024-01-03")
    assert float(df.iloc[0]["Close"]) == 100.5
    assert float(df.iloc[1]["Volume"]) == 1_100_000.0


def test_alphavantage_error_message_raises() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        alphavantage_daily_payload_to_ohlcv(
            {"Error Message": "Invalid API call. Please retry."},
            symbol="BAD",
        )


def test_alphavantage_note_raises_runtime() -> None:
    with pytest.raises(RuntimeError, match="frequency"):
        alphavantage_daily_payload_to_ohlcv(
            {"Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute."},
            symbol="IBM",
        )


def test_alphavantage_information_raises_runtime() -> None:
    with pytest.raises(RuntimeError, match="premium"):
        alphavantage_daily_payload_to_ohlcv(
            {"Information": "premium membership required"},
            symbol="IBM",
        )


def test_alphavantage_resolve_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        alphavantage_resolve_api_key()
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test-key")
    assert alphavantage_resolve_api_key() == "test-key"
    assert alphavantage_resolve_api_key("override") == "override"


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.urls: list[str] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> _FakeResp:
        self.urls.append(url)
        return _FakeResp(self._payload)


def test_alphavantage_provider_download_filters_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    sess = _FakeSession(_sample_daily_payload())
    prov = AlphaVantageMarketDataProvider(
        api_key="x",
        session=sess,
        outputsize="compact",
        inter_request_delay_seconds=0.0,
    )
    out = prov.download(["IBM"], "2024-01-04", "2024-01-10")
    assert not isinstance(out.columns, pd.MultiIndex)
    assert len(out) == 1
    assert out.index[0].strftime("%Y-%m-%d") == "2024-01-04"


def test_alphavantage_provider_rejects_non_daily_bar_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    prov = AlphaVantageMarketDataProvider(api_key="x", inter_request_delay_seconds=0.0)
    bad = BarSpec(BarUnit.HOURS, 1)
    with pytest.raises(ValueError, match="daily"):
        prov.download(["IBM"], "2024-01-01", "2024-01-10", bar_spec=bad)
