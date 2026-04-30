"""Tests for Tiingo OHLCV parsing and provider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from shunya.data.providers import (
    TiingoMarketDataProvider,
    tiingo_daily_json_to_ohlcv,
    tiingo_resolve_api_key,
    ticker_to_tiingo_symbol,
)
from shunya.data.timeframes import BarSpec, BarUnit


def _bar(
    date: str,
    o: float = 100.0,
    h: float = 101.0,
    l: float = 99.0,
    c: float = 100.5,
    v: float = 1_000_000.0,
) -> dict:
    return {"date": date, "open": o, "high": h, "low": l, "close": c, "volume": v}


def test_ticker_to_tiingo_symbol() -> None:
    assert ticker_to_tiingo_symbol("brk.b") == "BRK-B"
    assert ticker_to_tiingo_symbol("MSFT") == "MSFT"


def test_tiingo_daily_json_to_ohlcv_parses_rows() -> None:
    df = tiingo_daily_json_to_ohlcv(
        [
            _bar("2024-01-03T00:00:00Z"),
            _bar("2024-01-04T00:00:00Z", c=101.0, v=1_100_000.0),
        ],
        symbol="IBM",
    )
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2
    assert float(df.iloc[1]["Close"]) == 101.0
    assert float(df.iloc[1]["Volume"]) == 1_100_000.0


def test_tiingo_daily_json_detail_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        tiingo_daily_json_to_ohlcv({"detail": "Ticker not found."}, symbol="BAD")


def test_tiingo_resolve_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="token"):
        tiingo_resolve_api_key()
    monkeypatch.setenv("TIINGO_API_KEY", "tok-a")
    assert tiingo_resolve_api_key() == "tok-a"
    monkeypatch.setenv("SHUNYA_TIINGO_API_KEY", "tok-b")
    assert tiingo_resolve_api_key() == "tok-b"
    assert tiingo_resolve_api_key("override") == "override"


def test_tiingo_provider_download_filters_exclusive_end() -> None:
    client = MagicMock()
    client.get_ticker_price.return_value = [
        _bar("2024-01-03T00:00:00Z"),
        _bar("2024-01-04T00:00:00Z"),
        _bar("2024-01-05T00:00:00Z"),
    ]
    prov = TiingoMarketDataProvider(client=client, inter_request_delay_seconds=0.0)
    out = prov.download(["IBM"], "2024-01-03", "2024-01-05")
    assert not isinstance(out.columns, pd.MultiIndex)
    assert len(out) == 2
    client.get_ticker_price.assert_called_once()
    assert client.get_ticker_price.call_args[0][0] == "IBM"
    kwargs = client.get_ticker_price.call_args[1]
    assert kwargs["startDate"] == "2024-01-03"
    assert kwargs["endDate"] == "2024-01-04"


def test_tiingo_provider_maps_symbology() -> None:
    client = MagicMock()
    client.get_ticker_price.return_value = [_bar("2024-01-03T00:00:00Z")]
    prov = TiingoMarketDataProvider(client=client, inter_request_delay_seconds=0.0)
    prov.download(["brk.b"], "2024-01-03", "2024-01-10")
    assert client.get_ticker_price.call_args[0][0] == "BRK-B"


def test_tiingo_provider_multi_ticker_multiindex() -> None:
    client = MagicMock()

    def _price(ticker: str, **_kwargs: object) -> list[dict]:
        if ticker == "IBM":
            return [_bar("2024-01-03T00:00:00Z", c=100.0)]
        if ticker == "MSFT":
            return [_bar("2024-01-03T00:00:00Z", c=200.0)]
        raise AssertionError(ticker)

    client.get_ticker_price.side_effect = _price
    prov = TiingoMarketDataProvider(client=client, inter_request_delay_seconds=0.0)
    out = prov.download(["IBM", "MSFT"], "2024-01-03", "2024-01-10")
    assert isinstance(out.columns, pd.MultiIndex)
    assert out["IBM"]["Close"].iloc[0] == 100.0
    assert out["MSFT"]["Close"].iloc[0] == 200.0


def test_tiingo_provider_rejects_non_daily_bar_spec() -> None:
    prov = TiingoMarketDataProvider(client=MagicMock(), inter_request_delay_seconds=0.0)
    bad = BarSpec(BarUnit.HOURS, 1)
    with pytest.raises(ValueError, match="daily"):
        prov.download(["IBM"], "2024-01-01", "2024-01-10", bar_spec=bad)
