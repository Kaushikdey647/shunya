"""Tests for KiteHistoricalMarketDataProvider with a mocked KiteConnect client."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from shunya.data.kite_provider import KiteHistoricalMarketDataProvider, _bar_spec_to_kite_interval
from shunya.data.timeframes import BarSpec, BarUnit


def _mock_kite_with_candles(candles_by_token):
    kite = MagicMock()

    def historical_data(token, from_date, to_date, interval, **kw):
        return candles_by_token.get(token, [])

    kite.historical_data.side_effect = historical_data
    kite.instruments.return_value = [
        {"tradingsymbol": "INFY", "instrument_token": 408065},
        {"tradingsymbol": "TCS", "instrument_token": 2953217},
    ]
    return kite


def _sample_candles(n=3, base_date="2024-01-02"):
    candles = []
    for i in range(n):
        candles.append({
            "date": datetime(2024, 1, 2 + i, 0, 0, 0),
            "open": 1500.0 + i,
            "high": 1510.0 + i,
            "low": 1490.0 + i,
            "close": 1505.0 + i,
            "volume": 100000 + i * 1000,
        })
    return candles


class TestBarSpecToKiteInterval:
    def test_minute_intervals(self):
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.MINUTES, 1)) == "minute"
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.MINUTES, 5)) == "5minute"
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.MINUTES, 15)) == "15minute"
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.MINUTES, 60)) == "60minute"

    def test_invalid_minute_raises(self):
        with pytest.raises(ValueError, match="Kite minute interval"):
            _bar_spec_to_kite_interval(BarSpec(BarUnit.MINUTES, 7))

    def test_hour_maps_to_60minute(self):
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.HOURS, 1)) == "60minute"

    def test_day_week_month(self):
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.DAYS, 1)) == "day"
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.WEEKS, 1)) == "week"
        assert _bar_spec_to_kite_interval(BarSpec(BarUnit.MONTHS, 1)) == "month"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            _bar_spec_to_kite_interval(BarSpec(BarUnit.YEARS, 1))
        with pytest.raises(ValueError):
            _bar_spec_to_kite_interval(BarSpec(BarUnit.SECONDS, 1))


class TestKiteHistoricalMarketDataProvider:
    def test_single_ticker_download(self):
        candles = _sample_candles(3)
        kite = _mock_kite_with_candles({408065: candles})
        provider = KiteHistoricalMarketDataProvider(
            kite_client=kite,
            instrument_map={"INFY": 408065},
        )
        df = provider.download(
            ["INFY"],
            start="2024-01-02",
            end="2024-01-05",
        )
        assert not df.empty
        assert "Open" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) == 3

    def test_multi_ticker_download(self):
        infy_candles = _sample_candles(3)
        tcs_candles = _sample_candles(3)
        kite = _mock_kite_with_candles({408065: infy_candles, 2953217: tcs_candles})
        provider = KiteHistoricalMarketDataProvider(
            kite_client=kite,
            instrument_map={"INFY": 408065, "TCS": 2953217},
        )
        df = provider.download(
            ["INFY", "TCS"],
            start="2024-01-02",
            end="2024-01-05",
        )
        assert isinstance(df.columns, pd.MultiIndex)
        assert "INFY" in df.columns.get_level_values(0)
        assert "TCS" in df.columns.get_level_values(0)

    def test_empty_ticker_list_returns_empty(self):
        kite = _mock_kite_with_candles({})
        provider = KiteHistoricalMarketDataProvider(
            kite_client=kite,
            instrument_map={},
        )
        df = provider.download([], start="2024-01-02", end="2024-01-05")
        assert df.empty

    def test_no_candles_raises(self):
        kite = _mock_kite_with_candles({408065: []})
        provider = KiteHistoricalMarketDataProvider(
            kite_client=kite,
            instrument_map={"INFY": 408065},
        )
        with pytest.raises(ValueError, match="no candles"):
            provider.download(["INFY"], start="2024-01-02", end="2024-01-05")

    def test_auto_resolve_instruments(self):
        candles = _sample_candles(2)
        kite = _mock_kite_with_candles({408065: candles})
        provider = KiteHistoricalMarketDataProvider(kite_client=kite)
        df = provider.download(["INFY"], start="2024-01-02", end="2024-01-04")
        assert not df.empty
        kite.instruments.assert_called_once_with("NSE")

    def test_unknown_symbol_raises(self):
        kite = _mock_kite_with_candles({})
        kite.instruments.return_value = []
        provider = KiteHistoricalMarketDataProvider(kite_client=kite)
        with pytest.raises(ValueError, match="Cannot resolve"):
            provider.download(["UNKNOWN"], start="2024-01-02", end="2024-01-05")

    def test_index_named_date(self):
        candles = _sample_candles(3)
        kite = _mock_kite_with_candles({408065: candles})
        provider = KiteHistoricalMarketDataProvider(
            kite_client=kite,
            instrument_map={"INFY": 408065},
        )
        df = provider.download(["INFY"], start="2024-01-02", end="2024-01-05")
        assert df.index.name == "Date"
