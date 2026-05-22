"""Tests for Alpaca live bar JSON normalization."""

from __future__ import annotations

from datetime import datetime, timezone

from api.services.alpaca_live_bar_payload import alpaca_bar_to_ohlcv_dict


def test_alpaca_bar_to_ohlcv_dict_from_mapping() -> None:
    body = alpaca_bar_to_ohlcv_dict(
        {
            "t": datetime(2024, 1, 2, 15, 30, tzinfo=timezone.utc),
            "o": 10.0,
            "h": 11.0,
            "l": 9.5,
            "c": 10.5,
            "v": 1234.0,
        }
    )
    assert body["open"] == 10.0
    assert body["high"] == 11.0
    assert body["low"] == 9.5
    assert body["close"] == 10.5
    assert body["volume"] == 1234.0
    assert body["time"].startswith("2024-01-02T15:30:00")
    assert body["time"].endswith("Z")


def test_alpaca_bar_to_ohlcv_dict_naive_timestamp_utc() -> None:
    body = alpaca_bar_to_ohlcv_dict(
        {
            "t": datetime(2024, 1, 2, 15, 30),
            "o": 1,
            "h": 2,
            "l": 0.5,
            "c": 1.5,
            "v": 0,
        }
    )
    assert "2024-01-02T15:30:00" in body["time"]


def test_alpaca_bar_to_ohlcv_dict_from_bar_model() -> None:
    from alpaca.data.models import Bar

    ts = datetime(2024, 6, 1, 14, 30, tzinfo=timezone.utc)
    b = Bar(
        "AAPL",
        {"t": ts, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100.0, "n": 5, "vw": 1.25},
    )
    body = alpaca_bar_to_ohlcv_dict(b)
    assert body["open"] == 1.0
    assert body["volume"] == 100.0
    assert body["time"].startswith("2024-06-01T14:30:00")
