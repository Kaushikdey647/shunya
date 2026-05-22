"""Alpaca L1 quote/trade JSON normalization (dict + model fixtures)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from alpaca.data.models.quotes import Quote
from alpaca.data.models.trades import Trade, TradeCancel, TradeCorrection

from api.services.alpaca_l1_payload import (
    alpaca_quote_to_dict,
    alpaca_trade_cancel_to_dict,
    alpaca_trade_correction_to_dict,
    alpaca_trade_to_dict,
)


def test_alpaca_quote_to_dict_from_mapping() -> None:
    d = alpaca_quote_to_dict(
        {
            "S": "aapl",
            "t": datetime(2024, 1, 2, 15, 30, 0, tzinfo=timezone.utc),
            "bp": 100.0,
            "bs": 10.0,
            "ap": 100.5,
            "as": 20.0,
            "bx": "V",
            "ax": "P",
            "c": ["R"],
            "z": "A",
        }
    )
    assert d["symbol"] == "AAPL"
    assert d["time"].endswith("Z")
    assert d["bid_price"] == 100.0
    assert d["ask_size"] == 20.0


def test_alpaca_quote_to_dict_from_model() -> None:
    raw = {
        "t": "2024-01-02T15:30:00Z",
        "bp": 1.0,
        "bs": 2.0,
        "ap": 3.0,
        "as": 4.0,
    }
    q = Quote("MSFT", raw)
    d = alpaca_quote_to_dict(q)
    assert d["symbol"] == "MSFT"
    assert d["bid_price"] == 1.0


def test_alpaca_trade_to_dict_from_mapping() -> None:
    d = alpaca_trade_to_dict(
        {
            "S": "ibm",
            "t": datetime(2024, 1, 2, 15, 31, 0, tzinfo=timezone.utc),
            "p": 50.25,
            "s": 100.0,
            "i": 999,
            "x": "V",
        }
    )
    assert d["symbol"] == "IBM"
    assert d["price"] == 50.25
    assert d["size"] == 100.0
    assert d["id"] == 999


def test_alpaca_trade_to_dict_from_model() -> None:
    raw = {"t": "2024-01-02T15:31:00Z", "p": 10.0, "s": 1.0}
    t = Trade("GOOG", raw)
    d = alpaca_trade_to_dict(t)
    assert d["symbol"] == "GOOG"
    assert d["price"] == 10.0


def test_alpaca_trade_correction_to_dict() -> None:
    raw = {
        "t": "2024-01-02T16:00:00Z",
        "x": "V",
        "oi": 1,
        "op": 10.0,
        "os": 2.0,
        "oc": ["@"],
        "ci": 2,
        "cp": 10.5,
        "cs": 2.0,
        "cc": ["@"],
        "z": "A",
    }
    c = TradeCorrection("AAPL", raw)
    d = alpaca_trade_correction_to_dict(c)
    assert d["symbol"] == "AAPL"
    assert d["corrected_price"] == 10.5


def test_alpaca_trade_cancel_to_dict() -> None:
    raw = {
        "t": "2024-01-02T16:01:00Z",
        "x": "V",
        "p": 10.0,
        "s": 1.0,
        "i": 42,
        "z": "A",
    }
    c = TradeCancel("AAPL", raw)
    d = alpaca_trade_cancel_to_dict(c)
    assert d["symbol"] == "AAPL"
    assert d["id"] == 42


@pytest.mark.parametrize(
    "fn,bad",
    [
        (alpaca_quote_to_dict, {}),
        (alpaca_trade_to_dict, {}),
    ],
)
def test_alpaca_l1_raises_on_missing_fields(fn, bad) -> None:
    with pytest.raises(ValueError):
        fn(bad)
