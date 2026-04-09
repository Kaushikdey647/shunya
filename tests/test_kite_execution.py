"""Tests for KiteExecutionAdapter (KiteTrigger) with a mocked KiteConnect client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shunya.algorithm.kite_execution import KiteExecutionAdapter
from shunya.algorithm.orders import OrderSide, OrderSpec, OrderType, OrderVariety


def _mock_kite():
    kite = MagicMock()
    kite.place_order.return_value = "order-123"
    kite.orders.return_value = []
    kite.positions.return_value = {"net": [], "day": []}
    kite.margins.return_value = {"available": {"live_balance": "50000"}}
    kite.ltp.return_value = {"NSE:NIFTY 50": {"last_price": 22000.0}}
    kite.trigger_range.return_value = {
        "NSE:INFY": {"lower": 1400.0, "upper": 1600.0}
    }
    kite.order_history.return_value = [
        {"status": "COMPLETE", "filled_quantity": 10, "average_price": 1500.0}
    ]
    return kite


def _market_buy_spec(**overrides):
    defaults = dict(
        symbol="INFY",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        variety=OrderVariety.REGULAR,
        exchange="NSE",
        product="CNC",
    )
    defaults.update(overrides)
    return OrderSpec(**defaults)


class TestSubmitOrders:
    def test_dry_run_does_not_call_place_order(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec()
        attempts = adapter.submit_orders([spec], dry_run=True, correlation_id="cid")
        assert len(attempts) == 1
        assert attempts[0].success is True
        assert attempts[0].initial_status == "dry_run"
        kite.place_order.assert_not_called()

    def test_live_submit_calls_place_order(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec()
        attempts = adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        assert len(attempts) == 1
        assert attempts[0].success is True
        kite.place_order.assert_called_once()
        call_kw = kite.place_order.call_args
        assert call_kw.kwargs["variety"] == "regular"
        assert call_kw.kwargs["exchange"] == "NSE"
        assert call_kw.kwargs["tradingsymbol"] == "INFY"
        assert call_kw.kwargs["transaction_type"] == "BUY"
        assert call_kw.kwargs["quantity"] == 10
        assert call_kw.kwargs["order_type"] == "MARKET"

    def test_limit_order_passes_price(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec(order_type=OrderType.LIMIT, price=1490.0)
        adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        call_kw = kite.place_order.call_args.kwargs
        assert call_kw["order_type"] == "LIMIT"
        assert call_kw["price"] == 1490.0

    def test_cover_order_maps_variety_and_trigger(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec(
            variety=OrderVariety.COVER,
            trigger_price=1450.0,
        )
        adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        call_kw = kite.place_order.call_args.kwargs
        assert call_kw["variety"] == "co"
        assert call_kw["trigger_price"] == 1450.0

    def test_iceberg_params_forwarded(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec(
            variety=OrderVariety.ICEBERG,
            iceberg_legs=5,
            iceberg_quantity=2,
        )
        adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        call_kw = kite.place_order.call_args.kwargs
        assert call_kw["variety"] == "iceberg"
        assert call_kw["iceberg_legs"] == 5
        assert call_kw["iceberg_quantity"] == 2

    def test_market_protection_forwarded(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec(market_protection=-1)
        adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        call_kw = kite.place_order.call_args.kwargs
        assert call_kw["market_protection"] == -1

    def test_sell_order_transaction_type(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec(side=OrderSide.SELL)
        adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        call_kw = kite.place_order.call_args.kwargs
        assert call_kw["transaction_type"] == "SELL"

    def test_submit_retries_on_failure(self):
        kite = _mock_kite()
        kite.place_order.side_effect = [Exception("transient"), "order-456"]
        adapter = KiteExecutionAdapter(kite, retry_base_seconds=0.0)
        spec = _market_buy_spec()
        attempts = adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        assert attempts[0].success is True
        assert kite.place_order.call_count == 2

    def test_submit_fails_after_max_retries(self):
        kite = _mock_kite()
        kite.place_order.side_effect = Exception("permanent")
        adapter = KiteExecutionAdapter(kite, max_submit_retries=2, retry_base_seconds=0.0)
        spec = _market_buy_spec()
        attempts = adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        assert attempts[0].success is False
        assert "permanent" in attempts[0].error

    def test_default_product_used_when_spec_product_is_none(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite, default_product="MIS")
        spec = _market_buy_spec(product=None)
        adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        call_kw = kite.place_order.call_args.kwargs
        assert call_kw["product"] == "MIS"


class TestGetPositions:
    def test_parses_net_positions(self):
        kite = _mock_kite()
        kite.positions.return_value = {
            "net": [
                {"tradingsymbol": "INFY", "quantity": 10, "last_price": 1500.0},
                {"tradingsymbol": "TCS", "quantity": -5, "last_price": 3000.0},
            ],
            "day": [],
        }
        adapter = KiteExecutionAdapter(kite)
        positions = adapter.get_positions()
        assert positions["INFY"] == 15000.0
        assert positions["TCS"] == -15000.0


class TestIsMarketOpen:
    def test_returns_true_when_ltp_succeeds(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        assert adapter.is_market_open() is True

    def test_returns_false_when_ltp_fails(self):
        kite = _mock_kite()
        kite.ltp.side_effect = Exception("session expired")
        adapter = KiteExecutionAdapter(kite)
        assert adapter.is_market_open() is False


class TestBuyingPower:
    def test_reads_live_balance(self):
        kite = _mock_kite()
        adapter = KiteExecutionAdapter(kite)
        assert adapter.buying_power() == 50000.0


class TestCancelOpenOrders:
    def test_cancels_non_terminal_orders(self):
        kite = _mock_kite()
        kite.orders.return_value = [
            {"order_id": "o1", "variety": "regular", "status": "OPEN"},
            {"order_id": "o2", "variety": "co", "status": "TRIGGER PENDING"},
            {"order_id": "o3", "variety": "regular", "status": "COMPLETE"},
        ]
        adapter = KiteExecutionAdapter(kite)
        adapter.cancel_open_orders()
        kite.cancel_order.assert_called_once_with(variety="regular", order_id="o1")
        kite.exit_order.assert_called_once_with(variety="co", order_id="o2")


class TestTriggerRangeValidation:
    def test_warns_when_trigger_outside_range(self):
        kite = _mock_kite()
        kite.trigger_range.return_value = {
            "NSE:INFY": {"lower": 1400.0, "upper": 1500.0}
        }
        adapter = KiteExecutionAdapter(kite)
        spec = _market_buy_spec(
            variety=OrderVariety.COVER,
            trigger_price=1350.0,
        )
        # Should not raise, just warn
        attempts = adapter.submit_orders([spec], dry_run=False, correlation_id="cid")
        assert len(attempts) == 1
