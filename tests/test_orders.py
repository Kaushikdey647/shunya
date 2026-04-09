"""Tests for the broker-agnostic order model: RiskPolicy, OrderBuilder, OrderSpec."""

from __future__ import annotations

import pytest

from shunya.algorithm.orders import (
    OrderBuilder,
    OrderSide,
    OrderSpec,
    OrderType,
    OrderVariety,
    RiskPolicy,
)


class TestRiskPolicy:
    def test_default_construction(self):
        rp = RiskPolicy()
        assert rp.stop_loss_pct == 0.02
        assert rp.max_single_position_pct == 0.10
        assert rp.round_lot_size == 1

    def test_negative_stop_loss_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RiskPolicy(stop_loss_pct=-0.01)

    def test_invalid_position_pct_raises(self):
        with pytest.raises(ValueError, match="max_single_position_pct"):
            RiskPolicy(max_single_position_pct=0.0)
        with pytest.raises(ValueError, match="max_single_position_pct"):
            RiskPolicy(max_single_position_pct=1.5)

    def test_invalid_lot_size_raises(self):
        with pytest.raises(ValueError, match="round_lot_size"):
            RiskPolicy(round_lot_size=0)

    def test_compute_stop_loss_buy(self):
        rp = RiskPolicy(stop_loss_pct=0.05)
        sl = rp.compute_stop_loss(100.0, OrderSide.BUY)
        assert sl == 95.0

    def test_compute_stop_loss_sell(self):
        rp = RiskPolicy(stop_loss_pct=0.05)
        sl = rp.compute_stop_loss(100.0, OrderSide.SELL)
        assert sl == 105.0

    def test_compute_stop_loss_zero_pct(self):
        rp = RiskPolicy(stop_loss_pct=0.0)
        assert rp.compute_stop_loss(100.0, OrderSide.BUY) == 100.0

    def test_compute_quantity_basic(self):
        rp = RiskPolicy(round_lot_size=1)
        assert rp.compute_quantity(1000.0, 50.0) == 20

    def test_compute_quantity_floors_partial_lot(self):
        rp = RiskPolicy(round_lot_size=1)
        assert rp.compute_quantity(99.0, 50.0) == 1

    def test_compute_quantity_lot_size_rounding(self):
        rp = RiskPolicy(round_lot_size=10)
        assert rp.compute_quantity(550.0, 10.0) == 50
        assert rp.compute_quantity(490.0, 10.0) == 40

    def test_compute_quantity_zero_price(self):
        rp = RiskPolicy()
        assert rp.compute_quantity(1000.0, 0.0) == 0

    def test_compute_quantity_negative_price(self):
        rp = RiskPolicy()
        assert rp.compute_quantity(1000.0, -10.0) == 0


class TestOrderBuilder:
    def test_build_market_buy(self):
        rp = RiskPolicy(stop_loss_pct=0.02)
        spec = OrderBuilder.build("AAPL", 5000.0, 150.0, rp)
        assert spec is not None
        assert spec.symbol == "AAPL"
        assert spec.side == OrderSide.BUY
        assert spec.quantity == 33
        assert spec.order_type == OrderType.MARKET
        assert spec.price is None
        assert spec.variety == OrderVariety.REGULAR

    def test_build_market_sell(self):
        rp = RiskPolicy()
        spec = OrderBuilder.build("TSLA", -3000.0, 200.0, rp)
        assert spec is not None
        assert spec.side == OrderSide.SELL
        assert spec.quantity == 15

    def test_build_returns_none_for_zero_delta(self):
        rp = RiskPolicy()
        assert OrderBuilder.build("AAPL", 0.0, 150.0, rp) is None

    def test_build_returns_none_for_dust_quantity(self):
        rp = RiskPolicy(round_lot_size=100)
        assert OrderBuilder.build("AAPL", 50.0, 150.0, rp) is None

    def test_build_raises_for_zero_price(self):
        rp = RiskPolicy()
        with pytest.raises(ValueError, match="price must be positive"):
            OrderBuilder.build("AAPL", 1000.0, 0.0, rp)

    def test_build_limit_order(self):
        rp = RiskPolicy()
        spec = OrderBuilder.build(
            "INFY", 10000.0, 1500.0, rp,
            order_type=OrderType.LIMIT,
            limit_price=1490.0,
        )
        assert spec is not None
        assert spec.order_type == OrderType.LIMIT
        assert spec.price == 1490.0

    def test_build_limit_uses_current_price_when_no_limit(self):
        rp = RiskPolicy()
        spec = OrderBuilder.build(
            "INFY", 10000.0, 1500.0, rp,
            order_type=OrderType.LIMIT,
        )
        assert spec is not None
        assert spec.price == 1500.0

    def test_build_cover_order_sets_trigger_and_sl(self):
        rp = RiskPolicy(stop_loss_pct=0.03)
        spec = OrderBuilder.build(
            "RELIANCE", 50000.0, 2500.0, rp,
            variety=OrderVariety.COVER,
        )
        assert spec is not None
        assert spec.variety == OrderVariety.COVER
        assert spec.trigger_price == 2425.0
        assert spec.stop_loss == 2425.0

    def test_build_cover_sell_trigger_above(self):
        rp = RiskPolicy(stop_loss_pct=0.02)
        spec = OrderBuilder.build(
            "RELIANCE", -50000.0, 2500.0, rp,
            variety=OrderVariety.COVER,
        )
        assert spec is not None
        assert spec.side == OrderSide.SELL
        assert spec.trigger_price == 2550.0

    def test_build_with_exchange_and_product(self):
        rp = RiskPolicy()
        spec = OrderBuilder.build(
            "INFY", 10000.0, 1500.0, rp,
            exchange="NSE",
            product="MIS",
        )
        assert spec is not None
        assert spec.exchange == "NSE"
        assert spec.product == "MIS"

    def test_build_market_protection(self):
        rp = RiskPolicy(default_market_protection=-1)
        spec = OrderBuilder.build("AAPL", 5000.0, 150.0, rp)
        assert spec is not None
        assert spec.market_protection == -1

    def test_build_many_filters_dust(self):
        rp = RiskPolicy()
        deltas = {"AAA": 5000.0, "BBB": 0.5, "CCC": -3000.0}
        prices = {"AAA": 100.0, "BBB": 100.0, "CCC": 200.0}
        specs = OrderBuilder.build_many(
            deltas, prices, rp, min_order_notional=1.0
        )
        symbols = {s.symbol for s in specs}
        assert "AAA" in symbols
        assert "CCC" in symbols
        assert "BBB" not in symbols

    def test_build_many_skips_missing_prices(self):
        rp = RiskPolicy()
        deltas = {"AAA": 5000.0, "BBB": 3000.0}
        prices = {"AAA": 100.0}
        specs = OrderBuilder.build_many(deltas, prices, rp)
        assert len(specs) == 1
        assert specs[0].symbol == "AAA"


class TestOrderSpec:
    def test_frozen(self):
        spec = OrderSpec(
            symbol="AAPL", side=OrderSide.BUY, quantity=10,
            order_type=OrderType.MARKET,
        )
        with pytest.raises(AttributeError):
            spec.quantity = 20  # type: ignore[misc]

    def test_defaults(self):
        spec = OrderSpec(
            symbol="AAPL", side=OrderSide.BUY, quantity=10,
            order_type=OrderType.MARKET,
        )
        assert spec.validity == "DAY"
        assert spec.variety == OrderVariety.REGULAR
        assert spec.price is None
        assert spec.trigger_price is None
        assert spec.exchange is None
        assert spec.product is None
        assert spec.iceberg_legs is None
        assert spec.market_protection is None
