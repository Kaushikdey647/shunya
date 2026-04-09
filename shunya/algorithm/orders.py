"""Broker-agnostic order model, risk policy, order builder, and execution protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .execution import OrderAttempt


class OrderType(Enum):
    """Broker-neutral order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class OrderVariety(Enum):
    """Broker-neutral order variety (maps to Kite varieties / Alpaca equivalents)."""

    REGULAR = "regular"
    COVER = "co"
    AMO = "amo"
    ICEBERG = "iceberg"
    AUCTION = "auction"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderSpec:
    """Broker-neutral order specification built by :class:`OrderBuilder`."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    variety: OrderVariety = OrderVariety.REGULAR
    notional: Optional[float] = None
    exchange: Optional[str] = None
    product: Optional[str] = None
    tag: Optional[str] = None
    validity: str = "DAY"
    validity_ttl: Optional[int] = None
    iceberg_legs: Optional[int] = None
    iceberg_quantity: Optional[int] = None
    market_protection: Optional[int] = None


@dataclass
class RiskPolicy:
    """
    Execution / risk configuration that owns stop-loss and sizing rules.

    ``stop_loss_pct`` is a non-negative fraction (e.g. 0.02 for 2%).
    ``round_lot_size`` is the minimum tradeable increment (1 for most US equities,
    varies for NSE).
    """

    stop_loss_pct: float = 0.02
    max_single_position_pct: float = 0.10
    round_lot_size: int = 1
    default_market_protection: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stop_loss_pct < 0:
            raise ValueError("stop_loss_pct must be non-negative")
        if not (0.0 < self.max_single_position_pct <= 1.0):
            raise ValueError("max_single_position_pct must be in (0, 1]")
        if self.round_lot_size < 1:
            raise ValueError("round_lot_size must be >= 1")

    def compute_stop_loss(self, price: float, side: OrderSide) -> float:
        """Adverse stop-loss: buys trigger below, sells trigger above."""
        if self.stop_loss_pct == 0:
            return price
        if side is OrderSide.BUY:
            return round(price * (1.0 - self.stop_loss_pct), 2)
        return round(price * (1.0 + self.stop_loss_pct), 2)

    def compute_quantity(self, delta_usd: float, price: float) -> int:
        """Whole-lot quantity from a signed USD delta and a per-share price."""
        if price <= 0:
            return 0
        raw = abs(delta_usd) / price
        lots = int(math.floor(raw / self.round_lot_size))
        return lots * self.round_lot_size


class OrderBuilder:
    """
    Converts alpha USD deltas + live prices into :class:`OrderSpec` instances.

    The builder is stateless; all policy comes from the :class:`RiskPolicy`.
    """

    @staticmethod
    def build(
        symbol: str,
        delta_usd: float,
        price: float,
        risk_policy: RiskPolicy,
        order_type: OrderType = OrderType.MARKET,
        variety: OrderVariety = OrderVariety.REGULAR,
        *,
        exchange: Optional[str] = None,
        product: Optional[str] = None,
        tag: Optional[str] = None,
        limit_price: Optional[float] = None,
        validity: str = "DAY",
        validity_ttl: Optional[int] = None,
        iceberg_legs: Optional[int] = None,
        iceberg_quantity: Optional[int] = None,
    ) -> Optional[OrderSpec]:
        """
        Build a single :class:`OrderSpec` from an alpha USD delta and a live price.

        Returns ``None`` when the computed quantity is zero (dust delta).
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if delta_usd == 0:
            return None

        side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
        qty = risk_policy.compute_quantity(delta_usd, price)
        if qty == 0:
            return None

        effective_price: Optional[float] = None
        if order_type is OrderType.LIMIT:
            effective_price = limit_price if limit_price is not None else round(price, 2)

        trigger: Optional[float] = None
        sl: Optional[float] = None
        if variety is OrderVariety.COVER:
            sl = risk_policy.compute_stop_loss(price, side)
            trigger = sl

        mp = risk_policy.default_market_protection

        return OrderSpec(
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=order_type,
            price=effective_price,
            trigger_price=trigger,
            stop_loss=sl,
            variety=variety,
            notional=round(abs(delta_usd), 2),
            exchange=exchange,
            product=product,
            tag=tag,
            validity=validity,
            validity_ttl=validity_ttl,
            iceberg_legs=iceberg_legs,
            iceberg_quantity=iceberg_quantity,
            market_protection=mp,
        )

    @staticmethod
    def build_many(
        deltas: Dict[str, float],
        prices: Dict[str, float],
        risk_policy: RiskPolicy,
        order_type: OrderType = OrderType.MARKET,
        variety: OrderVariety = OrderVariety.REGULAR,
        *,
        exchange: Optional[str] = None,
        product: Optional[str] = None,
        tag_prefix: Optional[str] = None,
        min_order_notional: float = 1.0,
        limit_prices: Optional[Dict[str, float]] = None,
        validity: str = "DAY",
    ) -> List[OrderSpec]:
        """Batch-build specs from delta and price dicts, filtering dust."""
        specs: List[OrderSpec] = []
        for sym, d in deltas.items():
            if abs(d) < min_order_notional:
                continue
            p = prices.get(sym)
            if p is None or p <= 0:
                continue
            lp = limit_prices.get(sym) if limit_prices else None
            tag = f"{tag_prefix}-{sym}" if tag_prefix else None
            spec = OrderBuilder.build(
                symbol=sym,
                delta_usd=d,
                price=p,
                risk_policy=risk_policy,
                order_type=order_type,
                variety=variety,
                exchange=exchange,
                product=product,
                tag=tag,
                limit_price=lp,
                validity=validity,
            )
            if spec is not None:
                specs.append(spec)
        return specs


@runtime_checkable
class ExecutionAdapter(Protocol):
    """
    Broker-agnostic execution interface.

    Implementations translate :class:`OrderSpec` into broker-specific API calls.
    """

    def submit_orders(
        self,
        orders: List[OrderSpec],
        *,
        dry_run: bool,
        correlation_id: str,
    ) -> List[OrderAttempt]: ...

    def get_positions(self) -> Dict[str, float]: ...

    def is_market_open(self) -> bool: ...

    def buying_power(self) -> float: ...

    def cancel_open_orders(self) -> None: ...
