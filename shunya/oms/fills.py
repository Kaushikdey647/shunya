"""Immutable execution fill records (append-only in persistence)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True, slots=True)
class ExecutionFill:
    """One broker-confirmed execution line tied to a parent order."""

    trade_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    ts: datetime
    child_client_order_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price < 0:
            raise ValueError("price must be non-negative")
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")


def vwap_avg_price(fills: list[ExecutionFill]) -> float:
    """Volume-weighted average price over a sequence of fills."""
    if not fills:
        return 0.0
    num = sum(float(f.price) * float(f.quantity) for f in fills)
    den = sum(float(f.quantity) for f in fills)
    if den <= 0:
        return 0.0
    return num / den


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
