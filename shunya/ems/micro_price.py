"""L1 quote micro-pricing (mid peg vs spread cross) for EMS child limits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MicroPriceUrgency(str, Enum):
    """How aggressively a child limit chases the offer."""

    MID = "mid"
    """Peg to bid/ask midpoint."""

    PASSIVE_TOUCH = "passive_touch"
    """Buy at bid (sell at ask): join the touch."""

    CROSS = "cross"
    """Buy at ask (sell at bid): pay the spread."""


@dataclass(frozen=True, slots=True)
class QuoteL1:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        if self.bid <= 0 or self.ask <= 0:
            return max(self.bid, self.ask, 0.0)
        return (self.bid + self.ask) / 2.0


def limit_price_for_child(
    quote: QuoteL1,
    *,
    side: str,
    urgency: MicroPriceUrgency,
) -> float:
    """Return a limit price for a child slice given urgency (US equities, long-only typical)."""
    s = str(side).upper()
    if s == "BUY":
        if urgency == MicroPriceUrgency.CROSS:
            return float(quote.ask)
        if urgency == MicroPriceUrgency.PASSIVE_TOUCH:
            return float(quote.bid)
        return float(quote.mid)
    if "SELL" in s:
        if urgency == MicroPriceUrgency.CROSS:
            return float(quote.bid)
        if urgency == MicroPriceUrgency.PASSIVE_TOUCH:
            return float(quote.ask)
        return float(quote.mid)
    raise ValueError("side must be BUY or SELL")
