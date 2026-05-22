"""Capability rows (registry as data). v1: Python literals only."""

from __future__ import annotations

from dataclasses import dataclass

from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID


@dataclass(frozen=True, slots=True)
class UpstreamCapability:
    """One row in the capability matrix."""

    upstream_id: str
    authoritative_intraday: bool
    requires_alpaca_keys: bool


# Ordered matrix used for documentation and future UI; routing logic uses subsets.
UPSTREAM_CAPABILITIES: tuple[UpstreamCapability, ...] = (
    UpstreamCapability(
        STORED_OHLCV_DEFAULT_UPSTREAM_ID,
        authoritative_intraday=False,
        requires_alpaca_keys=False,
    ),
    UpstreamCapability("alpaca_sip", authoritative_intraday=True, requires_alpaca_keys=True),
    UpstreamCapability("alpaca_iex", authoritative_intraday=True, requires_alpaca_keys=True),
    UpstreamCapability("alpaca_delayed_sip", authoritative_intraday=True, requires_alpaca_keys=True),
)


def capability_for(upstream_id: str) -> UpstreamCapability | None:
    for c in UPSTREAM_CAPABILITIES:
        if c.upstream_id == upstream_id:
            return c
    return None
