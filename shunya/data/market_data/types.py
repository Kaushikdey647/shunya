"""Opaque identifiers and enums for market data routing (v1: dataclasses + literals)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

# --- Upstream vendors / feeds (stored in ohlcv_bars.source; never use "timescale" here) ---
UpstreamSourceId = Literal[
    "yfinance",
    "alpaca_sip",
    "alpaca_iex",
    "alpaca_delayed_sip",
]

UPSTREAM_IDS: tuple[str, ...] = (
    "yfinance",
    "alpaca_sip",
    "alpaca_iex",
    "alpaca_delayed_sip",
)


class Dataset(StrEnum):
    """Logical dataset (expand for quotes, news, …)."""

    OHLCV = "ohlcv"


class ReadPath(StrEnum):
    """How a response was materialized (orthogonal to upstream semantics)."""

    TIMESCALE = "timescale"
    LIVE_FETCH = "live_fetch"


class QualityTier(StrEnum):
    """Coarse entitlement / quality label for registry and JSONB metadata."""

    FREE = "free"
    DELAYED = "delayed"
    REALTIME = "realtime"
    BACKFILL = "backfill"
    BEST_EFFORT = "best_effort"


CachePolicy = Literal["prefer_timescale", "bypass_cache", "refresh_if_stale"]

RouteMode = Literal["auto", "best_effort"] | str


def is_upstream_source_id(value: str) -> bool:
    return value in UPSTREAM_IDS
