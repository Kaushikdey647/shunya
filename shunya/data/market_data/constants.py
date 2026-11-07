"""Single source of truth for Yahoo-ingested rows stored in ``ohlcv_bars.source``.

Classification rows, fundamentals EAV, and other tables may use different ``source`` values
(e.g. ``yfinance_statements``); do not reuse this constant for those unless intentionally aligned.
"""

from __future__ import annotations

import os
from typing import Final, Literal

STORED_OHLCV_DEFAULT_UPSTREAM_ID: Final[Literal["yfinance"]] = "yfinance"


_STORED_SOURCES: Final[tuple[str, ...]] = ("yfinance", "alpaca_iex", "alpaca_sip", "alpaca_delayed_sip")


def stored_daily_ohlcv_source() -> str:
    """``ohlcv_bars.source`` that Timescale-backed finTs / backtests read.

    ``SHUNYA_STORED_OHLCV_SOURCE`` (default ``yfinance``). Set ``alpaca_iex`` after
    ``shunya-timescale ingest-ohlcv --provider alpaca`` so research reads the same feed the live
    runner trades on.
    """
    raw = (os.environ.get("SHUNYA_STORED_OHLCV_SOURCE") or STORED_OHLCV_DEFAULT_UPSTREAM_ID).strip().lower()
    if raw not in _STORED_SOURCES:
        raise ValueError(f"SHUNYA_STORED_OHLCV_SOURCE must be one of {_STORED_SOURCES}, got {raw!r}")
    return raw
