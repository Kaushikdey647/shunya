"""Single source of truth for Yahoo-ingested rows stored in ``ohlcv_bars.source``.

Classification rows, fundamentals EAV, and other tables may use different ``source`` values
(e.g. ``yfinance_statements``); do not reuse this constant for those unless intentionally aligned.
"""

from __future__ import annotations

from typing import Final, Literal

STORED_OHLCV_DEFAULT_UPSTREAM_ID: Final[Literal["yfinance"]] = "yfinance"
