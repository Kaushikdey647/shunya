"""Market dashboard symbol validation (aligned with instruments router rules)."""

from __future__ import annotations

import re

SYMBOL_RE = re.compile(r"^[A-Z0-9^.\-]{1,32}$")


def normalize_market_symbol(raw: str) -> str:
    """
    Uppercase/strip Yahoo-style symbols (equities, indexes like ``^VIX``).

    Raises:
        ValueError: If the string does not match the allowed pattern.
    """
    s = raw.strip().upper()
    if not SYMBOL_RE.match(s):
        raise ValueError("invalid symbol")
    return s
