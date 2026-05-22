"""Stable router errors for tests and logs (human messages may change)."""

from __future__ import annotations


class MarketRouteError(Exception):
    """
    Routing or eligibility failure before IO.

    ``code`` is stable (assert in tests); ``message`` is safe for HTTP clients.
    """

    def __init__(self, code: str, message: str, *, rule_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.rule_id = rule_id


# Stable codes (do not rename without updating tests)
MARKET_ROUTE_NO_CREDENTIALS = "MARKET_ROUTE_NO_CREDENTIALS"
MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN = "MARKET_ROUTE_YFINANCE_INTRADAY_FORBIDDEN"
MARKET_ROUTE_UNKNOWN_UPSTREAM = "MARKET_ROUTE_UNKNOWN_UPSTREAM"
MARKET_ROUTE_UNSUPPORTED_MODE = "MARKET_ROUTE_UNSUPPORTED_MODE"
