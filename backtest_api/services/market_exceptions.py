"""Errors raised by market data services when Yahoo/yfinance cannot satisfy a request."""


class MarketProviderError(Exception):
    """Upstream market data failed or returned unusable data."""
