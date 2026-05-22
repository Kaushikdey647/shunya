"""Thin Yahoo Finance facade for HTTP paths outside the OHLCV router (shared session).

Wraps raw ``yfinance`` calls so market/instrument surfaces reuse one session builder and
stay easy to mock in tests via ``session=`` injection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from shunya.data.yfinance_session import build_yfinance_session


class YahooPublicAdapter:
    """Centralize ``build_yfinance_session`` + yfinance entrypoints for public Yahoo reads."""

    __slots__ = ("_session",)

    def __init__(self, *, session: Any | None = None) -> None:
        self._session = session if session is not None else build_yfinance_session()

    @property
    def session(self) -> Any:
        return self._session

    def ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol, session=self._session)

    def download_daily_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        """Batch daily bars (same contract as :func:`api.services.market_snapshot.build_snapshot`)."""
        raw = yf.download(
            tickers=list(symbols),
            period="1mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False,
            session=self._session,
        )
        return raw if isinstance(raw, pd.DataFrame) else pd.DataFrame()

    def predefined_screen(self, screen_key: str, *, count: int) -> Any:
        return yf.screen(screen_key, count=count, session=self._session)

    def search_headlines(
        self,
        query: str,
        *,
        max_results: int = 8,
        news_count: int,
        timeout: int = 25,
    ) -> Any:
        return yf.Search(
            query,
            max_results=max_results,
            news_count=news_count,
            include_nav_links=False,
            timeout=timeout,
            raise_errors=True,
            session=self._session,
        )

    def search_instruments(
        self,
        query: str,
        *,
        max_results: int = 16,
        news_count: int = 12,
        include_nav_links: bool = True,
        timeout: int = 25,
    ) -> Any:
        return yf.Search(
            query,
            max_results=max_results,
            news_count=news_count,
            include_nav_links=include_nav_links,
            timeout=timeout,
            raise_errors=True,
            session=self._session,
        )
