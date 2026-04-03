"""
Pluggable market data for :class:`finTs` (research parity and broker-aligned history).

Implement :class:`MarketDataProvider` to swap Yahoo Finance for Alpaca daily bars
(or custom feeds) without changing :class:`~src.algorithm.finstrat.FinStrat`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd
import requests
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


@runtime_checkable
class MarketDataProvider(Protocol):
    """Download OHLCV in ``yfinance``-compatible shape (multi-ticker MultiIndex columns)."""

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """Return raw dataframe: MultiIndex columns per ticker for multi-name, or single-level for one ticker."""


class YFinanceMarketDataProvider:
    """Default Yahoo Finance path (same as historical ``finTs`` behavior)."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        return yf.download(
            ticker_list,
            start=start,
            end=end,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            **({"session": self._session} if self._session is not None else {}),
        )


class AlpacaHistoricalMarketDataProvider:
    """
    Daily bars from Alpaca Market Data (closer to broker tape than Yahoo for parity checks).

    Requires ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` (or explicit keys). Free-tier
    data may differ by symbol universe; handle API errors at call time.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ) -> None:
        key = api_key or os.environ.get("APCA_API_KEY_ID")
        sec = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        self._client = StockHistoricalDataClient(
            api_key=key, secret_key=sec, sandbox=paper
        )

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        if not ticker_list:
            return pd.DataFrame()
        req = StockBarsRequest(
            symbol_or_symbols=list(ticker_list),
            timeframe=TimeFrame.Day,
            start=pd.Timestamp(start).date(),
            end=pd.Timestamp(end).date(),
        )
        barset = self._client.get_stock_bars(req)
        frames: List[pd.DataFrame] = []
        keys: List[str] = []
        for sym in ticker_list:
            bars = barset.data.get(sym)
            if not bars:
                continue
            records = []
            idx = []
            for b in bars:
                records.append(
                    {
                        "Open": float(b.open),
                        "High": float(b.high),
                        "Low": float(b.low),
                        "Close": float(b.close),
                        "Volume": float(b.volume),
                    }
                )
                idx.append(pd.Timestamp(b.timestamp))
            part = pd.DataFrame(records, index=idx).sort_index()
            part.index.name = "Date"
            frames.append(part)
            keys.append(sym)
        if not frames:
            return pd.DataFrame()
        # Match yfinance multi-ticker layout: column MultiIndex (Ticker, Field).
        if len(frames) == 1 and len(ticker_list) == 1:
            return frames[0]
        return pd.concat({k: f for k, f in zip(keys, frames, strict=True)}, axis=1)


def fetch_yfinance_classifications(
    ticker_list: List[str],
    *,
    session: Optional[requests.Session] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Best-effort classification lookup from yfinance.

    Returns ``{ticker: {"Sector": ..., "Industry": ..., "SubIndustry": ...}}``.
    Missing fields are omitted here; callers should apply deterministic fallbacks.
    """
    out: Dict[str, Dict[str, str]] = {}
    for sym in ticker_list:
        info: dict = {}
        try:
            ticker = (
                yf.Ticker(sym, session=session)
                if session is not None
                else yf.Ticker(sym)
            )
            info = dict(ticker.info or {})
        except Exception:
            info = {}

        sector = info.get("sector")
        industry = info.get("industryDisp") or info.get("industry")
        # yfinance typically does not expose strict GICS sub-industry;
        # use industryKey as the closest available stable detail.
        subindustry = info.get("industryKey")

        fields: Dict[str, str] = {}
        if isinstance(sector, str) and sector.strip():
            fields["Sector"] = sector.strip()
        if isinstance(industry, str) and industry.strip():
            fields["Industry"] = industry.strip()
        if isinstance(subindustry, str) and subindustry.strip():
            fields["SubIndustry"] = subindustry.strip()
        out[sym] = fields
    return out
