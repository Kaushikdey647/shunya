"""
Pluggable market data for :class:`finTs` (research parity and broker-aligned history).

Implement :class:`MarketDataProvider` to swap Yahoo Finance for Alpaca bars
(or custom feeds) without changing :class:`~shunya.algorithm.finstrat.FinStrat`.
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd
import requests
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .timeframes import (
    BarIndexPolicy,
    BarSpec,
    BarUnit,
    bar_spec_is_intraday,
    bar_spec_to_alpaca_timeframe,
    bar_spec_to_yfinance_interval,
    default_bar_index_policy,
    default_bar_spec,
    normalize_history_index,
    resample_ohlcv_yearly,
)


@runtime_checkable
class MarketDataProvider(Protocol):
    """
    Download OHLCV in a ``yfinance``-compatible shape.

    Contract:
    - Index: ``DatetimeIndex``, named ``"Date"``. Interpretation follows ``bar_index_policy``
      (default: :func:`~.timeframes.default_bar_index_policy` — NY session clock, naive).
      Daily-like bars use midnight in the policy timezone (or UTC if ``daily_anchor='utc'``);
      intraday bars preserve wall clock after conversion to the policy zone.
    - Single ticker: flat OHLCV columns (e.g. ``Open``, ``High``, ``Low``, ``Close``, ``Volume``).
    - Multi ticker: column MultiIndex shaped as ``(Ticker, Field)``.
    """

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
    ) -> pd.DataFrame:
        """Return raw dataframe: MultiIndex columns per ticker for multi-name, or single-level for one ticker."""


def _resample_monthly_ohlcv_to_years(df: pd.DataFrame) -> pd.DataFrame:
    """Resample monthly OHLCV (single- or multi-ticker yfinance layout) to yearly bars."""
    if df.empty:
        out = df.copy()
        if out.index.name is None:
            out.index.name = "Date"
        return out
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique().tolist()
        pieces: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            sub = df[t].copy()
            pieces[str(t)] = resample_ohlcv_yearly(sub)
        out = pd.concat(pieces, axis=1)
        return out.sort_index()
    return resample_ohlcv_yearly(df)


def _alpaca_request_bounds(
    start: Union[str, pd.Timestamp], end: Union[str, pd.Timestamp], spec: BarSpec
) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if bar_spec_is_intraday(spec):
        return s, e
    return s.normalize(), e.normalize()


class YFinanceMarketDataProvider:
    """Yahoo Finance path; ``interval`` derives from :class:`~.timeframes.BarSpec`."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
    ) -> pd.DataFrame:
        spec = bar_spec if bar_spec is not None else default_bar_spec()
        idx_policy = (
            bar_index_policy
            if bar_index_policy is not None
            else default_bar_index_policy()
        )
        if not ticker_list:
            return pd.DataFrame()
        yfi_interval = bar_spec_to_yfinance_interval(spec)
        fetch_interval: str
        post_year_resample = False
        month_norm_spec = BarSpec(BarUnit.MONTHS, 1)
        year_norm_spec = BarSpec(BarUnit.YEARS, 1)

        if yfi_interval == "__monthly_then_year_resample":
            fetch_interval = "1mo"
            post_year_resample = True
        else:
            fetch_interval = yfi_interval

        dl_kw: dict = dict(
            start=start,
            end=end,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            interval=fetch_interval,
        )
        if self._session is not None:
            dl_kw["session"] = self._session

        df = yf.download(ticker_list, **dl_kw)
        if post_year_resample:
            df = _resample_monthly_ohlcv_to_years(df)
            return normalize_history_index(df, year_norm_spec, policy=idx_policy)
        return normalize_history_index(df, spec, policy=idx_policy)


class AlpacaHistoricalMarketDataProvider:
    """
    Historical stock bars from Alpaca Market Data (closer to broker tape than Yahoo).

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
        if not key or not sec:
            raise ValueError(
                "Alpaca credentials are required for AlpacaHistoricalMarketDataProvider. "
                "Set APCA_API_KEY_ID/APCA_API_SECRET_KEY or pass api_key/secret_key."
            )
        self._client = StockHistoricalDataClient(
            api_key=key, secret_key=sec, sandbox=paper
        )

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
    ) -> pd.DataFrame:
        if not ticker_list:
            return pd.DataFrame()

        spec = bar_spec if bar_spec is not None else default_bar_spec()
        idx_policy = (
            bar_index_policy
            if bar_index_policy is not None
            else default_bar_index_policy()
        )
        tf_mapped = bar_spec_to_alpaca_timeframe(spec)
        post_year_resample = False
        request_spec = spec
        month_norm = BarSpec(BarUnit.MONTHS, 1)
        year_norm = BarSpec(BarUnit.YEARS, 1)

        if tf_mapped == "__monthly_then_year_resample":
            timeframe = TimeFrame(1, TimeFrameUnit.Month)
            post_year_resample = True
            request_spec = month_norm
        else:
            timeframe = tf_mapped

        start_ts, end_ts = _alpaca_request_bounds(start, end, request_spec)
        req = StockBarsRequest(
            symbol_or_symbols=list(ticker_list),
            timeframe=timeframe,
            start=start_ts.to_pydatetime(),
            end=end_ts.to_pydatetime(),
        )
        try:
            barset = self._client.get_stock_bars(req)
        except Exception as exc:
            raise RuntimeError(
                "Alpaca historical bars request failed. "
                "Check credentials, symbol permissions, and network/API status."
            ) from exc

        norm_for_piece = month_norm if post_year_resample else spec
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
            if post_year_resample:
                part = normalize_history_index(part, month_norm, policy=idx_policy)
                part = resample_ohlcv_yearly(part)
                part = normalize_history_index(part, year_norm, policy=idx_policy)
            else:
                part = normalize_history_index(part, norm_for_piece, policy=idx_policy)
            frames.append(part)
            keys.append(sym)

        missing = [sym for sym in ticker_list if sym not in keys]
        if missing:
            raise ValueError(
                "Alpaca historical bars missing for symbols: "
                + ", ".join(sorted(missing))
            )
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1 and len(ticker_list) == 1:
            return frames[0]
        out = pd.concat({k: f for k, f in zip(keys, frames, strict=True)}, axis=1)
        return out


def _info_str(info: dict, key: str) -> Optional[str]:
    v = info.get(key)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _info_int(info: dict, key: str) -> Optional[int]:
    v = info.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v):
        iv = int(v)
        return iv if iv == v else None
    return None


def extract_yfinance_classification_fields(info: dict) -> Dict[str, Any]:
    """
    Map yfinance ``ticker.info`` to DB columns on ``symbol_classifications`` plus
    finTs keys ``Sector`` / ``Industry`` / ``SubIndustry``.

    ``sub_industry`` / ``SubIndustry`` come **only** from ``info["subIndustry"]``.
    """
    sector = _info_str(info, "sector")
    industry = _info_str(info, "industryDisp") or _info_str(info, "industry")
    sub_industry = _info_str(info, "subIndustry")

    out: Dict[str, Any] = {}
    if sector:
        out["sector"] = sector
        out["Sector"] = sector
    if industry:
        out["industry"] = industry
        out["Industry"] = industry
    if sub_industry:
        out["sub_industry"] = sub_industry
        out["SubIndustry"] = sub_industry

    text_pairs = (
        ("sector_disp", "sectorDisp"),
        ("sector_key", "sectorKey"),
        ("industry_disp", "industryDisp"),
        ("industry_key", "industryKey"),
        ("quote_type", "quoteType"),
        ("type_disp", "typeDisp"),
        ("exchange", "exchange"),
        ("full_exchange_name", "fullExchangeName"),
        ("currency", "currency"),
        ("region", "region"),
        ("market", "market"),
        ("country", "country"),
        ("state", "state"),
        ("city", "city"),
        ("zip", "zip"),
        ("website", "website"),
        ("phone", "phone"),
        ("ir_website", "irWebsite"),
        ("long_name", "longName"),
        ("short_name", "shortName"),
    )
    for db_k, info_k in text_pairs:
        s = _info_str(info, info_k)
        if s:
            out[db_k] = s

    ft = _info_int(info, "fullTimeEmployees")
    if ft is not None:
        out["full_time_employees"] = ft
    return out


def fetch_yfinance_classifications(
    ticker_list: List[str],
    *,
    session: Optional[requests.Session] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Best-effort classification lookup from yfinance.

    Returns per ticker a dict with lowercase DB keys (``sector``, ``industry``,
    ``sub_industry``, …), ``full_time_employees`` when present, and finTs keys
    ``Sector`` / ``Industry`` / ``SubIndustry`` (``SubIndustry`` from ``subIndustry`` only).
    """
    out: Dict[str, Dict[str, Any]] = {}
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

        out[sym] = extract_yfinance_classification_fields(info)
    return out
