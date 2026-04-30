"""
Pluggable market data for :class:`finTs` (research parity and broker-aligned history).

Implement :class:`MarketDataProvider` to swap Yahoo Finance for Alpaca bars
(or custom feeds) without changing :class:`~shunya.algorithm.finstrat.FinStrat`.
"""

from __future__ import annotations

import logging
import os
import math
import time
from typing import Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable

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

_LOG = logging.getLogger(__name__)

_OHLCV_COLS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")
_REPAIRED_META = "Repaired?"


def env_yfinance_repair_default() -> bool:
    """
    Whether Yahoo downloads should use yfinance ``repair=True`` (price/dividend fixes).

    Reads ``SHUNYA_YFINANCE_REPAIR`` or ``SHUNYA_API_YFINANCE_REPAIR``; unset means enabled.
    Truthy: 1, true, yes, on (case-insensitive). Falsy: 0, false, no, off, empty.
    """
    for key in ("SHUNYA_YFINANCE_REPAIR", "SHUNYA_API_YFINANCE_REPAIR"):
        raw = os.environ.get(key)
        if raw is None or not str(raw).strip():
            continue
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    return True


def sanitize_yfinance_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only OHLCV columns so downstream finTa/finta and ingest see a stable contract.

    Strips yfinance ``repair=True`` metadata (e.g. ``Repaired?``) after optional logging.
    """
    if df.empty:
        return df.copy()

    def _log_repaired(part: pd.DataFrame) -> None:
        if _REPAIRED_META not in part.columns:
            return
        s = pd.to_numeric(part[_REPAIRED_META], errors="coerce").fillna(0)
        # Booleans / 0-1 flags
        truth = s.astype(float) > 0.5
        n = int(truth.sum())
        if n > 0:
            _LOG.debug(
                "yfinance repair: %d/%d rows marked repaired in chunk",
                n,
                len(part),
            )

    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique().tolist()
        pieces: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            sub = df[str(t)].copy()
            _log_repaired(sub)
            keep = [c for c in _OHLCV_COLS if c in sub.columns]
            if len(keep) < len(_OHLCV_COLS):
                miss = [c for c in _OHLCV_COLS if c not in sub.columns]
                raise KeyError(f"yfinance OHLCV missing columns after repair: {miss} (ticker={t!r})")
            pieces[str(t)] = sub[list(_OHLCV_COLS)]
        out = pd.concat(pieces, axis=1)
        return out
    work = df.copy()
    _log_repaired(work)
    keep = [c for c in _OHLCV_COLS if c in work.columns]
    if len(keep) < len(_OHLCV_COLS):
        miss = [c for c in _OHLCV_COLS if c not in work.columns]
        raise KeyError(f"yfinance OHLCV missing columns after repair: {miss}")
    return work[list(_OHLCV_COLS)]


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

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        repair: bool = True,
    ) -> None:
        self._session = session
        self._repair = repair

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
        repair: Optional[bool] = None,
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

        use_repair = self._repair if repair is None else repair
        dl_kw: dict = dict(
            start=start,
            end=end,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            interval=fetch_interval,
            repair=use_repair,
        )
        if self._session is not None:
            dl_kw["session"] = self._session

        df = yf.download(ticker_list, **dl_kw)
        # Avoid read-only buffers from numpy/yfinance when repair mutates arrays.
        df = df.copy()
        df = sanitize_yfinance_ohlcv(df)
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


_ALPHAVANTAGE_QUERY_URL = "https://www.alphavantage.co/query"
_TIME_SERIES_DAILY = "TIME_SERIES_DAILY"


def alphavantage_resolve_api_key(explicit: Optional[str] = None) -> str:
    """API key from ``explicit`` or ``ALPHAVANTAGE_API_KEY`` / ``ALPHA_VANTAGE_API_KEY``."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for key in ("ALPHAVANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    raise ValueError(
        "Alpha Vantage API key required: set ALPHAVANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY "
        "or pass api_key= to AlphaVantageMarketDataProvider."
    )


def alphavantage_daily_payload_to_ohlcv(
    payload: dict[str, Any],
    *,
    symbol: str,
) -> pd.DataFrame:
    """
    Parse Alpha Vantage JSON for ``TIME_SERIES_DAILY`` into a single-ticker OHLCV frame.

    Raises ``ValueError`` / ``RuntimeError`` for API errors, rate-limit notes, or missing series.
    """
    err = payload.get("Error Message")
    if isinstance(err, str) and err.strip():
        raise ValueError(f"alphavantage {symbol!r}: {err.strip()}")

    info = payload.get("Information")
    if isinstance(info, str) and info.strip():
        raise RuntimeError(f"alphavantage {symbol!r}: {info.strip()}")

    note = payload.get("Note")
    if isinstance(note, str) and note.strip():
        raise RuntimeError(f"alphavantage {symbol!r}: {note.strip()}")

    raw_series = payload.get("Time Series (Daily)")
    if not isinstance(raw_series, dict):
        raise ValueError(
            f"alphavantage {symbol!r}: missing 'Time Series (Daily)' "
            f"(keys={list(payload.keys())})"
        )

    records: list[dict[str, float]] = []
    idx_list: list[pd.Timestamp] = []
    for date_str in sorted(raw_series.keys()):
        row = raw_series[date_str]
        if not isinstance(row, dict):
            continue
        records.append(
            {
                "Open": float(row["1. open"]),
                "High": float(row["2. high"]),
                "Low": float(row["3. low"]),
                "Close": float(row["4. close"]),
                "Volume": float(row["5. volume"]),
            }
        )
        idx_list.append(pd.Timestamp(date_str))
    if not records:
        return pd.DataFrame(columns=list(_OHLCV_COLS))

    out = pd.DataFrame(records, index=pd.DatetimeIndex(idx_list))
    out.index.name = "Date"
    return out[list(_OHLCV_COLS)]


class AlphaVantageMarketDataProvider:
    """
    Daily OHLCV via Alpha Vantage ``TIME_SERIES_DAILY`` (one HTTP request per symbol).

    Free-tier keys typically only receive ``outputsize=compact`` (~100 points); ``full`` requires
    a premium plan per Alpha Vantage documentation.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        outputsize: Literal["compact", "full"] = "compact",
        inter_request_delay_seconds: float = 0.0,
    ) -> None:
        self._api_key = alphavantage_resolve_api_key(api_key)
        self._session = session
        self._outputsize = outputsize
        self._delay = max(0.0, float(inter_request_delay_seconds))

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
        if spec.unit != BarUnit.DAYS or spec.step != 1:
            raise ValueError(
                "AlphaVantageMarketDataProvider only supports daily bars (BarSpec(DAYS, 1)); "
                f"got {spec!r}"
            )

        idx_policy = (
            bar_index_policy
            if bar_index_policy is not None
            else default_bar_index_policy()
        )
        if not ticker_list:
            return pd.DataFrame()

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts <= start_ts:
            return pd.DataFrame()

        sess = self._session if self._session is not None else requests.Session()
        frames: List[pd.DataFrame] = []
        keys: List[str] = []

        for i, sym in enumerate(ticker_list):
            if i > 0 and self._delay > 0:
                time.sleep(self._delay)

            params = {
                "function": _TIME_SERIES_DAILY,
                "symbol": sym,
                "apikey": self._api_key,
                "datatype": "json",
                "outputsize": self._outputsize,
            }
            resp = sess.get(_ALPHAVANTAGE_QUERY_URL, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()

            part = alphavantage_daily_payload_to_ohlcv(payload, symbol=sym)
            part = part.loc[(part.index >= start_ts) & (part.index < end_ts)]
            part = normalize_history_index(part, spec, policy=idx_policy)
            if part.empty:
                raise ValueError(
                    f"alphavantage: no daily bars for {sym!r} in [{start_ts.date()}, {end_ts.date()}) "
                    "(check outputsize=compact vs requested window; premium outputsize=full may be required)."
                )
            frames.append(part)
            keys.append(sym)

        if len(frames) == 1 and len(ticker_list) == 1:
            return frames[0]
        return pd.concat({k: f for k, f in zip(keys, frames, strict=True)}, axis=1)


def tiingo_resolve_api_key(explicit: Optional[str] = None) -> str:
    """API token from ``explicit``, ``SHUNYA_TIINGO_API_KEY``, or ``TIINGO_API_KEY``."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for key in ("SHUNYA_TIINGO_API_KEY", "TIINGO_API_KEY"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    raise ValueError(
        "Tiingo API token required: set SHUNYA_TIINGO_API_KEY or TIINGO_API_KEY "
        "or pass api_key= to TiingoMarketDataProvider."
    )


def ticker_to_tiingo_symbol(ticker: str) -> str:
    """Map DB/Yahoo-style tickers to Tiingo symbology (e.g. ``BRK.B`` → ``BRK-B``)."""
    return str(ticker).strip().upper().replace(".", "-")


def tiingo_daily_json_to_ohlcv(payload: Any, *, symbol: str) -> pd.DataFrame:
    """
    Parse Tiingo EOD ``get_ticker_price(..., fmt='json')`` payload into a single-ticker OHLCV frame.

    Uses unadjusted ``open``/``high``/``low``/``close`` and ``volume`` (not ``adj*`` fields).
    """
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail is not None:
            raise ValueError(f"tiingo {symbol!r}: {detail}")
        raise ValueError(
            f"tiingo {symbol!r}: unexpected JSON object (keys={list(payload.keys())})"
        )
    if not isinstance(payload, list):
        raise ValueError(
            f"tiingo {symbol!r}: expected a list of bars, got {type(payload).__name__}"
        )

    records: list[dict[str, float]] = []
    idx_list: list[pd.Timestamp] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        ds = row.get("date")
        if ds is None:
            continue
        try:
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            v = float(row.get("volume", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        records.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
        idx_list.append(pd.Timestamp(ds))

    if not records:
        return pd.DataFrame(columns=list(_OHLCV_COLS))

    out = pd.DataFrame(records, index=pd.DatetimeIndex(idx_list))
    out = out.sort_index()
    out.index.name = "Date"
    # Tiingo returns UTC-aware datetimes; use naive UTC dates for [start, end) filters like Alpha Vantage.
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        out.index = idx.tz_convert("UTC").tz_localize(None)
    return out[list(_OHLCV_COLS)]


class TiingoMarketDataProvider:
    """
    Daily OHLCV via Tiingo EOD (``tiingo/daily/{ticker}/prices``), one request per symbol.

    Requires ``SHUNYA_TIINGO_API_KEY`` or ``TIINGO_API_KEY`` (or ``api_key=`` / inject ``client``).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        client: Any = None,
        inter_request_delay_seconds: float = 0.0,
    ) -> None:
        self._delay = max(0.0, float(inter_request_delay_seconds))
        if client is not None:
            self._client = client
        else:
            from tiingo import TiingoClient

            key = tiingo_resolve_api_key(api_key)
            self._client = TiingoClient({"api_key": key})

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
        if spec.unit != BarUnit.DAYS or spec.step != 1:
            raise ValueError(
                "TiingoMarketDataProvider only supports daily bars (BarSpec(DAYS, 1)); "
                f"got {spec!r}"
            )

        idx_policy = (
            bar_index_policy
            if bar_index_policy is not None
            else default_bar_index_policy()
        )
        if not ticker_list:
            return pd.DataFrame()

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts <= start_ts:
            return pd.DataFrame()

        last_date = end_ts - pd.Timedelta(days=1)
        if last_date < start_ts:
            return pd.DataFrame()

        start_str = start_ts.strftime("%Y-%m-%d")
        end_str = last_date.strftime("%Y-%m-%d")

        frames: List[pd.DataFrame] = []
        keys: List[str] = []

        for i, sym in enumerate(ticker_list):
            if i > 0 and self._delay > 0:
                time.sleep(self._delay)

            tiingo_sym = ticker_to_tiingo_symbol(sym)
            raw = self._client.get_ticker_price(
                tiingo_sym,
                startDate=start_str,
                endDate=end_str,
                fmt="json",
                frequency="daily",
            )
            part = tiingo_daily_json_to_ohlcv(raw, symbol=sym)
            part = part.loc[(part.index >= start_ts) & (part.index < end_ts)]
            part = normalize_history_index(part, spec, policy=idx_policy)
            if part.empty:
                raise ValueError(
                    f"tiingo: no daily bars for {sym!r} (Tiingo symbol {tiingo_sym!r}) "
                    f"in [{start_ts.date()}, {end_ts.date()})"
                )
            frames.append(part)
            keys.append(sym)

        if len(frames) == 1 and len(ticker_list) == 1:
            return frames[0]
        return pd.concat({k: f for k, f in zip(keys, frames, strict=True)}, axis=1)


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
    finTs keys ``Sector`` and ``Industry``.
    """
    sector = _info_str(info, "sector")
    industry = _info_str(info, "industryDisp") or _info_str(info, "industry")

    out: Dict[str, Any] = {}
    if sector:
        out["sector"] = sector
        out["Sector"] = sector
    if industry:
        out["industry"] = industry
        out["Industry"] = industry

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

    Returns per ticker a dict with lowercase DB keys (``sector``, ``industry``, …),
    ``full_time_employees`` when present, and finTs keys ``Sector`` / ``Industry``.
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
