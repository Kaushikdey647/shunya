"""Cached yfinance Ticker getters beyond overview (analyst, estimates, insider, etc.)."""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import pandas as pd
import yfinance as yf

from api.schemas.models import (
    InstrumentAnalystPriceTargetsResponse,
    InstrumentJsonBlobResponse,
    InstrumentValuationMeasuresPayload,
    InstrumentYfinanceTablePayload,
    InstrumentYfinanceTableResponse,
)
from api.services.instrument_cache_store import (
    instrument_yfinance_document_get,
    instrument_yfinance_document_put,
)
from api.services.yfinance_tables import dataframe_to_records, dict_to_jsonable
from shunya.data.timescale.market_cache_lib import (
    DOC_ANALYST_PRICE_TARGETS,
    DOC_CALENDAR,
    DOC_EARNINGS_ESTIMATE,
    DOC_EARNINGS_HISTORY,
    DOC_EPS_REVISIONS,
    DOC_EPS_TREND,
    DOC_GROWTH_ESTIMATES,
    DOC_INSIDER_PURCHASES,
    DOC_INSIDER_ROSTER_HOLDERS,
    DOC_INSIDER_TRANSACTIONS,
    DOC_MAJOR_HOLDERS,
    DOC_RECOMMENDATIONS,
    DOC_RECOMMENDATIONS_SUMMARY,
    DOC_REVENUE_ESTIMATE,
    DOC_SEC_FILINGS,
    DOC_SUSTAINABILITY,
    DOC_UPGRADES_DOWNGRADES,
    DOC_VALUATION_MEASURES,
)
from shunya.data.yfinance_session import build_yfinance_session

_log = logging.getLogger(__name__)


def _float_opt(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    return None


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol, session=build_yfinance_session())


def _table_response(symbol: str, df: pd.DataFrame | None) -> InstrumentYfinanceTableResponse:
    cols, recs = dataframe_to_records(df if isinstance(df, pd.DataFrame) else None)
    return InstrumentYfinanceTableResponse(
        symbol=symbol,
        available=bool(recs),
        data=InstrumentYfinanceTablePayload(columns=cols, records=recs),
    )


def _fetch_df_cached(
    symbol: str,
    *,
    resource_type: str,
    resource_key: str,
    loader: Callable[[yf.Ticker], Any],
) -> InstrumentYfinanceTableResponse:
    hit = instrument_yfinance_document_get(
        InstrumentYfinanceTableResponse,
        symbol=symbol,
        resource_type=resource_type,
        resource_key=resource_key,
    )
    if hit is not None:
        return hit
    df: pd.DataFrame | None = None
    try:
        t = _ticker(symbol)
        raw = loader(t)
        df = raw if isinstance(raw, pd.DataFrame) else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance %s failed for %s: %s", resource_type, symbol, exc)
        df = None
    out = _table_response(symbol, df)
    instrument_yfinance_document_put(
        symbol=symbol, resource_type=resource_type, resource_key=resource_key, obj=out
    )
    return out


def fetch_instrument_valuation_measures(symbol: str) -> InstrumentValuationMeasuresPayload:
    hit = instrument_yfinance_document_get(
        InstrumentValuationMeasuresPayload,
        symbol=symbol,
        resource_type=DOC_VALUATION_MEASURES,
        resource_key="",
    )
    if hit is not None:
        return hit
    df: pd.DataFrame | None = None
    try:
        t = _ticker(symbol)
        df = t.get_valuation_measures()
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance valuation_measures failed for %s: %s", symbol, exc)
        df = None
    cols, recs = dataframe_to_records(df if isinstance(df, pd.DataFrame) else None)
    out = InstrumentValuationMeasuresPayload(
        symbol=symbol, available=bool(recs), columns=cols, records=recs
    )
    instrument_yfinance_document_put(
        symbol=symbol, resource_type=DOC_VALUATION_MEASURES, resource_key="", obj=out
    )
    return out


def fetch_instrument_analyst_price_targets(symbol: str) -> InstrumentAnalystPriceTargetsResponse:
    hit = instrument_yfinance_document_get(
        InstrumentAnalystPriceTargetsResponse,
        symbol=symbol,
        resource_type=DOC_ANALYST_PRICE_TARGETS,
        resource_key="",
    )
    if hit is not None:
        return hit
    d: dict[str, Any] = {}
    try:
        t = _ticker(symbol)
        raw = t.get_analyst_price_targets()
        if isinstance(raw, dict):
            d = raw
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance analyst_price_targets failed for %s: %s", symbol, exc)
        d = {}
    out = InstrumentAnalystPriceTargetsResponse(
        symbol=symbol,
        available=bool(d),
        current=_float_opt(d.get("current")),
        low=_float_opt(d.get("low")),
        high=_float_opt(d.get("high")),
        mean=_float_opt(d.get("mean")),
        median=_float_opt(d.get("median")),
    )
    instrument_yfinance_document_put(
        symbol=symbol, resource_type=DOC_ANALYST_PRICE_TARGETS, resource_key="", obj=out
    )
    return out


def fetch_instrument_earnings_estimate(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_EARNINGS_ESTIMATE,
        resource_key="",
        loader=lambda t: t.get_earnings_estimate(),
    )


def fetch_instrument_revenue_estimate(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_REVENUE_ESTIMATE,
        resource_key="",
        loader=lambda t: t.get_revenue_estimate(),
    )


def fetch_instrument_earnings_history(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_EARNINGS_HISTORY,
        resource_key="",
        loader=lambda t: t.get_earnings_history(),
    )


def fetch_instrument_eps_trend(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_EPS_TREND,
        resource_key="",
        loader=lambda t: t.get_eps_trend(),
    )


def fetch_instrument_eps_revisions(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_EPS_REVISIONS,
        resource_key="",
        loader=lambda t: t.get_eps_revisions(),
    )


def fetch_instrument_growth_estimates(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_GROWTH_ESTIMATES,
        resource_key="",
        loader=lambda t: t.get_growth_estimates(),
    )


def fetch_instrument_recommendations(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_RECOMMENDATIONS,
        resource_key="",
        loader=lambda t: t.get_recommendations(),
    )


def fetch_instrument_recommendations_summary(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_RECOMMENDATIONS_SUMMARY,
        resource_key="",
        loader=lambda t: t.get_recommendations_summary(),
    )


def fetch_instrument_upgrades_downgrades(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_UPGRADES_DOWNGRADES,
        resource_key="",
        loader=lambda t: t.get_upgrades_downgrades(),
    )


def fetch_instrument_sustainability(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_SUSTAINABILITY,
        resource_key="",
        loader=lambda t: t.get_sustainability(),
    )


def fetch_instrument_insider_purchases(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_INSIDER_PURCHASES,
        resource_key="",
        loader=lambda t: t.get_insider_purchases(),
    )


def fetch_instrument_insider_transactions(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_INSIDER_TRANSACTIONS,
        resource_key="",
        loader=lambda t: t.get_insider_transactions(),
    )


def fetch_instrument_insider_roster_holders(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_INSIDER_ROSTER_HOLDERS,
        resource_key="",
        loader=lambda t: t.get_insider_roster_holders(),
    )


def fetch_instrument_major_holders(symbol: str) -> InstrumentYfinanceTableResponse:
    return _fetch_df_cached(
        symbol,
        resource_type=DOC_MAJOR_HOLDERS,
        resource_key="",
        loader=lambda t: t.get_major_holders(),
    )


def fetch_instrument_calendar(symbol: str) -> InstrumentJsonBlobResponse:
    hit = instrument_yfinance_document_get(
        InstrumentJsonBlobResponse, symbol=symbol, resource_type=DOC_CALENDAR, resource_key=""
    )
    if hit is not None:
        return hit
    try:
        t = _ticker(symbol)
        raw = t.get_calendar()
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance calendar failed for %s: %s", symbol, exc)
        raw = {}
    d = raw if isinstance(raw, dict) else {}
    out = InstrumentJsonBlobResponse(
        symbol=symbol, available=bool(d), data=dict_to_jsonable(d) if d else {}
    )
    instrument_yfinance_document_put(symbol=symbol, resource_type=DOC_CALENDAR, resource_key="", obj=out)
    return out


def fetch_instrument_sec_filings(symbol: str) -> InstrumentJsonBlobResponse:
    hit = instrument_yfinance_document_get(
        InstrumentJsonBlobResponse, symbol=symbol, resource_type=DOC_SEC_FILINGS, resource_key=""
    )
    if hit is not None:
        return hit
    try:
        t = _ticker(symbol)
        raw = t.get_sec_filings()
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance sec_filings failed for %s: %s", symbol, exc)
        raw = {}
    d = raw if isinstance(raw, dict) else {}
    out = InstrumentJsonBlobResponse(
        symbol=symbol, available=bool(d), data=dict_to_jsonable(d) if d else {}
    )
    instrument_yfinance_document_put(symbol=symbol, resource_type=DOC_SEC_FILINGS, resource_key="", obj=out)
    return out
