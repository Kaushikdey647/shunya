from __future__ import annotations

# TODO(market-data-router): Instrument routes using yfinance directly should delegate to router/adapter stack.

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from api.schemas.models import (
    IngestionRunOut,
    InstrumentAnalystPriceTargetsResponse,
    InstrumentFinancialFrequencyLiteral,
    InstrumentFinancialStatementResponse,
    InstrumentHoldersResponse,
    InstrumentIvHeatmapResponse,
    InstrumentJsonBlobResponse,
    InstrumentNavLink,
    InstrumentOptionChainResponse,
    InstrumentOptionExpirationsResponse,
    InstrumentOhlcvResponse,
    InstrumentOverviewResponse,
    InstrumentSearchNewsItem,
    InstrumentSearchQuote,
    InstrumentSearchResponse,
    InstrumentStatementLiteral,
    InstrumentTickerNewsItem,
    InstrumentTickerNewsResponse,
    InstrumentValuationMeasuresPayload,
    InstrumentYfinanceTableResponse,
)
from api.services.instrument_dashboard import (
    fetch_instrument_financials,
    fetch_instrument_holders,
    fetch_instrument_overview,
    fetch_option_chain,
    fetch_option_expirations,
    fetch_option_iv_heatmap,
)
from api.services.instrument_yfinance_extended import (
    fetch_instrument_analyst_price_targets,
    fetch_instrument_calendar,
    fetch_instrument_earnings_estimate,
    fetch_instrument_earnings_history,
    fetch_instrument_eps_revisions,
    fetch_instrument_eps_trend,
    fetch_instrument_growth_estimates,
    fetch_instrument_insider_purchases,
    fetch_instrument_insider_roster_holders,
    fetch_instrument_insider_transactions,
    fetch_instrument_major_holders,
    fetch_instrument_recommendations,
    fetch_instrument_recommendations_summary,
    fetch_instrument_revenue_estimate,
    fetch_instrument_sec_filings,
    fetch_instrument_sustainability,
    fetch_instrument_upgrades_downgrades,
    fetch_instrument_valuation_measures,
)
from api.services.instrument_ohlcv import PendingOhlcvWriteback, resolve_instrument_ohlcv_sync
from api.services.market_symbols import SYMBOL_RE, normalize_market_symbol
from shunya.integration.yahoo_public import YahooPublicAdapter
from shunya.data.timescale.ohlcv_writeback import (
    create_deferred_ingestion_run_sync,
    get_ingestion_run_sync,
    replace_ohlcv_range_sync,
)

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/instruments", tags=["instruments"])

ALLOWED_INTERVALS = frozenset(
    {
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    }
)

ALLOWED_PERIODS = frozenset(
    {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }
)

MAX_SEARCH_LEN = 64


def _normalize_symbol(raw: str) -> str:
    try:
        return normalize_market_symbol(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid symbol") from exc


def _quote_from_raw_safe(item: dict[str, Any]) -> InstrumentSearchQuote | None:
    sym = item.get("symbol") or item.get("ticker")
    if not sym or not isinstance(sym, str):
        return None
    sym_u = sym.strip().upper()
    if not SYMBOL_RE.match(sym_u):
        return None
    exch = item.get("exchDisp")
    if not isinstance(exch, str):
        exch = item.get("exchange")
    if not isinstance(exch, str):
        exch = None
    sn = item.get("shortname")
    ln = item.get("longname")
    qt = item.get("typeDisp") or item.get("quoteType")
    return InstrumentSearchQuote(
        symbol=sym_u,
        shortname=sn if isinstance(sn, str) else None,
        longname=ln if isinstance(ln, str) else None,
        exchange=exch,
        quote_type=qt if isinstance(qt, str) else None,
    )


def _news_from_raw(item: dict[str, Any]) -> InstrumentSearchNewsItem | None:
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    link = item.get("link")
    pub = item.get("publisher")
    return InstrumentSearchNewsItem(
        title=title.strip(),
        link=link if isinstance(link, str) else None,
        publisher=pub if isinstance(pub, str) else None,
    )


def _published_at_from_raw(pub_raw: Any, ppt_raw: Any) -> str | None:
    if isinstance(pub_raw, str) and pub_raw.strip():
        return pub_raw.strip()
    if isinstance(ppt_raw, (int, float)):
        ts = int(ppt_raw)
        if ts > 10_000_000_000:
            ts //= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return None


def _str_opt(val: Any) -> str | None:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _news_item_from_ticker_news_dict(item: dict[str, Any]) -> InstrumentTickerNewsItem | None:
    """Map yfinance ``Ticker.news`` entry (nested ``content`` or legacy flat) to API model."""
    if not isinstance(item, dict):
        return None
    inner = item.get("content")
    if isinstance(inner, dict):
        title = inner.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        pub_raw = inner.get("pubDate") or inner.get("displayTime")
        published_at = _published_at_from_raw(pub_raw, None)
        link: str | None = None
        cu = inner.get("canonicalUrl")
        site = region = lang = None
        if isinstance(cu, dict):
            u = cu.get("url")
            if isinstance(u, str) and u.strip():
                link = u.strip()
            site = _str_opt(cu.get("site"))
            region = _str_opt(cu.get("region"))
            lang = _str_opt(cu.get("lang"))
        if link is None:
            ct = inner.get("clickThroughUrl")
            if isinstance(ct, dict):
                u = ct.get("url")
                if isinstance(u, str) and u.strip():
                    link = u.strip()
        publisher = None
        provider_url = None
        provider_source_id = None
        prov = inner.get("provider")
        if isinstance(prov, dict):
            publisher = _str_opt(prov.get("displayName"))
            provider_url = _str_opt(prov.get("url"))
            provider_source_id = _str_opt(prov.get("sourceId"))
        summary = _str_opt(inner.get("summary"))
        description = _str_opt(inner.get("description"))
        content_type = _str_opt(inner.get("contentType"))
        story_id = _str_opt(item.get("id")) or _str_opt(inner.get("id"))
        is_hosted = inner.get("isHosted") if isinstance(inner.get("isHosted"), bool) else None
        thumb_url = None
        thumb = inner.get("thumbnail")
        if isinstance(thumb, dict):
            thumb_url = _str_opt(thumb.get("originalUrl"))
        editors_pick = None
        meta = inner.get("metadata")
        if isinstance(meta, dict) and isinstance(meta.get("editorsPick"), bool):
            editors_pick = meta["editorsPick"]
        is_premium_news = is_premium_free = None
        fin = inner.get("finance")
        if isinstance(fin, dict):
            pf = fin.get("premiumFinance")
            if isinstance(pf, dict):
                if isinstance(pf.get("isPremiumNews"), bool):
                    is_premium_news = pf["isPremiumNews"]
                if isinstance(pf.get("isPremiumFreeNews"), bool):
                    is_premium_free = pf["isPremiumFreeNews"]
        return InstrumentTickerNewsItem(
            title=title.strip(),
            link=link,
            publisher=publisher,
            published_at=published_at,
            story_id=story_id,
            content_type=content_type,
            summary=summary,
            description=description,
            provider_url=provider_url,
            provider_source_id=provider_source_id,
            canonical_site=site,
            canonical_region=region,
            canonical_lang=lang,
            is_hosted=is_hosted,
            thumbnail_url=thumb_url,
            editors_pick=editors_pick,
            is_premium_news=is_premium_news,
            is_premium_free_news=is_premium_free,
        )
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    link = item.get("link")
    pub = item.get("publisher")
    published_at = _published_at_from_raw(item.get("pubDate"), item.get("providerPublishTime"))
    return InstrumentTickerNewsItem(
        title=title.strip(),
        link=link if isinstance(link, str) else None,
        publisher=pub if isinstance(pub, str) else None,
        published_at=published_at,
    )


def _run_ticker_news(symbol: str, limit: int) -> InstrumentTickerNewsResponse:
    try:
        adapter = YahooPublicAdapter()
        t = adapter.ticker(symbol)
        raw_list = t.news or []
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance ticker news failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail="news provider unavailable") from exc

    out: list[InstrumentTickerNewsItem] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        row = _news_item_from_ticker_news_dict(item)
        if row:
            out.append(row)
        if len(out) >= limit:
            break
    return InstrumentTickerNewsResponse(symbol=symbol, news=out)


def _nav_from_raw(item: dict[str, Any]) -> InstrumentNavLink | None:
    title = item.get("title")
    url = item.get("url") or item.get("href")
    if isinstance(title, str) and isinstance(url, str) and title.strip() and url.strip():
        return InstrumentNavLink(title=title.strip(), url=url.strip())
    return None


def _run_search(q: str) -> InstrumentSearchResponse:
    try:
        adapter = YahooPublicAdapter()
        s = adapter.search_instruments(q)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance search failed: %s", exc)
        raise HTTPException(status_code=502, detail="search provider unavailable") from exc

    quotes: list[InstrumentSearchQuote] = []
    seen: set[str] = set()
    for item in s.quotes or []:
        if not isinstance(item, dict):
            continue
        row = _quote_from_raw_safe(item)
        if row and row.symbol not in seen:
            seen.add(row.symbol)
            quotes.append(row)

    news: list[InstrumentSearchNewsItem] = []
    for item in s.news or []:
        if not isinstance(item, dict):
            continue
        row = _news_from_raw(item)
        if row:
            news.append(row)

    nav_links: list[InstrumentNavLink] = []
    raw_nav = s.nav
    if isinstance(raw_nav, list):
        for item in raw_nav:
            if isinstance(item, dict):
                nl = _nav_from_raw(item)
                if nl:
                    nav_links.append(nl)

    return InstrumentSearchResponse(quotes=quotes, news=news, nav_links=nav_links)


def _deferred_replace_task(pw: PendingOhlcvWriteback, run_id: int) -> None:
    replace_ohlcv_range_sync(
        pw.dsn,
        symbol=pw.symbol,
        interval_key=pw.interval_key,
        source=pw.source,
        start_inclusive=pw.start_inclusive,
        end_exclusive=pw.end_exclusive,
        ohlcv_df=pw.ohlcv_df,
        ingestion_run_id=run_id,
    )


@router.get("/search", response_model=InstrumentSearchResponse)
async def get_instrument_search(
    q: str = Query(..., min_length=1, max_length=MAX_SEARCH_LEN),
) -> InstrumentSearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    return await asyncio.to_thread(_run_search, query)


@router.get("/ingestion-runs/{run_id}", response_model=IngestionRunOut)
async def get_ingestion_run(run_id: int) -> IngestionRunOut:
    if run_id < 1:
        raise HTTPException(status_code=400, detail="invalid run id")
    try:
        from shunya.data.timescale.dbutil import get_database_url

        dsn = get_database_url()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="database not configured") from exc

    row = await asyncio.to_thread(get_ingestion_run_sync, dsn, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="ingestion run not found")
    return IngestionRunOut(**row)


@router.get("/{symbol}/overview", response_model=InstrumentOverviewResponse)
async def get_instrument_overview(symbol: str) -> InstrumentOverviewResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_overview, sym)


@router.get("/{symbol}/financials", response_model=InstrumentFinancialStatementResponse)
async def get_instrument_financials(
    symbol: str,
    statement: InstrumentStatementLiteral = Query(..., description="income, balance, or cashflow"),
    frequency: InstrumentFinancialFrequencyLiteral = Query("quarterly"),
    periods: int = Query(8, ge=1, le=8),
) -> InstrumentFinancialStatementResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(
        fetch_instrument_financials,
        sym,
        statement=statement,
        frequency=frequency,
        periods=periods,
    )


@router.get("/{symbol}/holders", response_model=InstrumentHoldersResponse)
async def get_instrument_holders(symbol: str) -> InstrumentHoldersResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_holders, sym)


@router.get("/{symbol}/options/expirations", response_model=InstrumentOptionExpirationsResponse)
async def get_instrument_option_expirations(symbol: str) -> InstrumentOptionExpirationsResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_option_expirations, sym)


@router.get("/{symbol}/options/chain", response_model=InstrumentOptionChainResponse)
async def get_instrument_option_chain(
    symbol: str,
    expiry: str = Query(..., min_length=10, max_length=10, description="Expiry YYYY-MM-DD"),
) -> InstrumentOptionChainResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_option_chain, sym, expiry)


@router.get("/{symbol}/options/iv-heatmap", response_model=InstrumentIvHeatmapResponse)
async def get_instrument_option_iv_heatmap(
    symbol: str,
    max_expirations: int = Query(24, ge=1, le=40),
) -> InstrumentIvHeatmapResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_option_iv_heatmap, sym, max_expirations)


@router.get("/{symbol}/valuation-measures", response_model=InstrumentValuationMeasuresPayload)
async def get_instrument_valuation_measures(symbol: str) -> InstrumentValuationMeasuresPayload:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_valuation_measures, sym)


@router.get("/{symbol}/analyst/price-targets", response_model=InstrumentAnalystPriceTargetsResponse)
async def get_instrument_analyst_price_targets(symbol: str) -> InstrumentAnalystPriceTargetsResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_analyst_price_targets, sym)


@router.get("/{symbol}/analyst/earnings-estimate", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_earnings_estimate(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_earnings_estimate, sym)


@router.get("/{symbol}/analyst/revenue-estimate", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_revenue_estimate(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_revenue_estimate, sym)


@router.get("/{symbol}/analyst/earnings-history", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_earnings_history(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_earnings_history, sym)


@router.get("/{symbol}/analyst/eps-trend", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_eps_trend(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_eps_trend, sym)


@router.get("/{symbol}/analyst/eps-revisions", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_eps_revisions(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_eps_revisions, sym)


@router.get("/{symbol}/analyst/growth-estimates", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_growth_estimates(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_growth_estimates, sym)


@router.get("/{symbol}/analyst/recommendations", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_recommendations(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_recommendations, sym)


@router.get("/{symbol}/analyst/recommendations-summary", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_recommendations_summary(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_recommendations_summary, sym)


@router.get("/{symbol}/analyst/upgrades-downgrades", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_upgrades_downgrades(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_upgrades_downgrades, sym)


@router.get("/{symbol}/sustainability", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_sustainability(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_sustainability, sym)


@router.get("/{symbol}/insider/purchases", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_insider_purchases(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_insider_purchases, sym)


@router.get("/{symbol}/insider/transactions", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_insider_transactions(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_insider_transactions, sym)


@router.get("/{symbol}/insider/roster-holders", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_insider_roster_holders(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_insider_roster_holders, sym)


@router.get("/{symbol}/insider/major-holders", response_model=InstrumentYfinanceTableResponse)
async def get_instrument_major_holders(symbol: str) -> InstrumentYfinanceTableResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_major_holders, sym)


@router.get("/{symbol}/calendar", response_model=InstrumentJsonBlobResponse)
async def get_instrument_calendar(symbol: str) -> InstrumentJsonBlobResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_calendar, sym)


@router.get("/{symbol}/sec-filings", response_model=InstrumentJsonBlobResponse)
async def get_instrument_sec_filings(symbol: str) -> InstrumentJsonBlobResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(fetch_instrument_sec_filings, sym)


@router.get("/{symbol}/news", response_model=InstrumentTickerNewsResponse)
async def get_instrument_news(
    symbol: str,
    limit: int = Query(40, ge=1, le=100),
) -> InstrumentTickerNewsResponse:
    sym = _normalize_symbol(symbol)
    return await asyncio.to_thread(_run_ticker_news, sym, limit)


@router.get("/{symbol}/ohlcv", response_model=InstrumentOhlcvResponse)
async def get_instrument_ohlcv(
    symbol: str,
    background_tasks: BackgroundTasks,
    interval: Literal[
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    ] = Query("1d"),
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = Query(
        "1y"
    ),
    defer_storage: bool = Query(False, description="If true, queue Timescale writeback and poll ingestion-runs"),
    route: str = Query(
        "auto",
        description="Market route: auto, best_effort, or explicit upstream (yfinance, alpaca_sip, …).",
    ),
) -> InstrumentOhlcvResponse:
    sym = _normalize_symbol(symbol)
    if interval not in ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail="invalid interval")
    if period not in ALLOWED_PERIODS:
        raise HTTPException(status_code=400, detail="invalid period")

    result = await asyncio.to_thread(
        resolve_instrument_ohlcv_sync, sym, interval, period, defer_storage=defer_storage, route=route
    )

    if result.pending_deferred_writeback is not None:
        pw = result.pending_deferred_writeback
        params = {
            "symbol": pw.symbol,
            "interval": pw.interval_key,
            "start": pw.start_inclusive.isoformat(),
            "end_exclusive": pw.end_exclusive.isoformat(),
        }
        run_id = await asyncio.to_thread(
            create_deferred_ingestion_run_sync,
            pw.dsn,
            source=pw.source,
            job="api_ohlcv_replace",
            params=params,
        )
        background_tasks.add_task(_deferred_replace_task, pw, run_id)
        return result.response.model_copy(update={"storage_job_id": run_id})

    return result.response
