"""Market overview: batched OHLCV snapshot, movers, headlines."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from api.schemas.models import (
    MarketHeadlinesResponse,
    MarketMoversResponse,
    MarketSnapshotRequest,
    MarketSnapshotResponse,
    MoversKindLiteral,
)
from api.services.market_exceptions import MarketProviderError
from api.services.market_headlines import fetch_market_headlines
from api.services.market_movers import fetch_movers
from api.services.market_snapshot import build_snapshot
from api.services.market_symbols import normalize_market_symbol

router = APIRouter(prefix="/market", tags=["market"])


def _normalize_symbol_list(raw_symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_symbols:
        try:
            sym = normalize_market_symbol(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid symbol") from exc
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


@router.post("/snapshot", response_model=MarketSnapshotResponse)
async def post_market_snapshot(body: MarketSnapshotRequest) -> MarketSnapshotResponse:
    symbols = _normalize_symbol_list(body.symbols)
    try:
        rows = await asyncio.to_thread(build_snapshot, symbols)
    except MarketProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "market data unavailable") from exc
    return MarketSnapshotResponse(rows=rows)


@router.get("/movers", response_model=MarketMoversResponse)
async def get_market_movers(
    kind: Annotated[MoversKindLiteral, Query(description="Screener segment")],
    limit: Annotated[int, Query(ge=1, le=250)] = 25,
) -> MarketMoversResponse:
    try:
        rows = await asyncio.to_thread(fetch_movers, kind, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid movers kind") from exc
    except MarketProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "screener unavailable") from exc
    return MarketMoversResponse(kind=kind, rows=rows)


@router.get("/headlines", response_model=MarketHeadlinesResponse)
async def get_market_headlines(
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> MarketHeadlinesResponse:
    try:
        headlines = await asyncio.to_thread(fetch_market_headlines, limit)
    except MarketProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc) or "news unavailable") from exc
    return MarketHeadlinesResponse(headlines=headlines)
