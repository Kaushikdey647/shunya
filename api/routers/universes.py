"""User-defined equity universes (CRUD, membership, summary analytics)."""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, status

from api.repositories import universes as universes_repo
from api.routers.instruments import ALLOWED_PERIODS
from api.schemas.models import (
    UniverseCreate,
    UniverseMembersAddRequest,
    UniverseMembersMutationOut,
    UniverseMembersReplaceRequest,
    UniverseMemberOut,
    UniverseOut,
    UniversePatch,
    UniverseReturnAnalyticsOut,
    UniverseSummaryOut,
    UniverseTickerListOut,
)
from api.services.universe_return_analytics import compute_universe_return_analytics
from api.services.notify_background import schedule_notification
from shunya.errors import ErrorCode, ShunyaError

router = APIRouter(prefix="/universes", tags=["universes"])


def _require_uuid(universe_id: str) -> str:
    try:
        return str(UUID(universe_id.strip()))
    except ValueError as exc:
        raise ShunyaError("Invalid universe id.", code=ErrorCode.VALIDATION_ERROR, http_status=400) from exc


@router.post("", response_model=UniverseOut, status_code=status.HTTP_201_CREATED)
def create_universe(body: UniverseCreate, background_tasks: BackgroundTasks) -> UniverseOut:
    try:
        out = universes_repo.insert_universe(body)
    except RuntimeError as exc:
        if str(exc) == "duplicate_universe_name":
            raise ShunyaError(
                "Universe name already exists.",
                code=ErrorCode.UNIVERSE_NAME_CONFLICT,
                http_status=409,
            ) from exc
        raise
    schedule_notification(
        background_tasks,
        level="info",
        title="Universe created",
        message=f'Universe "{out.name}" created.',
        code="universe.created",
        context={"universe_id": out.id},
    )
    return out


@router.get("", response_model=list[UniverseOut])
def list_universes(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[UniverseOut]:
    return universes_repo.list_universes(limit=limit, offset=offset)


@router.get("/{universe_id}", response_model=UniverseOut)
def get_universe(universe_id: str) -> UniverseOut:
    uid = _require_uuid(universe_id)
    row = universes_repo.get_universe(uid)
    if row is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    return row


@router.patch("/{universe_id}", response_model=UniverseOut)
def patch_universe(universe_id: str, body: UniversePatch, background_tasks: BackgroundTasks) -> UniverseOut:
    uid = _require_uuid(universe_id)
    try:
        row = universes_repo.update_universe(uid, body)
    except RuntimeError as exc:
        if str(exc) == "duplicate_universe_name":
            raise ShunyaError(
                "Universe name already exists.",
                code=ErrorCode.UNIVERSE_NAME_CONFLICT,
                http_status=409,
            ) from exc
        raise
    if row is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    schedule_notification(
        background_tasks,
        level="info",
        title="Universe updated",
        message=f'Universe "{row.name}" updated.',
        code="universe.updated",
        context={"universe_id": uid},
    )
    return row


@router.delete("/{universe_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_universe(universe_id: str, background_tasks: BackgroundTasks) -> None:
    uid = _require_uuid(universe_id)
    row = universes_repo.get_universe(uid)
    if row is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    if not universes_repo.delete_universe(uid):
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    schedule_notification(
        background_tasks,
        level="info",
        title="Universe deleted",
        message=f'Universe "{row.name}" deleted.',
        code="universe.deleted",
        context={"universe_id": uid},
    )


@router.get("/{universe_id}/tickers", response_model=UniverseTickerListOut)
def list_universe_tickers(universe_id: str) -> UniverseTickerListOut:
    uid = _require_uuid(universe_id)
    if universes_repo.get_universe(uid) is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    return UniverseTickerListOut(tickers=universes_repo.constituent_tickers(uid))


@router.get("/{universe_id}/members", response_model=list[UniverseMemberOut])
def list_universe_members(
    universe_id: str,
    limit: int = Query(default=200, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> list[UniverseMemberOut]:
    uid = _require_uuid(universe_id)
    if universes_repo.get_universe(uid) is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    rows, _ = universes_repo.list_members(uid, limit=limit, offset=offset)
    return rows


@router.post("/{universe_id}/members", response_model=UniverseMembersMutationOut)
def add_universe_members(
    universe_id: str, body: UniverseMembersAddRequest, background_tasks: BackgroundTasks
) -> UniverseMembersMutationOut:
    uid = _require_uuid(universe_id)
    if universes_repo.get_universe(uid) is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    try:
        n = universes_repo.add_members_strict(uid, body.tickers)
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("unknown_tickers:"):
            raise ShunyaError(
                "One or more tickers are not in the symbols table.",
                code=ErrorCode.UNIVERSE_UNKNOWN_TICKER,
                http_status=400,
                context={"detail": msg},
            ) from exc
        if msg.startswith("non_equity_ticker:"):
            raise ShunyaError(
                "Only equity names may be added (latest quote_type must be EQUITY or unknown).",
                code=ErrorCode.UNIVERSE_MEMBER_NOT_EQUITY,
                http_status=400,
                context={"detail": msg},
            ) from exc
        raise ShunyaError(msg, code=ErrorCode.VALIDATION_ERROR, http_status=400) from exc
    fresh = universes_repo.get_universe(uid)
    mc = fresh.member_count if fresh else 0
    out = UniverseMembersMutationOut(changed=n, member_count=mc)
    schedule_notification(
        background_tasks,
        level="info",
        title="Universe members added",
        message=f"Added {n} member(s) to universe {uid}.",
        code="universe.members_added",
        context={"universe_id": uid, "changed": n},
    )
    return out


@router.delete("/{universe_id}/members", response_model=UniverseMembersMutationOut)
def remove_universe_members(
    universe_id: str,
    tickers: Annotated[list[str], Query(description="Repeat tickers= for each symbol to remove.")],
    background_tasks: BackgroundTasks,
) -> UniverseMembersMutationOut:
    uid = _require_uuid(universe_id)
    if universes_repo.get_universe(uid) is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    norm = [str(t).strip().upper() for t in tickers if str(t).strip()]
    n = universes_repo.remove_members(uid, norm)
    fresh = universes_repo.get_universe(uid)
    mc = fresh.member_count if fresh else 0
    out = UniverseMembersMutationOut(changed=n, member_count=mc)
    schedule_notification(
        background_tasks,
        level="info",
        title="Universe members removed",
        message=f"Removed {n} member(s) from universe {uid}.",
        code="universe.members_removed",
        context={"universe_id": uid, "changed": n},
    )
    return out


@router.put("/{universe_id}/members", response_model=UniverseMembersMutationOut)
def replace_universe_members(
    universe_id: str, body: UniverseMembersReplaceRequest, background_tasks: BackgroundTasks
) -> UniverseMembersMutationOut:
    uid = _require_uuid(universe_id)
    if universes_repo.get_universe(uid) is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    try:
        universes_repo.replace_members(uid, body.tickers)
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("unknown_tickers:"):
            raise ShunyaError(
                "One or more tickers are not in the symbols table.",
                code=ErrorCode.UNIVERSE_UNKNOWN_TICKER,
                http_status=400,
                context={"detail": msg},
            ) from exc
        if msg.startswith("non_equity_ticker:"):
            raise ShunyaError(
                "Only equity names may be members.",
                code=ErrorCode.UNIVERSE_MEMBER_NOT_EQUITY,
                http_status=400,
                context={"detail": msg},
            ) from exc
        raise ShunyaError(msg, code=ErrorCode.VALIDATION_ERROR, http_status=400) from exc
    fresh = universes_repo.get_universe(uid)
    mc = fresh.member_count if fresh else 0
    out = UniverseMembersMutationOut(changed=len(body.tickers), member_count=mc)
    schedule_notification(
        background_tasks,
        level="info",
        title="Universe members replaced",
        message=f"Replaced members for universe {uid} ({len(body.tickers)} tickers).",
        code="universe.members_replaced",
        context={"universe_id": uid, "count": len(body.tickers)},
    )
    return out


@router.get("/{universe_id}/return-analytics", response_model=UniverseReturnAnalyticsOut)
async def get_universe_return_analytics(
    universe_id: str,
    period: str = Query("1y", description="Lookback (same values as instrument OHLCV ``period``)."),
    interval: str = Query("1d", description="Only ``1d`` is supported."),
    source: str = Query("yfinance", min_length=1, max_length=64),
    max_members: int = Query(500, ge=2, le=5000),
    n_pca_components: int = Query(5, ge=1, le=15),
) -> UniverseReturnAnalyticsOut:
    """Correlations, cross-sectional volatility, PCA, and cap-weight concentration from Timescale OHLCV."""
    uid = _require_uuid(universe_id)
    if period not in ALLOWED_PERIODS:
        raise ShunyaError(
            "Invalid period.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
            context={"allowed": sorted(ALLOWED_PERIODS)},
        )
    return await asyncio.to_thread(
        compute_universe_return_analytics,
        uid,
        period=period,
        interval=interval,
        source=source,
        max_members=max_members,
        n_pca_components=n_pca_components,
    )


@router.get("/{universe_id}/summary", response_model=UniverseSummaryOut)
def universe_summary(universe_id: str) -> UniverseSummaryOut:
    uid = _require_uuid(universe_id)
    row = universes_repo.universe_summary(uid)
    if row is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)
    return row
