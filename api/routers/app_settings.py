"""HTTP surface for merged app tunables (DB overlay + env)."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel, Field

from api.db import resolve_database_url
from api.repositories import runtime_config as runtime_repo
from api.services.market_clock_hub import get_market_clock_hub, snapshot_to_tick_payload
from api.services.notify_background import schedule_notification
from api.settings import get_settings
from api.tunable_config import (
    TUNABLE_KEYS,
    RuntimeOverlayPatch,
    clear_tunables_cache,
    get_effective_tunables,
    merge_overlay_patch,
    tunable_sources,
)
from shunya.time.market_clock import build_market_clock_snapshot

router = APIRouter(prefix="/settings", tags=["settings"])
_log = logging.getLogger(__name__)


class AppSettingsEnvironmentOut(BaseModel):
    """Non-secret deployment flags (no values for tokens/URLs)."""

    database_configured: bool = Field(description="DATABASE_URL or SHUNYA_API_DATABASE_URL is set")
    alpaca_enabled: bool
    ollama_host_configured: bool
    trade_desk_write_configured: bool = Field(
        description="PATCH allowed when SHUNYA_API_TRADE_DESK_TOKEN is set"
    )


class AppSettingsResponse(BaseModel):
    environment: AppSettingsEnvironmentOut
    runtime: dict[str, Any]
    sources: dict[str, str]


class MarketClockResponse(BaseModel):
    """Server wall clocks and US RTH gate for Alpaca L1 (see :mod:`shunya.time.market_clock`)."""

    utc_iso: str = Field(description="Current instant as ISO-8601 UTC (Z suffix).")
    us_line: str = Field(description="``[US] DD-MM HH:MM:SS.mmm`` in America/New_York.")
    in_line: str = Field(description="``[IN] DD-MM HH:MM:SS.mmm`` in Asia/Kolkata.")
    us_listed_rth_open: bool = Field(
        description="True during weekday US listed RTH 09:30–16:00 ET (holidays not excluded)."
    )
    alpaca_l1_us_equities_stream_allowed: bool = Field(
        description="Same as RTH unless SHUNYA_ALPACA_L1_IGNORE_US_RTH is set on the API process."
    )


def require_trade_desk_token_for_settings(
    x_token: Annotated[Optional[str], Header(alias="X-Shunya-Trade-Desk-Token")] = None,
) -> None:
    settings = get_settings()
    expected = settings.trade_desk_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Runtime settings write is disabled: set SHUNYA_API_TRADE_DESK_TOKEN on the API process.",
        )
    if not x_token or x_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Shunya-Trade-Desk-Token header.")


def _environment_out() -> AppSettingsEnvironmentOut:
    import os

    s = get_settings()
    db_ok = bool(
        s.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("SHUNYA_DATABASE_URL")
    )
    host = (s.ollama_host or "").strip()
    return AppSettingsEnvironmentOut(
        database_configured=db_ok,
        alpaca_enabled=bool(s.alpaca_enabled),
        ollama_host_configured=bool(host),
        trade_desk_write_configured=bool(s.trade_desk_token),
    )


@router.get("/app", response_model=AppSettingsResponse)
def get_app_settings() -> AppSettingsResponse:
    """Effective tunables plus read-only environment flags (no secrets)."""
    eff = get_effective_tunables()
    return AppSettingsResponse(
        environment=_environment_out(),
        runtime=eff.as_public_dict(),
        sources=tunable_sources(),
    )


@router.get("/market-clock", response_model=MarketClockResponse)
def get_market_clock() -> MarketClockResponse:
    """US / India wall-clock strings and US RTH gate (point-in-time; prefer WebSocket for live updates)."""
    snap = build_market_clock_snapshot()
    return MarketClockResponse(
        utc_iso=snap.utc_iso,
        us_line=snap.us_line,
        in_line=snap.in_line,
        us_listed_rth_open=snap.us_listed_rth_open,
        alpaca_l1_us_equities_stream_allowed=snap.alpaca_l1_us_equities_stream_allowed,
    )


async def _wait_market_clock_ws_disconnect(websocket: WebSocket) -> None:
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return
    except WebSocketDisconnect:
        return


@router.websocket("/market-clock/stream")
async def market_clock_stream_ws(websocket: WebSocket) -> None:
    """Server-pushed clock ticks (``hello`` then repeating ``tick`` frames, same fields as GET)."""
    await websocket.accept()
    hub = await get_market_clock_hub()
    try:
        await hub.attach_websocket(websocket)
    except Exception as exc:  # noqa: BLE001
        _log.warning("market clock stream attach failed: %s", exc)
        await websocket.close(code=1011)
        return

    try:
        await websocket.send_json({"type": "hello", "schema": 1})
        snap0 = build_market_clock_snapshot()
        await websocket.send_json(snapshot_to_tick_payload(snap0))
        await _wait_market_clock_ws_disconnect(websocket)
    finally:
        await hub.detach_websocket(websocket)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@router.patch("/app", response_model=AppSettingsResponse, dependencies=[Depends(require_trade_desk_token_for_settings)])
def patch_app_settings(body: RuntimeOverlayPatch, background_tasks: BackgroundTasks) -> AppSettingsResponse:
    """Merge tunables into ``api_runtime_config`` (requires trade-desk token)."""
    try:
        resolve_database_url()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    existing = runtime_repo.fetch_runtime_payload()
    if existing is None:
        raise HTTPException(
            status_code=503,
            detail="Could not read runtime settings from the database (table missing or DB down).",
        )

    merged = merge_overlay_patch(existing, body)
    clean = {k: merged[k] for k in TUNABLE_KEYS if k in merged}
    try:
        runtime_repo.save_runtime_payload(clean)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Failed to save settings: {exc!s}") from exc
    clear_tunables_cache()
    eff = get_effective_tunables(force_refresh=True)
    keys = list(body.model_dump(exclude_unset=True).keys())
    nk = len(keys)
    schedule_notification(
        background_tasks,
        level="info",
        title="App settings updated",
        message=f"Runtime tunables saved ({nk} key(s)).",
        code="settings.app_patched",
        context={"patch_keys": ",".join(keys)[:240]},
    )
    return AppSettingsResponse(
        environment=_environment_out(),
        runtime=eff.as_public_dict(),
        sources=tunable_sources(),
    )
