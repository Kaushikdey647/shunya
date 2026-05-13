"""HTTP surface for merged app tunables (DB overlay + env)."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from api.db import resolve_database_url
from api.repositories import runtime_config as runtime_repo
from api.settings import get_settings
from api.tunable_config import (
    TUNABLE_KEYS,
    RuntimeOverlayPatch,
    clear_tunables_cache,
    get_effective_tunables,
    merge_overlay_patch,
    tunable_sources,
)

router = APIRouter(prefix="/settings", tags=["settings"])


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


@router.patch("/app", response_model=AppSettingsResponse, dependencies=[Depends(require_trade_desk_token_for_settings)])
def patch_app_settings(body: RuntimeOverlayPatch) -> AppSettingsResponse:
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
    return AppSettingsResponse(
        environment=_environment_out(),
        runtime=eff.as_public_dict(),
        sources=tunable_sources(),
    )
