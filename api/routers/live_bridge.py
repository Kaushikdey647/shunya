"""Local bridge to the hosted live runner (``SHUNYA_LIVE_URL`` + ``SHUNYA_LIVE_TOKEN``).

The browser only talks to the local API; the hosted token stays server-side. ``POST /live/kill``
falls back to cancelling / flattening directly at Alpaca with the local keys when the hosted
service is unreachable, so a dead runner can never block a kill.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api.repositories import alphas as alphas_repo
from api.repositories import universes as universes_repo
from api.routers.trade_desk import require_trade_desk_token
from shunya.live.signals import UnsupportedForLive, validate_live_strategy

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["live"], dependencies=[Depends(require_trade_desk_token)])

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class HostedUnavailable(RuntimeError):
    """Hosted runner not configured or not reachable."""


def _hosted(method: str, path: str, *, json: Any = None, params: Optional[Dict[str, Any]] = None) -> Any:
    base = os.environ.get("SHUNYA_LIVE_URL", "").strip().rstrip("/")
    token = os.environ.get("SHUNYA_LIVE_TOKEN", "").strip()
    if not base or not token:
        raise HostedUnavailable("SHUNYA_LIVE_URL / SHUNYA_LIVE_TOKEN not set on the local API")
    try:
        r = httpx.request(
            method,
            f"{base}{path}",
            json=json,
            params={k: v for k, v in (params or {}).items() if v is not None},
            headers={"X-Shunya-Live-Token": token},
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        raise HostedUnavailable(f"hosted runner unreachable: {exc}") from exc
    if r.status_code >= 500:
        raise HostedUnavailable(f"hosted runner error {r.status_code}: {r.text[:300]}")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)
    return r.json()


def _proxy(method: str, path: str, **kw: Any) -> Any:
    try:
        return _hosted(method, path, **kw)
    except HostedUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _local_trading_client(request: Request) -> Any:
    rt = getattr(request.app.state, "trade_desk_runtime", None)
    return rt.trading_client if rt is not None else None


@router.get("/status")
def bridge_status(request: Request) -> Dict[str, Any]:
    """Hosted status; when the runner is down, a direct Alpaca view so the dashboard still shows the account."""
    try:
        return {"hosted_reachable": True, **_hosted("GET", "/live/status")}
    except HostedUnavailable as exc:
        out: Dict[str, Any] = {"hosted_reachable": False, "hosted_error": str(exc)}
        client = _local_trading_client(request)
        if client is not None:
            from shunya.live.broker import AlpacaBroker
            from shunya.live.store import Control

            b = AlpacaBroker(client, lambda: Control(state="halted", reason="read-only bridge"))
            try:
                out["account"] = asdict(b.account())
                out["positions"] = [asdict(p) for p in b.positions()]
                out["open_orders"] = [asdict(o) for o in b.open_orders()]
                out["clock"] = b.clock()
            except Exception as e:  # noqa: BLE001
                out["broker_error"] = str(e)
        return out


class KillIn(BaseModel):
    """Kill request from the dashboard."""

    level: Literal["halt", "cancel", "flatten"] = "cancel"
    reason: str = Field(min_length=1, max_length=500)


@router.post("/kill")
def bridge_kill(body: KillIn, request: Request) -> Dict[str, Any]:
    """Hosted kill first; on failure cancel / flatten directly at Alpaca with local keys."""
    payload = {**body.model_dump(), "actor": "dashboard"}
    try:
        return {"route": "hosted", **_hosted("POST", "/live/kill", json=payload)}
    except HostedUnavailable as exc:
        hosted_error = str(exc)
    _log.error("hosted kill failed (%s); falling back to direct Alpaca", hosted_error)
    out: Dict[str, Any] = {"route": "direct_alpaca", "hosted_error": hosted_error, "level": body.level}
    client = _local_trading_client(request)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=f"Hosted kill failed ({hosted_error}) and local Alpaca is disabled "
            "(SHUNYA_API_ALPACA_ENABLED). Cancel in the Alpaca dashboard now.",
        )
    if body.level == "halt":
        out["warning"] = "hosted flag not set; level 'halt' has no broker action — use 'cancel' or 'flatten'"
        return out
    try:
        out["orders_cancelled"] = len(client.cancel_orders() or [])
        if body.level == "flatten":
            out["positions_closed"] = len(client.close_all_positions(cancel_orders=True) or [])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"direct Alpaca kill failed: {e}") from e
    out["warning"] = "hosted halt flag NOT set; the runner may trade again once reachable — retry kill"
    return out


class ResumeIn(BaseModel):
    """Re-arm request."""

    reason: str = Field(min_length=1, max_length=500)


@router.post("/resume")
def bridge_resume(body: ResumeIn) -> Any:
    """Re-arm the hosted runner."""
    return _proxy("POST", "/live/resume", json={**body.model_dump(), "actor": "dashboard"})


class DeployIn(BaseModel):
    """Snapshot a local alpha + universe and push it to the hosted runner."""

    alpha_id: str
    universe_id: Optional[str] = None
    tickers: Optional[List[str]] = None
    name: Optional[str] = None
    lookback_days: int = Field(default=400, ge=30, le=3650)
    finstrat_config: Optional[Dict[str, Any]] = None
    book_id: Optional[str] = None
    weight: Optional[float] = Field(default=None, ge=0, le=1)
    ramp_days: int = Field(default=1, ge=1, le=60)


def _tradable(tickers: List[str]) -> List[str]:
    """Subset of ``tickers`` flagged ``is_alpaca_tradable`` in Timescale."""
    import psycopg

    from api.db import resolve_database_url
    from shunya.data.timescale.reference_sync import tradable_symbols

    with psycopg.connect(resolve_database_url()) as conn, conn.cursor() as cur:
        return tradable_symbols(cur, tickers)


@router.post("/deploy", status_code=201)
def bridge_deploy(body: DeployIn) -> Dict[str, Any]:
    """Freeze alpha source, FinStrat config and tickers; optionally attach as a sleeve to ``book_id``."""
    raw = alphas_repo.get_alpha_raw(body.alpha_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="alpha not found")
    if not (raw.get("source_code") or "").strip():
        raise HTTPException(status_code=422, detail="only inline-source alphas can be deployed (no import_ref)")
    if body.tickers:
        tickers = body.tickers
        bad = sorted(set(tickers) - set(_tradable(tickers)))
        if bad:
            raise HTTPException(status_code=422, detail=f"not Alpaca-tradable: {', '.join(bad[:20])}")
    elif body.universe_id:
        tickers = universes_repo.constituent_tickers(body.universe_id)
    else:
        raise HTTPException(status_code=422, detail="pass universe_id or tickers")
    if not tickers:
        raise HTTPException(status_code=422, detail="universe has no members")
    cfg = {**(raw.get("finstrat_config") or {}), **(body.finstrat_config or {})}
    try:
        cfg = validate_live_strategy(raw["source_code"], cfg)
    except (UnsupportedForLive, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    alpha = alphas_repo.get_alpha(body.alpha_id)
    name = body.name or (alpha.name if alpha is not None else body.alpha_id)
    version = _proxy(
        "POST",
        "/live/strategy-versions",
        json={
            "name": name,
            "alpha_source": raw["source_code"],
            "finstrat_config": cfg,
            "tickers": tickers,
            "lookback_days": body.lookback_days,
        },
    )
    out: Dict[str, Any] = {
        "version_id": version["id"],
        "n_tickers": len(tickers),
        "warnings": version.get("warnings", []),
    }
    if body.book_id and body.weight is not None:
        out["sleeve_id"] = _proxy(
            "POST",
            f"/live/books/{body.book_id}/sleeves",
            json={"version_id": version["id"], "weight": body.weight, "ramp_days": body.ramp_days},
        )["id"]
    return out


@router.get("/books")
def bridge_books() -> Any:
    """Hosted books with sleeves."""
    return _proxy("GET", "/live/books")


@router.post("/books", status_code=201)
def bridge_create_book(body: Dict[str, Any] = Body(...)) -> Any:
    """Create a hosted book."""
    return _proxy("POST", "/live/books", json=body)


@router.patch("/books/{book_id}")
def bridge_patch_book(book_id: str, body: Dict[str, Any] = Body(...)) -> Any:
    """Patch a hosted book."""
    return _proxy("PATCH", f"/live/books/{book_id}", json=body)


@router.patch("/books/{book_id}/sleeves/{sleeve_id}")
def bridge_patch_sleeve(book_id: str, sleeve_id: str, body: Dict[str, Any] = Body(...)) -> Any:
    """Change a sleeve's weight / status."""
    return _proxy("PATCH", f"/live/books/{book_id}/sleeves/{sleeve_id}", json=body)


@router.get("/strategy-versions")
def bridge_versions() -> Any:
    """Deployed snapshots."""
    return _proxy("GET", "/live/strategy-versions")


@router.get("/runs")
def bridge_runs(book_id: Optional[str] = None, limit: int = Query(30, ge=1, le=200)) -> Any:
    """Recent runs."""
    return _proxy("GET", "/live/runs", params={"book_id": book_id, "limit": limit})


@router.get("/runs/{run_id}")
def bridge_run(run_id: str) -> Any:
    """One run with targets and orders."""
    return _proxy("GET", f"/live/runs/{run_id}")


@router.get("/audit")
def bridge_audit(limit: int = Query(50, ge=1, le=500)) -> Any:
    """Kill / resume history."""
    return _proxy("GET", "/live/audit", params={"limit": limit})
