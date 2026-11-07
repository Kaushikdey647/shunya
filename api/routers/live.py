"""Hosted live-runner API: status, books/sleeves, strategy versions, kill / resume.

Every route requires ``X-Shunya-Live-Token`` matching ``SHUNYA_LIVE_TOKEN``.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import asdict
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from shunya.live.control import engage_kill, resume
from shunya.live.signals import UnsupportedForLive, live_strategy_warnings, validate_live_strategy
from shunya.live.status import build_status

LIVE_TOKEN_HEADER = "X-Shunya-Live-Token"


def require_live_token(
    x_shunya_live_token: Annotated[Optional[str], Header(alias=LIVE_TOKEN_HEADER)] = None,
) -> None:
    """503 when the server has no token configured; 401 on missing / wrong token."""
    expected = os.environ.get("SHUNYA_LIVE_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="SHUNYA_LIVE_TOKEN is not configured on the server.")
    if not x_shunya_live_token or not hmac.compare_digest(x_shunya_live_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing live token.")


router = APIRouter(prefix="/live", tags=["live"], dependencies=[Depends(require_live_token)])


def _store(request: Request) -> Any:
    return request.app.state.live_store


def _broker(request: Request) -> Any:
    return request.app.state.live_broker


class KillIn(BaseModel):
    """Kill-switch request."""

    level: Literal["halt", "cancel", "flatten"] = "cancel"
    reason: str = Field(min_length=1, max_length=500)
    actor: str = Field(default="dashboard", max_length=100)


class ResumeIn(BaseModel):
    """Re-arm request."""

    reason: str = Field(min_length=1, max_length=500)
    actor: str = Field(default="dashboard", max_length=100)


class StrategyVersionIn(BaseModel):
    """Frozen alpha snapshot pushed from the local research DB."""

    name: str = Field(min_length=1, max_length=200)
    alpha_source: str = Field(min_length=1)
    finstrat_config: Dict[str, Any] = Field(default_factory=dict)
    tickers: List[str] = Field(min_length=1, max_length=3000)
    lookback_days: int = Field(default=400, ge=30, le=3650)


class BookIn(BaseModel):
    """New book."""

    name: str = Field(min_length=1, max_length=200)
    capital_fraction: float = Field(default=1.0, gt=0, le=1)
    risk_config: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["active", "inactive"] = "active"


class BookPatch(BaseModel):
    """Book update."""

    name: Optional[str] = None
    capital_fraction: Optional[float] = Field(default=None, gt=0, le=1)
    risk_config: Optional[Dict[str, Any]] = None
    status: Optional[Literal["active", "inactive"]] = None


class SleeveIn(BaseModel):
    """Attach a strategy version to a book."""

    version_id: str
    weight: float = Field(ge=0, le=1)
    ramp_days: int = Field(default=1, ge=1, le=60)


class SleevePatch(BaseModel):
    """Sleeve update."""

    weight: Optional[float] = Field(default=None, ge=0, le=1)
    status: Optional[Literal["active", "paused", "draining", "retired"]] = None
    ramp_days: Optional[int] = Field(default=None, ge=1, le=60)


def _check_weights(store: Any, book_id: str, *, sleeve_id: Optional[str], new_weight: float) -> None:
    others = sum(s.weight for s in store.list_sleeves(book_id) if s.id != sleeve_id and s.status != "draining")
    if others + new_weight > 1.0 + 1e-9:
        raise HTTPException(status_code=422, detail=f"sleeve weights would sum to {others + new_weight:.3f} > 1")


@router.get("/status")
def live_status(request: Request) -> Dict[str, Any]:
    """Kill state, heartbeat, Alpaca clock/account/positions/orders, books, ledger drift."""
    return build_status(_store(request), _broker(request))


@router.post("/kill")
def live_kill(body: KillIn, request: Request) -> Dict[str, Any]:
    """Set the sticky halt flag, then cancel orders / flatten per ``level``."""
    return engage_kill(_store(request), _broker(request), level=body.level, reason=body.reason, actor=body.actor)


@router.post("/resume")
def live_resume(body: ResumeIn, request: Request) -> Dict[str, Any]:
    """Re-arm trading."""
    resume(_store(request), reason=body.reason, actor=body.actor)
    return {"state": "armed"}


@router.get("/audit")
def live_audit(request: Request, limit: int = Query(50, ge=1, le=500)) -> List[Dict[str, Any]]:
    """Kill / resume / auto-halt history."""
    return _store(request).list_audit(limit)


@router.get("/strategy-versions")
def list_versions(request: Request) -> List[Dict[str, Any]]:
    """Deployed snapshots (metadata only)."""
    return _store(request).list_versions()


@router.post("/strategy-versions", status_code=201)
def create_version(body: StrategyVersionIn, request: Request) -> Dict[str, Any]:
    """Validate (compiles alpha, live-supported config) and store a snapshot."""
    try:
        cfg = validate_live_strategy(body.alpha_source, body.finstrat_config)
    except (UnsupportedForLive, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    vid = _store(request).create_version(
        name=body.name,
        alpha_source=body.alpha_source,
        finstrat_config=cfg,
        tickers=sorted({t.strip().upper() for t in body.tickers if t.strip()}),
        lookback_days=body.lookback_days,
    )
    return {"id": vid, "warnings": live_strategy_warnings(body.alpha_source)}


@router.get("/books")
def list_books(request: Request) -> List[Dict[str, Any]]:
    """Books with their sleeves."""
    store = _store(request)
    out = []
    for b in store.list_books():
        sleeves = [
            {**{k: v for k, v in asdict(s).items() if k != "version"}, "version_id": s.version.id, "name": s.version.name}
            for s in store.list_sleeves(b.id)
        ]
        out.append({**asdict(b), "sleeves": sleeves})
    return out


@router.post("/books", status_code=201)
def create_book(body: BookIn, request: Request) -> Dict[str, str]:
    """Create a book (v1: at most one active)."""
    store = _store(request)
    if body.status == "active" and store.list_books(active_only=True):
        raise HTTPException(status_code=409, detail="an active book already exists (one per account in v1)")
    return {"id": store.create_book(**body.model_dump())}


@router.patch("/books/{book_id}")
def patch_book(book_id: str, body: BookPatch, request: Request) -> Dict[str, Any]:
    """Update book fields."""
    store = _store(request)
    if store.get_book(book_id) is None:
        raise HTTPException(status_code=404, detail="book not found")
    store.update_book(book_id, **body.model_dump(exclude_unset=True))
    return asdict(store.get_book(book_id))


@router.post("/books/{book_id}/sleeves", status_code=201)
def create_sleeve(book_id: str, body: SleeveIn, request: Request) -> Dict[str, str]:
    """Attach a strategy version to a book."""
    store = _store(request)
    if store.get_book(book_id) is None:
        raise HTTPException(status_code=404, detail="book not found")
    _check_weights(store, book_id, sleeve_id=None, new_weight=body.weight)
    sid = store.create_sleeve(book_id=book_id, version_id=body.version_id, weight=body.weight, ramp_days=body.ramp_days)
    store.audit("dashboard", "sleeve_create", {"book_id": book_id, **body.model_dump()})
    return {"id": sid}


@router.patch("/books/{book_id}/sleeves/{sleeve_id}")
def patch_sleeve(book_id: str, sleeve_id: str, body: SleevePatch, request: Request) -> Dict[str, Any]:
    """Change weight / status (pause, drain, retire) / ramp."""
    store = _store(request)
    current = next((s for s in store.list_sleeves(book_id) if s.id == sleeve_id), None)
    if current is None:
        raise HTTPException(status_code=404, detail="sleeve not found")
    if body.weight is not None:
        _check_weights(store, book_id, sleeve_id=sleeve_id, new_weight=body.weight)
    patch = body.model_dump(exclude_unset=True)
    store.update_sleeve(sleeve_id, **patch)
    store.audit("dashboard", "sleeve_update", {"sleeve_id": sleeve_id, **patch})
    return {"id": sleeve_id, **patch}


@router.get("/runs")
def list_runs(
    request: Request, book_id: Optional[str] = None, limit: int = Query(30, ge=1, le=200)
) -> List[Dict[str, Any]]:
    """Recent runs, newest first."""
    return [asdict(r) for r in _store(request).list_runs(book_id=book_id, limit=limit)]


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> Dict[str, Any]:
    """One run with sleeve targets and orders."""
    store = _store(request)
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        **asdict(run),
        "targets": store.list_sleeve_targets(run_id),
        "orders": [asdict(o) for o in store.list_orders(run_id)],
    }
