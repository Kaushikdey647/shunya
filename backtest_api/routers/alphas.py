from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from backtest_api.alpha_validation import validate_import_ref
from backtest_api.repositories import alphas as repo
from backtest_api.schemas.models import AlphaCreate, AlphaOut, AlphaPatch

router = APIRouter(prefix="/alphas", tags=["alphas"])


@router.post("", response_model=AlphaOut, status_code=status.HTTP_201_CREATED)
def create_alpha(body: AlphaCreate) -> AlphaOut:
    try:
        return repo.insert_alpha(body)
    except RuntimeError as exc:
        if str(exc) == "duplicate_alpha_name":
            raise HTTPException(status_code=409, detail="Alpha name already exists.") from exc
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=list[AlphaOut])
def list_alphas(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AlphaOut]:
    return repo.list_alphas(limit=limit, offset=offset)


@router.get("/{alpha_id}", response_model=AlphaOut)
def get_alpha(alpha_id: str) -> AlphaOut:
    row = repo.get_alpha(alpha_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Alpha not found.")
    return row


@router.patch("/{alpha_id}", response_model=AlphaOut)
def patch_alpha(alpha_id: str, body: AlphaPatch) -> AlphaOut:
    if body.import_ref:
        try:
            validate_import_ref(body.import_ref)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    row = repo.update_alpha(alpha_id, body)
    if row is None:
        raise HTTPException(status_code=404, detail="Alpha not found.")
    return row


@router.delete("/{alpha_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_alpha(alpha_id: str) -> None:
    try:
        ok = repo.delete_alpha(alpha_id)
    except RuntimeError as exc:
        if str(exc) == "foreign_key_violation":
            raise HTTPException(
                status_code=409,
                detail="Cannot delete alpha while backtest jobs reference it.",
            ) from exc
        raise
    if not ok:
        raise HTTPException(status_code=404, detail="Alpha not found.")
