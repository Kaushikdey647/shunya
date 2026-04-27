from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from backtest_api.index_universe import list_indices_for_api
from backtest_api.schemas.models import EquityIndexOut

router = APIRouter(prefix="/indices", tags=["indices"])


@router.get("", response_model=list[EquityIndexOut])
def list_equity_indices() -> list[EquityIndexOut]:
    try:
        rows = list_indices_for_api()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not load indices from database: {exc!s}",
        ) from exc
    return [EquityIndexOut.model_validate(r) for r in rows]
