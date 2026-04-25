from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from backtest_api.data_service import compute_data_summary
from backtest_api.schemas.models import DataSummaryRequest, DataSummaryResponse

router = APIRouter(prefix="/data", tags=["data"])


@router.post("", response_model=DataSummaryResponse)
async def post_data_summary(body: DataSummaryRequest) -> DataSummaryResponse:
    try:
        return await asyncio.to_thread(compute_data_summary, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
