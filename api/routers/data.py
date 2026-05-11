from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query

from api.data_service import compute_data_summary
from api.db_dashboard import compute_data_dashboard
from api.schemas.models import (
    DashboardBucketParamLiteral,
    DataDashboardResponse,
    DataSummaryRequest,
    DataSummaryResponse,
)
from shunya.errors import ErrorCode, ShunyaError

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data"])

DASHBOARD_INTERVALS = frozenset(
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


@router.get(
    "/dashboard",
    response_model=DataDashboardResponse,
    summary="DB coverage heatmap and risk metrics",
    description=(
        "Reference window is global MIN(ts)..MAX(ts) for the chosen interval and source. "
        "Completeness is the fraction of heatmap buckets with at least one bar. "
        "Bucket granularity defaults to auto (day → week → month) so columns stay under "
        "`max_buckets` (default 200); adjacent periods may be merged (OR). "
        "Optional env: SHUNYA_DASHBOARD_MAX_TICKERS caps symbols (alphabetical order)."
    ),
)
async def get_data_dashboard(
    interval: str = Query(
        "1d",
        description="Stored `ohlcv_bars.interval` key (same as instrument OHLCV API).",
    ),
    source: str = Query("yfinance", description="Stored `ohlcv_bars.source`."),
    bucket: DashboardBucketParamLiteral = Query(
        "auto",
        description="Coverage column granularity; auto picks day/week/month within max_buckets.",
    ),
    max_buckets: int = Query(
        200,
        ge=8,
        le=500,
        description="Maximum heatmap columns after compression.",
    ),
) -> DataDashboardResponse:
    if interval not in DASHBOARD_INTERVALS:
        raise ShunyaError(
            "invalid interval",
            code=ErrorCode.DATA_INVALID_INTERVAL,
            http_status=400,
        )
    if not source or len(source) > 64:
        raise ShunyaError(
            "invalid source",
            code=ErrorCode.DATA_INVALID_SOURCE,
            http_status=400,
        )
    try:
        return await asyncio.to_thread(
            compute_data_dashboard,
            interval=interval,
            source=source,
            bucket=bucket,
            max_buckets=max_buckets,
        )
    except ShunyaError:
        raise
    except ValueError as exc:
        raise ShunyaError(
            str(exc),
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        ) from exc
    except RuntimeError as exc:
        _log.warning("dashboard unavailable: %s", exc)
        raise ShunyaError(
            str(exc),
            code=ErrorCode.FIN_TS_TIMESCALE_UNAVAILABLE,
            http_status=503,
        ) from exc
    except ShunyaError:
        raise
    except Exception as exc:  # noqa: BLE001
        _log.exception("dashboard failed")
        raise ShunyaError(
            "dashboard computation failed",
            code=ErrorCode.DATA_DASHBOARD_FAILED,
            http_status=500,
        ) from exc


@router.post("", response_model=DataSummaryResponse)
async def post_data_summary(body: DataSummaryRequest) -> DataSummaryResponse:
    try:
        return await asyncio.to_thread(compute_data_summary, body)
    except ShunyaError:
        raise
    except ValueError as exc:
        raise ShunyaError(
            str(exc),
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ShunyaError(
            str(exc),
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        ) from exc
