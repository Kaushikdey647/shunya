"""FastAPI exception handlers for structured ``detail`` payloads."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from shunya.errors import ErrorCode, ShunyaError


def _validation_detail(exc: RequestValidationError) -> dict[str, Any]:
    return {
        "code": str(ErrorCode.VALIDATION_ERROR),
        "message": "Request validation failed.",
        "fields": {"errors": jsonable_encoder(exc.errors())},
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ShunyaError)
    async def shunya_error_handler(_request: Request, exc: ShunyaError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": exc.to_detail_body()},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": _validation_detail(exc)},
        )
