"""Stable error codes and domain exceptions (no FastAPI imports)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping


class ErrorCode(StrEnum):
    """Machine-readable codes shared with HTTP API and UI clients."""

    # FinTs / market data
    FIN_TS_TIMESCALE_DEPENDENCY = "FIN_TS_TIMESCALE_DEPENDENCY"
    FIN_TS_TIMESCALE_DSN_REQUIRED = "FIN_TS_TIMESCALE_DSN_REQUIRED"
    FIN_TS_TIMESCALE_UNAVAILABLE = "FIN_TS_TIMESCALE_UNAVAILABLE"
    FIN_TS_FUNDAMENTALS_DSN_REQUIRED = "FIN_TS_FUNDAMENTALS_DSN_REQUIRED"
    FIN_TS_ALPACA_KEYS_REQUIRED = "FIN_TS_ALPACA_KEYS_REQUIRED"

    # Alphas / backtests
    ALPHA_NOT_FOUND = "ALPHA_NOT_FOUND"
    ALPHA_NAME_CONFLICT = "ALPHA_NAME_CONFLICT"
    BACKTEST_JOB_NOT_FOUND = "BACKTEST_JOB_NOT_FOUND"
    BACKTEST_RESULT_MISSING = "BACKTEST_RESULT_MISSING"
    BACKTEST_RESULT_NOT_READY = "BACKTEST_RESULT_NOT_READY"
    DELETE_BATCH_TOO_LARGE = "DELETE_BATCH_TOO_LARGE"
    INVALID_STATUS_FILTER = "INVALID_STATUS_FILTER"
    BACKTEST_JOB_EXECUTION_ERROR = "BACKTEST_JOB_EXECUTION_ERROR"
    BACKTEST_JOB_SERVER_RESTART = "BACKTEST_JOB_SERVER_RESTART"
    BACKTEST_INDEX_UNKNOWN = "BACKTEST_INDEX_UNKNOWN"
    BACKTEST_INDEX_NOT_FOUND = "BACKTEST_INDEX_NOT_FOUND"
    BACKTEST_INDEX_NO_MEMBERS = "BACKTEST_INDEX_NO_MEMBERS"
    BACKTEST_INDEX_OHLCV = "BACKTEST_INDEX_OHLCV"
    BACKTEST_UNIVERSE_NOT_FOUND = "BACKTEST_UNIVERSE_NOT_FOUND"
    BACKTEST_UNIVERSE_NO_MEMBERS = "BACKTEST_UNIVERSE_NO_MEMBERS"
    BACKTEST_UNIVERSE_OHLCV = "BACKTEST_UNIVERSE_OHLCV"
    UNIVERSE_NOT_FOUND = "UNIVERSE_NOT_FOUND"
    UNIVERSE_NAME_CONFLICT = "UNIVERSE_NAME_CONFLICT"
    UNIVERSE_MEMBER_NOT_EQUITY = "UNIVERSE_MEMBER_NOT_EQUITY"
    UNIVERSE_UNKNOWN_TICKER = "UNIVERSE_UNKNOWN_TICKER"

    # Data summary / dashboard
    DATA_INVALID_INTERVAL = "DATA_INVALID_INTERVAL"
    DATA_INVALID_SOURCE = "DATA_INVALID_SOURCE"
    DATA_DASHBOARD_FAILED = "DATA_DASHBOARD_FAILED"

    # Validation / generic
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ShunyaError(Exception):
    """Base domain error with a stable ``code`` for API and UI mapping."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | str = ErrorCode.INTERNAL_ERROR,
        http_status: int = 500,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = str(code)
        self.http_status = int(http_status)
        self.context: dict[str, Any] = dict(context) if context else {}

    @property
    def status_code(self) -> int:
        """Alias for ``http_status`` (legacy FinTs / API naming)."""
        return self.http_status

    def to_detail_body(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            out["fields"] = self.context
        return out


class FinTsConfigurationError(ShunyaError):
    """Invalid or unavailable FinTs / market data configuration."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | str = ErrorCode.FIN_TS_TIMESCALE_UNAVAILABLE,
        status_code: int = 503,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, http_status=status_code, context=context)
