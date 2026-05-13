"""Synchronous backtest job execution (invoked from async worker via ``asyncio.to_thread``)."""

from __future__ import annotations

import traceback
from typing import Any, Optional

from api.backtest_windows import BACKTEST_SIM_END_EXCLUSIVE, BACKTEST_SIM_START
from api.errors import FinTsConfigurationError
from api.repositories import alphas as alphas_repo
from api.repositories import backtests as jobs_repo
from api.runner import run_backtest_from_payload
from api.services.ohlcv_yfinance_backfill import (
    backfill_ohlcv_from_yfinance,
    payload_has_index_code,
    tickers_for_ohlcv_backfill,
)
from api.tunable_config import get_effective_tunables
from shunya.errors import ErrorCode


def _recoverable_fin_ts_data_error(exc: BaseException) -> bool:
    if not isinstance(exc, ValueError):
        return False
    head = str(exc).split(":", 1)[0].strip()
    return head in (
        "strict_ohlcv",
        "strict_empty",
        "strict_provider_universe",
        "provider_index_out_of_range",
        "strict_trading_grid",
    )


def _format_exc(exc: BaseException) -> str:
    return f"{exc!s}\n{traceback.format_exc()}"


def _truncate(msg: str, limit: int) -> str:
    if len(msg) <= limit:
        return msg
    return msg[: max(0, limit - 24)] + "\n...[truncated]..."


def execute_claimed_backtest_job(
    job_id: str, payload: dict[str, Any]
) -> tuple[Optional[str], Optional[str], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Run backtest; for index jobs and recoverable OHLCV errors, yfinance-backfill then retry once.

    Returns ``(error_message, error_code, serialized_result, summary)`` — on success the first
    two are ``None`` and the latter two are set; on failure ``error_message`` is set and
    ``error_code`` may be a stable :class:`~shunya.errors.ErrorCode` string.
    """
    jobs_repo.append_job_log(job_id, "Worker started; resolving alpha and universe.")
    alpha_id = str(payload["alpha_id"])
    ar = alphas_repo.get_alpha_raw(alpha_id)
    if ar is None:
        return "Alpha not found.", str(ErrorCode.ALPHA_NOT_FOUND), None, None
    ir = ar.get("import_ref")
    sc = ar.get("source_code")
    finstrat = dict(ar["finstrat_config"])

    def run_once() -> tuple[dict[str, Any], dict[str, Any]]:
        return run_backtest_from_payload(
            payload,
            ir if ir is not None else None,
            sc if sc is not None else None,
            finstrat,
        )

    try:
        serialized, summary = run_once()
        return None, None, serialized, summary
    except FinTsConfigurationError as exc:
        return _truncate(exc.message, 8000), exc.code, None, None
    except Exception as exc1:  # noqa: BLE001
        if not (payload_has_index_code(payload) and _recoverable_fin_ts_data_error(exc1)):
            return (
                _truncate(_format_exc(exc1), 8000),
                str(ErrorCode.BACKTEST_JOB_EXECUTION_ERROR),
                None,
                None,
            )

        tun = get_effective_tunables()
        tickers = tickers_for_ohlcv_backfill(payload)
        n_upsert, bf_err = backfill_ohlcv_from_yfinance(
            tickers,
            start_date=BACKTEST_SIM_START,
            end_date_exclusive=BACKTEST_SIM_END_EXCLUSIVE,
            batch_size=int(tun.index_ohlcv_backfill_batch_size),
        )

        try:
            serialized, summary = run_once()
            return None, None, serialized, summary
        except FinTsConfigurationError as exc2:
            original = _truncate(_format_exc(exc1), 3200)
            retry = _truncate(exc2.message, 3500)
            bf_block = _truncate(
                f"rows_upserted={n_upsert}" + (f"\nerror={bf_err}" if bf_err else "\nstatus=ok"),
                2000,
            )
            chained = f"Original:\n{original}\n\n--- Backfill:\n{bf_block}\n\n--- Retry:\n{retry}"
            return _truncate(chained, 8000), str(ErrorCode.BACKTEST_JOB_EXECUTION_ERROR), None, None
        except Exception as exc2:  # noqa: BLE001
            original = _truncate(_format_exc(exc1), 2800)
            retry = _truncate(_format_exc(exc2), 2800)
            bf_block = _truncate(
                f"rows_upserted={n_upsert}" + (f"\nerror={bf_err}" if bf_err else "\nstatus=ok"),
                2000,
            )
            chained = f"Original:\n{original}\n\n--- Backfill:\n{bf_block}\n\n--- Retry:\n{retry}"
            return _truncate(chained, 8000), str(ErrorCode.BACKTEST_JOB_EXECUTION_ERROR), None, None
