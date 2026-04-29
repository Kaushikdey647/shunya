"""Synchronous backtest job execution (invoked from async worker via ``asyncio.to_thread``)."""

from __future__ import annotations

import traceback
from typing import Any, Optional

from backtest_api.backtest_windows import BACKTEST_SIM_END_EXCLUSIVE, BACKTEST_SIM_START
from backtest_api.errors import FinTsConfigurationError
from backtest_api.repositories import alphas as alphas_repo
from backtest_api.runner import run_backtest_from_payload
from backtest_api.services.ohlcv_yfinance_backfill import (
    backfill_ohlcv_from_yfinance,
    payload_has_index_code,
    tickers_for_ohlcv_backfill,
)
from backtest_api.settings import get_settings


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
) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Run backtest; for index jobs and recoverable OHLCV errors, yfinance-backfill then retry once.

    Returns ``(error_message, serialized_result, summary)`` — on success ``error_message`` is
    ``None`` and the other two are set; on failure ``error_message`` is set.
    """
    _ = job_id
    alpha_id = str(payload["alpha_id"])
    ar = alphas_repo.get_alpha_raw(alpha_id)
    if ar is None:
        return "Alpha not found.", None, None
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
        return None, serialized, summary
    except FinTsConfigurationError as exc:
        return _truncate(exc.message, 8000), None, None
    except Exception as exc1:  # noqa: BLE001
        if not (payload_has_index_code(payload) and _recoverable_fin_ts_data_error(exc1)):
            return _truncate(_format_exc(exc1), 8000), None, None

        settings = get_settings()
        tickers = tickers_for_ohlcv_backfill(payload)
        n_upsert, bf_err = backfill_ohlcv_from_yfinance(
            tickers,
            start_date=BACKTEST_SIM_START,
            end_date_exclusive=BACKTEST_SIM_END_EXCLUSIVE,
            batch_size=int(settings.index_ohlcv_backfill_batch_size),
        )

        try:
            serialized, summary = run_once()
            return None, serialized, summary
        except FinTsConfigurationError as exc2:
            original = _truncate(_format_exc(exc1), 3200)
            retry = _truncate(exc2.message, 3500)
            bf_block = _truncate(
                f"rows_upserted={n_upsert}" + (f"\nerror={bf_err}" if bf_err else "\nstatus=ok"),
                2000,
            )
            chained = f"Original:\n{original}\n\n--- Backfill:\n{bf_block}\n\n--- Retry:\n{retry}"
            return _truncate(chained, 8000), None, None
        except Exception as exc2:  # noqa: BLE001
            original = _truncate(_format_exc(exc1), 2800)
            retry = _truncate(_format_exc(exc2), 2800)
            bf_block = _truncate(
                f"rows_upserted={n_upsert}" + (f"\nerror={bf_err}" if bf_err else "\nstatus=ok"),
                2000,
            )
            chained = f"Original:\n{original}\n\n--- Backfill:\n{bf_block}\n\n--- Retry:\n{retry}"
            return _truncate(chained, 8000), None, None
