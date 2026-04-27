"""Normalize :class:`~backtest_api.schemas.models.BacktestCreate` for index-universe backtests."""

from __future__ import annotations

import logging

from fastapi import HTTPException, status

from backtest_api.index_catalog import benchmark_for_index
from backtest_api.index_universe import (
    constituent_tickers,
    index_exists,
    tickers_with_ohlcv_in_window,
    validate_ohlcv_window,
)
from backtest_api.schemas.models import BacktestCreate, BarSpecModel
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_spec
from shunya.data.timescale.intervals import bar_spec_to_interval_key

_LOG = logging.getLogger(__name__)


def _interval_key_from_fin_ts_bar_spec(bar_spec: BarSpecModel | None) -> str:
    if bar_spec is None:
        return bar_spec_to_interval_key(default_bar_spec())
    spec = BarSpec(unit=BarUnit(bar_spec.unit), step=int(bar_spec.step))
    return bar_spec_to_interval_key(spec)


def resolve_index_backtest_if_needed(body: BacktestCreate) -> BacktestCreate:
    """
    When ``index_code`` is set: load constituents from Timescale, set benchmark raw index ticker,
    force Timescale-only ``fin_ts``, and validate OHLCV coverage for universe + benchmark.
    """
    raw_code = body.index_code
    if raw_code is None or not str(raw_code).strip():
        return body

    code = str(raw_code).strip().upper()
    try:
        bench = benchmark_for_index(code)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown index_code for benchmark mapping: {code}",
        ) from exc

    if not index_exists(code):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Index {code!r} not found in equity_indexes.",
        )

    tickers = constituent_tickers(code)
    if not tickers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No symbol_index_membership rows for index {code!r}. "
                "Populate from PyTickerSymbols (no OHLCV required): "
                "`uv run python -m shunya.data.timescale.cli sync-index-memberships` "
                "with DATABASE_URL set (after migrations). "
                "For OHLCV + memberships together use `uv run python scripts/bootstrap_sp500_ohlcv.py`."
            ),
        )

    interval = _interval_key_from_fin_ts_bar_spec(body.fin_ts.bar_spec)
    start_d = body.fin_ts.start_date
    end_d = body.fin_ts.end_date

    try:
        validate_ohlcv_window(
            [bench],
            start_date=start_d,
            end_date=end_d,
            interval=interval,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Benchmark {bench!r} has no OHLCV in the backtest window ({start_d!r} .. {end_d!r}, "
                f"interval={interval!r}). Ingest it, e.g. "
                f"`uv run python -m shunya.data.timescale.cli ingest-ohlcv --symbols {bench!r} "
                f"--start {start_d} --end {end_d}`. Original error: {exc}"
            ),
        ) from exc

    if body.omit_index_members_missing_ohlcv:
        covered = tickers_with_ohlcv_in_window(
            tickers,
            start_date=start_d,
            end_date=end_d,
            interval=interval,
        )
        dropped = len(tickers) - len(covered)
        if not covered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No index constituents have OHLCV in the backtest window after filtering; "
                    "ingest bars or disable omit_index_members_missing_ohlcv to see the strict error."
                ),
            )
        if dropped:
            _LOG.info(
                "Index %s: omitting %d/%d constituents missing OHLCV in [%s, %s)",
                code,
                dropped,
                len(tickers),
                start_d,
                end_d,
            )
        tickers = covered
    else:
        universe_plus_bench = sorted(set(tickers + [bench]))
        try:
            validate_ohlcv_window(
                universe_plus_bench,
                start_date=start_d,
                end_date=end_d,
                interval=interval,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    ft = body.fin_ts.model_copy(
        update={
            "ticker_list": tickers,
            "market_data_provider": "timescale",
            "attach_yfinance_classifications": False,
            "attach_fundamentals": False,
        }
    )
    return body.model_copy(
        update={
            "index_code": code,
            "fin_ts": ft,
            "benchmark_ticker": bench,
        }
    )
