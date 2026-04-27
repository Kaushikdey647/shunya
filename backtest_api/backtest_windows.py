"""Fixed tune / test calendar windows and daily bar spec for HTTP backtests."""

from __future__ import annotations

from backtest_api.schemas.models import BacktestCreate, BarSpecModel

# OHLCV and FinTs use [start, end) semantics (end exclusive).
BACKTEST_SIM_START = "2020-01-01"
BACKTEST_SIM_END_EXCLUSIVE = "2026-01-01"
# First instant in the "test" slice; tune-only results exclude this date onward.
BACKTEST_TEST_START = "2025-01-01"


def standard_backtest_bar_spec() -> BarSpecModel:
    return BarSpecModel(unit="DAYS", step=1)


def normalize_backtest_create(body: BacktestCreate) -> BacktestCreate:
    """Force simulation window and daily bars; ignore client-supplied dates/bar_spec."""
    ft = body.fin_ts.model_copy(
        update={
            "start_date": BACKTEST_SIM_START,
            "end_date": BACKTEST_SIM_END_EXCLUSIVE,
            "bar_spec": standard_backtest_bar_spec(),
        }
    )
    return body.model_copy(update={"fin_ts": ft})
