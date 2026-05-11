"""Backtest execution port — swap Backtrader for another simulator behind this API."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from shunya.algorithm.finstrat import FinStrat
from shunya.data.fints import finTs


@runtime_checkable
class BacktestEngine(Protocol):
    """Runs a :class:`FinStrat` on a panel and returns the same shape as :meth:`FinBT.results` with ``show=False``."""

    def run_backtest(
        self,
        fin_strat: FinStrat,
        fin_ts: finTs,
        *,
        finbt_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one backtest; keys include ``metrics``, ``equity_curve``, ``turnover_history``, etc."""
        ...
