"""Default Backtrader-backed :class:`~shunya.ports.backtest_engine.BacktestEngine`."""

from __future__ import annotations

from typing import Any

from shunya.algorithm.finbt import FinBT
from shunya.algorithm.finstrat import FinStrat
from shunya.data.fints import finTs
from shunya.ports.backtest_engine import BacktestEngine


class BacktraderBacktestEngine:
    """Adapter wrapping :class:`FinBT` / cerebro (current production path)."""

    def run_backtest(
        self,
        fin_strat: FinStrat,
        fin_ts: finTs,
        *,
        finbt_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        bt = FinBT(fin_strat, fin_ts, **finbt_kwargs).run()
        return bt.results(show=False)
