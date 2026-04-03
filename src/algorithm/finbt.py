from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import backtrader as bt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.fints import finTs
from .finstrat import FinStrat
from .targets import apply_group_gross_cap


class _FinBTStrategy(bt.Strategy):
    """Rebalance each bar to :meth:`FinStrat.pass_` notionals (per data feed / ticker)."""

    params = (
        ("fin_strat", None),
        ("ticker_order", []),
        ("group_column", None),
        ("sector_gross_cap_fraction", None),
        ("sector_cap_mode", "rescale"),
        ("sector_group_column", "Sector"),
    )

    def __init__(self) -> None:
        self._ticker_to_data: dict[str, bt.LineRoot] = {}
        for d in self.datas:
            name = d._name
            if name:
                self._ticker_to_data[name] = d
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

    def _current_dt(self) -> pd.Timestamp:
        return pd.Timestamp(bt.num2date(self.datas[0].datetime[0]))

    def next(self) -> None:
        self.equity_curve.append((self._current_dt(), float(self.broker.getvalue())))
        fs: FinStrat = self.p.fin_strat
        try:
            panel, names = fs.panel_at(self._current_dt(), live=True)
        except ValueError:
            return
        capital = float(self.broker.getvalue())
        if capital <= 0:
            return
        pass_kw: dict = {"tickers": names}
        if fs.neutralization == "group":
            col = self.p.group_column
            if not col:
                raise ValueError(
                    "FinStrat uses neutralization='group'; pass group_column= to FinBT(…)"
                )
            pass_kw["group_ids"] = fs.group_labels_at(self._current_dt(), names, col)
        raw = fs.pass_(panel, capital, **pass_kw)
        targets = np.asarray(jnp.asarray(raw), dtype=float)
        name_to_target = {n: float(targets[i]) for i, n in enumerate(names)}
        if self.p.sector_gross_cap_fraction is not None:
            gcol = self.p.sector_group_column
            group_ids = fs.group_labels_at(self._current_dt(), names, gcol)
            group_map = {n: str(group_ids[i]) for i, n in enumerate(names)}
            name_to_target, _breached = apply_group_gross_cap(
                name_to_target,
                group_map,
                max_group_gross_fraction=float(self.p.sector_gross_cap_fraction),
                on_breach=self.p.sector_cap_mode,
            )
        for t in self.p.ticker_order:
            d = self._ticker_to_data.get(t)
            if d is None:
                continue
            tv = name_to_target.get(t, 0.0)
            self.order_target_value(data=d, target=tv)


class FinBT:
    """
    Backtests a :class:`FinStrat` on the same :class:`finTs` panel using backtrader.

    Requires multi-ticker ``fin_ts.df`` with ``(Ticker, Date)`` MultiIndex. Construct
    :class:`FinStrat` with the **same** ``fin_ts`` instance.

    :meth:`run` resets :meth:`FinStrat.reset_pipeline_state` so BRAIN-style **decay**
    (temporal EMA) starts clean. If ``fin_strat.neutralization == 'group'``,
    pass ``group_column`` naming a column present on ``fin_ts.df`` for each
    ``(Ticker, Date)`` row.

    ``commission`` is passed to backtrader's broker as the commission rate; set
    ``slippage_pct`` to a positive fraction (e.g. ``0.0005`` for 5 bps) for adverse
    percent slippage on executions.
    """

    def __init__(
        self,
        fin_strat: FinStrat,
        fin_ts: finTs,
        *,
        cash: float = 100_000.0,
        commission: float = 0.0,
        slippage_pct: float = 0.0,
        group_column: Optional[str] = None,
        sector_gross_cap_fraction: Optional[float] = None,
        sector_cap_mode: str = "rescale",
        sector_group_column: str = "Sector",
    ) -> None:
        if fin_strat._ts is not fin_ts:
            raise ValueError("fin_strat must be built with the same fin_ts instance (identity).")
        df = fin_ts.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("fin_ts.df is empty")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("FinBT requires multi-ticker finTs with MultiIndex (Ticker, Date).")
        if tuple(df.index.names) != ("Ticker", "Date"):
            raise ValueError(f"Expected index names ('Ticker', 'Date'), got {tuple(df.index.names)!r}")

        self._strat = fin_strat
        self._ts = fin_ts
        self._cash = float(cash)
        self._commission = float(commission)
        self._slippage_pct = float(slippage_pct)
        if self._slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        self._group_column = group_column
        if fin_strat.neutralization == "group":
            if not self._group_column:
                self._group_column = "Sector"
            if self._group_column not in df.columns:
                raise KeyError(
                    f"group_column {self._group_column!r} not found in fin_ts.df. "
                    "Expected one of {'Sector', 'Industry', 'SubIndustry'} "
                    "or a custom column added per (Ticker, Date)."
                )
        self._sector_gross_cap_fraction = sector_gross_cap_fraction
        self._sector_cap_mode = str(sector_cap_mode)
        self._sector_group_column = str(sector_group_column)
        if self._sector_gross_cap_fraction is not None:
            if not (0.0 < float(self._sector_gross_cap_fraction) <= 1.0):
                raise ValueError("sector_gross_cap_fraction must be in (0, 1]")
            if self._sector_cap_mode not in ("rescale", "raise"):
                raise ValueError("sector_cap_mode must be 'rescale' or 'raise'")
            if self._sector_group_column not in df.columns:
                raise KeyError(
                    f"sector_group_column {self._sector_group_column!r} not found in fin_ts.df"
                )
        self._cerebro: Optional[bt.Cerebro] = None
        self._run_result: Optional[List[Any]] = None

    @staticmethod
    def _ohlcv_frames(fin_ts: finTs) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        df = fin_ts.df
        need = ["Open", "High", "Low", "Close", "Volume"]
        for t in fin_ts.ticker_list:
            if t not in df.index.get_level_values(0):
                continue
            sub = df.xs(t, level="Ticker").copy()
            miss = [c for c in need if c not in sub.columns]
            if miss:
                raise KeyError(f"Ticker {t!r} missing columns {miss}")
            ohlc = sub[need].sort_index()
            ohlc.index = pd.to_datetime(ohlc.index)
            ohlc = ohlc[~ohlc.index.duplicated(keep="last")]
            out[t] = ohlc
        if len(out) < 2:
            raise ValueError("Need at least two tickers with OHLCV rows for FinBT.")
        return out

    def run(self, **cerebro_kw: Any) -> FinBT:
        """Build cerebro, attach data feeds, run the backtest. Chainable."""
        self._strat.reset_pipeline_state()
        frames = self._ohlcv_frames(self._ts)
        tickers = [t for t in self._ts.ticker_list if t in frames]

        cerebro = bt.Cerebro(**cerebro_kw)
        cerebro.broker.setcash(self._cash)
        cerebro.broker.setcommission(commission=self._commission)
        if self._slippage_pct > 0:
            cerebro.broker.set_slippage_perc(
                perc=self._slippage_pct,
                slip_open=True,
                slip_limit=True,
                slip_match=True,
                slip_out=False,
            )

        for t in tickers:
            data = bt.feeds.PandasData(dataname=frames[t], name=t)
            cerebro.adddata(data, name=t)

        cerebro.addstrategy(
            _FinBTStrategy,
            fin_strat=self._strat,
            ticker_order=tickers,
            group_column=self._group_column,
            sector_gross_cap_fraction=self._sector_gross_cap_fraction,
            sector_cap_mode=self._sector_cap_mode,
            sector_group_column=self._sector_group_column,
        )
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0, annualize=True)

        self._cerebro = cerebro
        self._run_result = cerebro.run()
        return self

    def results(self, *, show: bool = True) -> Dict[str, Any]:
        """
        Drill-down dashboard: equity curve, drawdown, and key analyzer metrics.

        Call :meth:`run` first. If ``show`` is True, displays matplotlib figures (Jupyter-friendly).
        Returns a dict with ``figure``, ``metrics``, ``equity_curve`` (DataFrame), and raw analyzer dicts.
        """
        if not self._run_result:
            raise RuntimeError("Call run() before results().")

        strat: _FinBTStrategy = self._run_result[0]
        equity = pd.DataFrame(strat.equity_curve, columns=["Date", "Equity"]).set_index("Date")
        equity["Peak"] = equity["Equity"].cummax()
        equity["DrawdownPct"] = (equity["Equity"] / equity["Peak"] - 1.0) * 100.0

        ret_a = strat.analyzers.returns.get_analysis()
        dd_a = strat.analyzers.drawdown.get_analysis()
        sh_a = strat.analyzers.sharpe.get_analysis()

        metrics = {
            "start_value": self._cash,
            "end_value": float(equity["Equity"].iloc[-1]) if len(equity) else self._cash,
            "total_return_pct": (ret_a.get("rtot", 0.0) or 0.0) * 100.0,
            "avg_daily_return_pct": (ret_a.get("ravg", 0.0) or 0.0) * 100.0,
            "max_drawdown_pct": (dd_a.get("max", {}).get("drawdown", None) or 0.0),
            "max_drawdown_len": dd_a.get("max", {}).get("len", None),
            "sharpe_ratio": sh_a.get("sharperatio", None),
        }

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1, 1]})
        ax_eq, ax_dd, ax_tbl = axes

        equity["Equity"].plot(ax=ax_eq, color="C0", linewidth=1.2)
        ax_eq.set_title("Portfolio equity")
        ax_eq.set_ylabel("Value")
        ax_eq.grid(True, alpha=0.3)

        equity["DrawdownPct"].plot(ax=ax_dd, color="C3", linewidth=1.0)
        ax_dd.set_title("Drawdown (%)")
        ax_dd.set_ylabel("%")
        ax_dd.grid(True, alpha=0.3)

        ax_tbl.axis("off")
        lines = [
            f"End equity: {metrics['end_value']:,.2f}",
            f"Total return (analyzer): {metrics['total_return_pct']:.2f}%",
            f"Avg daily return: {metrics['avg_daily_return_pct']:.4f}%",
            f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%",
            f"Sharpe (bt annualized): {metrics['sharpe_ratio']}",
        ]
        ax_tbl.text(0.02, 0.95, "\n".join(lines), transform=ax_tbl.transAxes, va="top", family="monospace")

        plt.tight_layout()
        if show:
            plt.show()

        return {
            "figure": fig,
            "metrics": metrics,
            "equity_curve": equity,
            "returns_analysis": ret_a,
            "drawdown_analysis": dd_a,
            "sharpe_analysis": sh_a,
        }
