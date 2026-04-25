from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import backtrader as bt
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.fints import finTs
from ..data.timeframes import bar_spec_is_intraday
from .finstrat import FinStrat
from .targets import (
    apply_group_gross_cap,
    apply_group_net_cap,
    cap_deltas_by_adv,
    enforce_turnover_budget,
)


class _FinBTStrategy(bt.Strategy):
    """Rebalance each bar to :meth:`FinStrat.pass_` notionals (per data feed / ticker)."""

    params = (
        ("fin_strat", None),
        ("ticker_order", []),
        ("group_column", None),
        ("sector_gross_cap_fraction", None),
        ("sector_cap_mode", "rescale"),
        ("sector_group_column", "Sector"),
        ("group_net_cap_fraction", None),
        ("turnover_budget_fraction", None),
        ("adv_participation_fraction", None),
        ("constraints_mode", "rescale"),
        ("validate_finite_targets", True),
    )

    def __init__(self) -> None:
        self._ticker_to_data: dict[str, bt.LineRoot] = {}
        for d in self.datas:
            name = d._name
            if name:
                self._ticker_to_data[name] = d
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.target_history: List[Tuple[pd.Timestamp, Dict[str, float]]] = []
        self.turnover_history: List[Tuple[pd.Timestamp, float]] = []
        self.group_exposure_history: List[Tuple[pd.Timestamp, Dict[str, Dict[str, float]]]] = []
        self._prev_targets: Dict[str, float] = {}
        self._prev_exec_dt: Optional[pd.Timestamp] = None

    def _current_dt(self) -> pd.Timestamp:
        return pd.Timestamp(bt.num2date(self.datas[0].datetime[0]))

    def next(self) -> None:
        dt = self._current_dt()
        fs: FinStrat = self.p.fin_strat
        ts = fs._ts
        if fs.intraday_session_isolated_lag and bar_spec_is_intraday(ts.bar_spec):
            if self._prev_exec_dt is not None:
                if ts.trading_session_key(dt) != ts.trading_session_key(self._prev_exec_dt):
                    fs.reset_pipeline_state()
        try:
            self.equity_curve.append((dt, float(self.broker.getvalue())))
            try:
                names = fs.tickers_at(dt)
            except ValueError:
                return
            if not names:
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
                pass_kw["group_ids"] = fs.group_labels_at(dt, names, col)
            raw = fs.pass_(None, capital, execution_date=dt, **pass_kw)
            targets = np.asarray(jnp.asarray(raw), dtype=float)
            name_to_target = {n: float(targets[i]) for i, n in enumerate(names)}
            if self.p.sector_gross_cap_fraction is not None:
                gcol = self.p.sector_group_column
                group_ids = fs.group_labels_at(dt, names, gcol)
                group_map = {n: str(group_ids[i]) for i, n in enumerate(names)}
                name_to_target, _breached = apply_group_gross_cap(
                    name_to_target,
                    group_map,
                    max_group_gross_fraction=float(self.p.sector_gross_cap_fraction),
                    on_breach=self.p.sector_cap_mode,
                )
            if self.p.group_net_cap_fraction is not None:
                gcol = self.p.sector_group_column
                group_ids = fs.group_labels_at(dt, names, gcol)
                group_map = {n: str(group_ids[i]) for i, n in enumerate(names)}
                name_to_target, _breached_net = apply_group_net_cap(
                    name_to_target,
                    group_map,
                    max_group_net_fraction=float(self.p.group_net_cap_fraction),
                    on_breach=self.p.constraints_mode,
                )
            if self.p.turnover_budget_fraction is not None and self._prev_targets:
                name_to_target, _obs_turn, _limit_turn = enforce_turnover_budget(
                    name_to_target,
                    self._prev_targets,
                    max_turnover_fraction=float(self.p.turnover_budget_fraction),
                    on_breach=self.p.constraints_mode,
                )
            if self.p.adv_participation_fraction is not None and self._prev_targets:
                raw_deltas = {
                    n: float(name_to_target.get(n, 0.0) - self._prev_targets.get(n, 0.0))
                    for n in names
                }
                adv_usd = {}
                df = fs._ts.df
                for n in names:
                    key = (n, dt)
                    if key not in df.index:
                        continue
                    row = df.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[-1]
                    close = float(row.get("Close", np.nan))
                    vol = float(row.get("Volume", np.nan))
                    if np.isfinite(close) and np.isfinite(vol) and close > 0 and vol >= 0:
                        adv_usd[n] = close * vol
                capped_deltas, _breached_adv = cap_deltas_by_adv(
                    raw_deltas,
                    adv_usd,
                    max_adv_fraction=float(self.p.adv_participation_fraction),
                    on_breach=self.p.constraints_mode,
                )
                for n in names:
                    base = float(self._prev_targets.get(n, 0.0))
                    name_to_target[n] = base + float(capped_deltas.get(n, 0.0))

            full_targets = {t: float(name_to_target.get(t, 0.0)) for t in self.p.ticker_order}
            if self.p.validate_finite_targets:
                for t, v in full_targets.items():
                    if not np.isfinite(v):
                        raise ValueError(f"non-finite portfolio target for {t!r}: {v!r}")
            self.target_history.append((dt, full_targets))
            if self._prev_targets:
                turnover = sum(
                    abs(full_targets.get(t, 0.0) - self._prev_targets.get(t, 0.0))
                    for t in self.p.ticker_order
                )
                self.turnover_history.append((dt, float(turnover)))
            self._prev_targets = dict(full_targets)

            if self.p.sector_group_column:
                try:
                    gcol = self.p.sector_group_column
                    gids = fs.group_labels_at(dt, names, gcol)
                    gross_by_group: Dict[str, float] = {}
                    net_by_group: Dict[str, float] = {}
                    for i, n in enumerate(names):
                        g = str(gids[i])
                        v = float(name_to_target.get(n, 0.0))
                        gross_by_group[g] = gross_by_group.get(g, 0.0) + abs(v)
                        net_by_group[g] = net_by_group.get(g, 0.0) + v
                    self.group_exposure_history.append(
                        (dt, {"gross_by_group": gross_by_group, "net_by_group": net_by_group})
                    )
                except Exception:
                    pass

            for t in self.p.ticker_order:
                d = self._ticker_to_data.get(t)
                if d is None:
                    continue
                tv = name_to_target.get(t, 0.0)
                self.order_target_value(data=d, target=tv)
        finally:
            self._prev_exec_dt = dt


class FinBT:
    """
    Backtests a :class:`FinStrat` on the same :class:`finTs` panel using backtrader.

    Requires multi-ticker ``fin_ts.df`` with ``(Ticker, Date)`` MultiIndex. Construct
    :class:`FinStrat` with the **same** ``fin_ts`` instance.
    Bar cadence follows ``fin_ts.bar_spec``; :meth:`results` Sharpe and printed return
    labels assume backtrader's per-bar semantics (daily bars ≈ annualized trading figures;
    intraday bars need manual scaling for calendar-year metrics).

    :meth:`run` resets :meth:`FinStrat.reset_pipeline_state` so BRAIN-style **decay**
    (temporal EMA) starts clean. If ``fin_strat.neutralization == 'group'``,
    pass ``group_column`` naming a column present on ``fin_ts.df`` for each
    ``(Ticker, Date)`` row.

    ``commission`` is passed to backtrader's broker as the commission rate; set
    ``slippage_pct`` to a positive fraction for adverse percent slippage on
    executions. A common conservative choice for liquid, active strategies is
    ``0.0002`` (2 bps per trade); fast markets or larger notionals may warrant
    ``0.0005`` (5 bps) or higher.
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
        group_net_cap_fraction: Optional[float] = None,
        turnover_budget_fraction: Optional[float] = None,
        adv_participation_fraction: Optional[float] = None,
        constraints_mode: str = "rescale",
        validate_finite_targets: bool = True,
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
        self._group_net_cap_fraction = group_net_cap_fraction
        self._turnover_budget_fraction = turnover_budget_fraction
        self._adv_participation_fraction = adv_participation_fraction
        self._constraints_mode = str(constraints_mode)
        if self._sector_gross_cap_fraction is not None:
            if not (0.0 < float(self._sector_gross_cap_fraction) <= 1.0):
                raise ValueError("sector_gross_cap_fraction must be in (0, 1]")
            if self._sector_cap_mode not in ("rescale", "raise"):
                raise ValueError("sector_cap_mode must be 'rescale' or 'raise'")
            if self._sector_group_column not in df.columns:
                raise KeyError(
                    f"sector_group_column {self._sector_group_column!r} not found in fin_ts.df"
                )
        if self._group_net_cap_fraction is not None and not (0.0 < float(self._group_net_cap_fraction) <= 1.0):
            raise ValueError("group_net_cap_fraction must be in (0, 1]")
        if self._turnover_budget_fraction is not None and not (0.0 < float(self._turnover_budget_fraction) <= 2.0):
            raise ValueError("turnover_budget_fraction must be in (0, 2]")
        if self._adv_participation_fraction is not None and not (0.0 < float(self._adv_participation_fraction) <= 1.0):
            raise ValueError("adv_participation_fraction must be in (0, 1]")
        if self._constraints_mode not in ("rescale", "raise"):
            raise ValueError("constraints_mode must be 'rescale' or 'raise'")
        self._validate_finite_targets = bool(validate_finite_targets)
        self._cerebro: Optional[bt.Cerebro] = None
        self._run_result: Optional[List[Any]] = None

    @staticmethod
    def _ohlcv_frames(fin_ts: finTs) -> Dict[str, pd.DataFrame]:
        df = fin_ts.df
        need = ["Open", "High", "Low", "Close", "Volume"]
        raw: Dict[str, pd.DataFrame] = {}
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
            raw[t] = ohlc
        if len(raw) < 2:
            raise ValueError("Need at least two tickers with OHLCV rows for FinBT.")

        cal = fin_ts._aligned_calendar
        if cal is not None and len(cal):
            common = cal
        else:
            intraday = bar_spec_is_intraday(fin_ts.bar_spec)
            if intraday:
                idx_sets = [set(pd.DatetimeIndex(f.index)) for f in raw.values()]
            else:
                idx_sets = [
                    set(pd.DatetimeIndex(f.index).normalize()) for f in raw.values()
                ]
            common = pd.DatetimeIndex(sorted(set.intersection(*idx_sets)))
        if len(common) == 0:
            raise ValueError("Empty common OHLCV calendar; align tickers or widen dates.")

        out: Dict[str, pd.DataFrame] = {}
        for t, ohlc in raw.items():
            aligned = ohlc.reindex(common)
            if aligned[need].isna().any().any():
                raise ValueError(
                    f"FinBT: missing OHLCV for {t!r} on common calendar; "
                    "run finTs.align_universe(...) on required columns first."
                )
            out[t] = aligned
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
            group_net_cap_fraction=self._group_net_cap_fraction,
            turnover_budget_fraction=self._turnover_budget_fraction,
            adv_participation_fraction=self._adv_participation_fraction,
            constraints_mode=self._constraints_mode,
            validate_finite_targets=self._validate_finite_targets,
        )
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0, annualize=True)

        self._cerebro = cerebro
        self._run_result = cerebro.run()
        return self

    def results(
        self, *, show: bool = True, include_pre_execution: bool = False
    ) -> Dict[str, Any]:
        """
        Drill-down dashboard: equity curve, drawdown, and key analyzer metrics.

        Call :meth:`run` first. If ``show`` is True, builds matplotlib figures and displays them
        (Jupyter-friendly). If ``show`` is False, skips figure construction for faster batch use;
        ``figure`` in the return dict is then ``None``.
        Returns a dict with ``figure``, ``metrics``, ``equity_curve`` (DataFrame), and raw analyzer dicts.
        """
        if not self._run_result:
            raise RuntimeError("Call run() before results().")

        strat: _FinBTStrategy = self._run_result[0]
        equity_full = pd.DataFrame(strat.equity_curve, columns=["Date", "Equity"]).set_index("Date")
        execution_start: Optional[pd.Timestamp] = (
            pd.Timestamp(strat.target_history[0][0]) if strat.target_history else None
        )
        if not include_pre_execution and execution_start is not None:
            equity = equity_full.loc[equity_full.index >= execution_start].copy()
            if equity.empty:
                equity = equity_full.copy()
                execution_start = None
        else:
            equity = equity_full.copy()
        equity["Peak"] = equity["Equity"].cummax()
        equity["DrawdownPct"] = (equity["Equity"] / equity["Peak"] - 1.0) * 100.0

        ret_a = strat.analyzers.returns.get_analysis()
        dd_a = strat.analyzers.drawdown.get_analysis()
        sh_a = strat.analyzers.sharpe.get_analysis()

        bar = self._ts.bar_spec
        eq_ret = equity["Equity"].pct_change().dropna()
        start_val = float(equity["Equity"].iloc[0]) if len(equity) else float(self._cash)
        end_val = float(equity["Equity"].iloc[-1]) if len(equity) else float(self._cash)
        total_return_pct = ((end_val / start_val - 1.0) * 100.0) if start_val > 0 else 0.0
        avg_bar_return_pct = float(eq_ret.mean() * 100.0) if len(eq_ret) else 0.0
        max_drawdown_pct = float(abs(equity["DrawdownPct"].min())) if len(equity) else 0.0
        in_dd = (equity["DrawdownPct"] < 0).astype(int) if len(equity) else pd.Series(dtype=int)
        max_dd_len = 0
        run = 0
        for v in in_dd.to_numpy(dtype=int):
            if v:
                run += 1
                if run > max_dd_len:
                    max_dd_len = run
            else:
                run = 0
        if bar.unit.name == "SECONDS":
            periods_per_year = (252.0 * 6.5 * 60.0 * 60.0) / max(1.0, float(bar.step))
        elif bar.unit.name == "MINUTES":
            periods_per_year = (252.0 * 6.5 * 60.0) / max(1.0, float(bar.step))
        elif bar.unit.name == "HOURS":
            periods_per_year = (252.0 * 6.5) / max(1.0, float(bar.step))
        elif bar.unit.name == "DAYS":
            periods_per_year = 252.0 / max(1.0, float(bar.step))
        elif bar.unit.name == "WEEKS":
            periods_per_year = 52.0 / max(1.0, float(bar.step))
        elif bar.unit.name == "MONTHS":
            periods_per_year = 12.0 / max(1.0, float(bar.step))
        else:
            periods_per_year = 1.0 / max(1.0, float(bar.step))
        if len(eq_ret) > 1 and float(eq_ret.std(ddof=1)) > 0:
            sharpe_trimmed = float(
                np.sqrt(periods_per_year) * (eq_ret.mean() / eq_ret.std(ddof=1))
            )
        else:
            sharpe_trimmed = None
        metrics = {
            "start_value": start_val,
            "end_value": end_val,
            "total_return_pct": total_return_pct,
            "avg_daily_return_pct": avg_bar_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_len": int(max_dd_len),
            "sharpe_ratio": sharpe_trimmed,
            "execution_start": str(execution_start) if execution_start is not None else None,
            "trimmed_pre_execution": bool(not include_pre_execution and execution_start is not None),
            "bar_unit": str(bar.unit),
            "bar_step": int(bar.step),
            "analyzer_total_return_pct": (ret_a.get("rtot", 0.0) or 0.0) * 100.0,
            "analyzer_avg_return_pct": (ret_a.get("ravg", 0.0) or 0.0) * 100.0,
            "analyzer_max_drawdown_pct": (dd_a.get("max", {}).get("drawdown", None) or 0.0),
            "analyzer_sharpe_ratio": sh_a.get("sharperatio", None),
        }

        turnover_df = pd.DataFrame(
            strat.turnover_history,
            columns=["Date", "TurnoverUSD"],
        ).set_index("Date") if strat.turnover_history else pd.DataFrame(columns=["TurnoverUSD"])
        if not include_pre_execution and execution_start is not None and not turnover_df.empty:
            turnover_df = turnover_df.loc[turnover_df.index >= execution_start].copy()
        if not turnover_df.empty and not equity.empty:
            aligned = turnover_df.join(equity[["Equity"]], how="left").ffill()
            turnover_pct = aligned["TurnoverUSD"] / aligned["Equity"].replace(0, np.nan)
            metrics["avg_turnover_pct"] = float(turnover_pct.mean(skipna=True) * 100.0)
            metrics["max_turnover_pct"] = float(turnover_pct.max(skipna=True) * 100.0)
            metrics["rebalance_count"] = int(len(turnover_df))
        else:
            metrics["avg_turnover_pct"] = 0.0
            metrics["max_turnover_pct"] = 0.0
            metrics["rebalance_count"] = 0

        if strat.target_history:
            latest_targets = strat.target_history[-1][1]
            gross = sum(abs(v) for v in latest_targets.values())
            top = max((abs(v) for v in latest_targets.values()), default=0.0)
            metrics["top_name_gross_share_pct"] = float((top / gross) * 100.0) if gross > 0 else 0.0
        else:
            metrics["top_name_gross_share_pct"] = 0.0

        if strat.group_exposure_history:
            _, ge = strat.group_exposure_history[-1]
            gross_map = ge.get("gross_by_group", {})
            net_map = ge.get("net_by_group", {})
            total_g = sum(float(v) for v in gross_map.values())
            max_group_gross = max((float(v) for v in gross_map.values()), default=0.0)
            max_group_net_abs = max((abs(float(v)) for v in net_map.values()), default=0.0)
            metrics["max_group_gross_share_pct"] = (
                float((max_group_gross / total_g) * 100.0) if total_g > 0 else 0.0
            )
            metrics["max_group_net_share_pct"] = (
                float((max_group_net_abs / total_g) * 100.0) if total_g > 0 else 0.0
            )
        else:
            metrics["max_group_gross_share_pct"] = 0.0
            metrics["max_group_net_share_pct"] = 0.0

        fig: Optional[Any] = None
        if show:
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
                f"Total return: {metrics['total_return_pct']:.2f}%",
                f"Avg return per bar: {metrics['avg_daily_return_pct']:.4f}%",
                f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%",
                f"Sharpe (annualized): {metrics['sharpe_ratio']}",
                f"Bar spec: {metrics['bar_unit']} x{metrics['bar_step']}",
                f"Avg turnover: {metrics['avg_turnover_pct']:.2f}%",
                f"Top-name gross share: {metrics['top_name_gross_share_pct']:.2f}%",
            ]
            if metrics["execution_start"] is not None:
                lines.append(f"Execution start: {metrics['execution_start']}")
            ax_tbl.text(0.02, 0.95, "\n".join(lines), transform=ax_tbl.transAxes, va="top", family="monospace")

            plt.tight_layout()
            plt.show()

        target_history = list(strat.target_history)
        group_history = list(strat.group_exposure_history)
        if not include_pre_execution and execution_start is not None:
            target_history = [
                (dt, tg) for dt, tg in target_history if pd.Timestamp(dt) >= execution_start
            ]
            group_history = [
                (dt, ge) for dt, ge in group_history if pd.Timestamp(dt) >= execution_start
            ]

        return {
            "figure": fig,
            "metrics": metrics,
            "equity_curve": equity,
            "equity_curve_full": equity_full,
            "returns_analysis": ret_a,
            "drawdown_analysis": dd_a,
            "sharpe_analysis": sh_a,
            "turnover_history": turnover_df,
            "target_history": target_history,
            "group_exposure_history": group_history,
        }
