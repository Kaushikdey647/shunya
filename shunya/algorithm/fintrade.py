from __future__ import annotations

import logging
import os
import uuid
from typing import Literal, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import PositionSide
from alpaca.trading.models import Position

from ..data.fints import finTs
from .decision import DecisionContext, resolve_panel_timestamp, validate_panel_timestamp
from .execution import AlpacaExecutionAdapter, ExecutionReport
from .finstrat import FinStrat
from .orders import OrderBuilder, OrderType, OrderVariety, RiskPolicy
from .targets import (
    apply_group_gross_cap,
    apply_group_net_cap,
    broker_deltas,
    cap_deltas_by_adv,
    enforce_turnover_budget,
    target_usd_universe,
)

logger = logging.getLogger(__name__)


def _signed_position_usd(position: Position) -> float:
    """Best-effort signed USD exposure (long positive, short negative)."""
    mv = position.market_value
    if mv is None:
        return 0.0
    v = float(mv)
    if getattr(position, "side", None) == PositionSide.SHORT:
        return -abs(v)
    return v


class FinTrade:
    """
    Live / paper execution: turn a :class:`FinStrat` into dollar targets and submit
    **delta** market orders via the Alpaca Trading API (`alpaca-py`).

    Uses :class:`AlpacaExecutionAdapter` for clock checks, optional buying-power caps,
    asset validation, ``client_order_id`` idempotency hints, and submit retries.

    Pass :class:`DecisionContext` to make the signal date and data provenance explicit.
    For Yahoo-backed panels, prefer ``as_of`` = last fully settled session and
    ``data_source="yfinance_research"`` (default on the context).

    Credentials default to ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` when omitted.

    Unlike :class:`FinBT`, :meth:`run` does **not** call
    :meth:`FinStrat.reset_pipeline_state` so temporal ``decay`` persists across runs.
    """

    def __init__(
        self,
        fin_strat: FinStrat,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        trading_client: Optional[TradingClient] = None,
        group_column: Optional[str] = None,
        execution_adapter: Optional[AlpacaExecutionAdapter] = None,
        buying_power_buffer: float = 0.98,
    ) -> None:
        self._strat = fin_strat
        self._group_column = group_column
        if fin_strat.neutralization == "group" and not self._group_column:
            self._group_column = "Sector"
        if trading_client is not None:
            self._client = trading_client
        else:
            key = api_key or os.environ.get("APCA_API_KEY_ID")
            sec = secret_key or os.environ.get("APCA_API_SECRET_KEY")
            self._client = TradingClient(api_key=key, secret_key=sec, paper=paper)
        self._adapter = execution_adapter or AlpacaExecutionAdapter(
            self._client,
            buying_power_buffer=buying_power_buffer,
        )

    def run(
        self,
        tradecapital: float,
        fin_ts: finTs,
        *,
        decision: Optional[DecisionContext] = None,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        min_order_notional: float = 1.0,
        dry_run: bool = False,
        require_market_open: bool = False,
        cap_to_buying_power: bool = True,
        observe_order_status: bool = True,
        status_max_polls: int = 3,
        status_poll_interval_seconds: float = 0.5,
        sector_gross_cap_fraction: Optional[float] = None,
        sector_cap_mode: Literal["rescale", "raise"] = "rescale",
        sector_group_column: str = "Sector",
        decision_enforce_weekday: bool = True,
        decision_strict_same_session: bool = False,
        decision_max_staleness_days: Optional[int] = 7,
        group_net_cap_fraction: Optional[float] = None,
        turnover_budget_fraction: Optional[float] = None,
        adv_participation_fraction: Optional[float] = None,
        constraints_mode: Literal["rescale", "raise"] = "rescale",
        reconcile_after_submit: bool = True,
        reconciliation_policy: Literal["warn_only", "retry_once", "cancel_and_retarget"] = "warn_only",
        reconciliation_tolerance_notional: float = 5.0,
        risk_policy: Optional[RiskPolicy] = None,
        order_type: OrderType = OrderType.MARKET,
        order_variety: OrderVariety = OrderVariety.REGULAR,
        order_exchange: Optional[str] = None,
        order_product: Optional[str] = None,
    ) -> ExecutionReport:
        """
        Build panel, run :meth:`FinStrat.pass_`, diff vs Alpaca, submit orders.

        ``tradecapital`` is the gross book requested for ``pass_``; when
        ``cap_to_buying_power`` is True it is clipped to Alpaca buying power times the
        adapter buffer before sizing.
        """
        if self._strat._ts is not fin_ts:
            raise ValueError("fin_ts must be the same instance bound to FinStrat (identity).")
        df = fin_ts.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("fin_ts.df is empty")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("FinTrade requires multi-ticker finTs with MultiIndex (Ticker, Date).")
        if tuple(df.index.names) != ("Ticker", "Date"):
            raise ValueError(f"Expected index names ('Ticker', 'Date'), got {tuple(df.index.names)!r}")
        if tradecapital <= 0:
            raise ValueError(f"tradecapital must be positive, got {tradecapital}")
        if constraints_mode not in ("rescale", "raise"):
            raise ValueError("constraints_mode must be 'rescale' or 'raise'")
        if reconciliation_policy not in ("warn_only", "retry_once", "cancel_and_retarget"):
            raise ValueError(
                "reconciliation_policy must be 'warn_only', 'retry_once', or 'cancel_and_retarget'"
            )
        if reconciliation_tolerance_notional < 0:
            raise ValueError("reconciliation_tolerance_notional must be non-negative")
        n = self._strat.neutralization
        if n == "group":
            if not self._group_column:
                raise ValueError("neutralization='group' requires group_column")
            if self._group_column not in df.columns:
                raise KeyError(
                    f"group_column {self._group_column!r} not found in fin_ts.df. "
                    "Expected one of {'Sector', 'Industry', 'SubIndustry'} "
                    "or a custom column added per (Ticker, Date)."
                )
        elif n == "sector":
            if "Sector" not in df.columns:
                raise KeyError(
                    "neutralization='sector' requires a 'Sector' column on fin_ts.df."
                )
        elif n == "industry":
            if "Industry" not in df.columns:
                raise KeyError(
                    "neutralization='industry' requires an 'Industry' column on fin_ts.df."
                )

        idx_max = pd.Timestamp(df.index.get_level_values("Date").max())
        dt = resolve_panel_timestamp(
            decision=decision,
            explicit_as_of=as_of,
            index_max_date=idx_max,
        )
        tz = decision.timezone if decision is not None else "America/New_York"
        dt, decision_warnings = validate_panel_timestamp(
            resolved_as_of=dt,
            index_max_date=idx_max,
            timezone=tz,
            enforce_weekday=decision_enforce_weekday,
            strict_same_session=decision_strict_same_session,
            max_staleness_days=decision_max_staleness_days,
        )
        correlation_id = uuid.uuid4().hex
        data_source = decision.data_source if decision is not None else None

        warnings: list[str] = list(decision_warnings)
        if data_source == "yfinance_research":
            warnings.append(
                "data_source=yfinance_research: Yahoo history may not match Alpaca fills — use AlpacaHistoricalMarketDataProvider + DecisionContext(data_source='alpaca_bars') for tighter parity."
            )

        clock_open: Optional[bool] = None
        try:
            clock_open = self._client.get_clock().is_open
        except Exception as e:
            logger.warning("Could not read market clock: %s", e)
            warnings.append("market_clock_unavailable")

        if require_market_open and not dry_run:
            self._adapter.assert_market_open()

        names = self._strat.tickers_at(dt)
        if not names:
            raise ValueError(f"No tradable tickers with finite OHLCV at {dt!s}")

        pass_kw: dict = {"tickers": names}
        if self._strat.neutralization == "group":
            col = self._group_column
            if not col:
                raise ValueError("FinStrat uses neutralization='group'; pass group_column= to FinTrade")
            pass_kw["group_ids"] = self._strat.group_labels_at(dt, names, col)

        eff_cap = float(tradecapital)
        buying_power_obs: Optional[float] = None
        if cap_to_buying_power or not dry_run:
            try:
                buying_power_obs = self._adapter.buying_power()
                if cap_to_buying_power and buying_power_obs is not None:
                    cap_bp = float(buying_power_obs) * self._adapter.buying_power_fraction
                    if eff_cap > cap_bp:
                        warnings.append(
                            f"tradecapital {eff_cap} capped to buying_power*fraction ≈ {cap_bp:.2f}"
                        )
                        eff_cap = min(eff_cap, cap_bp)
            except Exception as e:
                logger.warning("Could not read buying power: %s", e)
                warnings.append("buying_power_unavailable")

        targets_vec = np.asarray(
            jnp.asarray(self._strat.pass_(None, eff_cap, execution_date=dt, **pass_kw)),
            dtype=float,
        )
        targets = target_usd_universe(names, targets_vec, fin_ts.ticker_list)
        group_map = {t: "UnknownSector" for t in fin_ts.ticker_list}
        need_group_map = (
            sector_gross_cap_fraction is not None or group_net_cap_fraction is not None
        )
        if need_group_map:
            if sector_group_column not in df.columns:
                raise KeyError(
                    f"sector_group_column {sector_group_column!r} not found in fin_ts.df"
                )
            for t in fin_ts.ticker_list:
                key = (t, dt)
                if key not in df.index:
                    continue
                g = df.loc[key, sector_group_column]
                if isinstance(g, pd.Series):
                    g = g.iloc[-1]
                group_map[t] = str(g) if pd.notna(g) else "UnknownSector"
        if sector_gross_cap_fraction is not None:
            if not (0.0 < sector_gross_cap_fraction <= 1.0):
                raise ValueError("sector_gross_cap_fraction must be in (0, 1]")
            targets, breached_groups = apply_group_gross_cap(
                targets,
                group_map,
                max_group_gross_fraction=sector_gross_cap_fraction,
                on_breach=sector_cap_mode,
            )
            if breached_groups:
                warnings.append(
                    f"sector_gross_cap_applied groups={breached_groups}, "
                    f"fraction={sector_gross_cap_fraction}, mode={sector_cap_mode}"
                )
        if group_net_cap_fraction is not None:
            targets, breached_net = apply_group_net_cap(
                targets,
                group_map,
                max_group_net_fraction=float(group_net_cap_fraction),
                on_breach=constraints_mode,
            )
            if breached_net:
                warnings.append(
                    f"group_net_cap_applied groups={breached_net}, "
                    f"fraction={group_net_cap_fraction}, mode={constraints_mode}"
                )

        positions = self._client.get_all_positions()
        current = {t: 0.0 for t in fin_ts.ticker_list}
        for p in positions:
            sym = str(p.symbol)
            if sym in current:
                current[sym] = _signed_position_usd(p)

        if turnover_budget_fraction is not None:
            targets, obs_turnover, turn_limit = enforce_turnover_budget(
                targets,
                current,
                max_turnover_fraction=float(turnover_budget_fraction),
                on_breach=constraints_mode,
            )
            if obs_turnover > turn_limit + 1e-9:
                warnings.append(
                    f"turnover_budget_applied observed={obs_turnover:.2f} "
                    f"limit={turn_limit:.2f}, mode={constraints_mode}"
                )

        deltas = broker_deltas(targets, current, fin_ts.ticker_list)
        if adv_participation_fraction is not None:
            adv_usd = {}
            for t in fin_ts.ticker_list:
                key = (t, dt)
                if key not in df.index:
                    continue
                row = df.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                close = float(row.get("Close", np.nan))
                vol = float(row.get("Volume", np.nan))
                if np.isfinite(close) and np.isfinite(vol) and close > 0 and vol >= 0:
                    adv_usd[t] = close * vol
            deltas, breached_adv = cap_deltas_by_adv(
                deltas,
                adv_usd,
                max_adv_fraction=float(adv_participation_fraction),
                on_breach=constraints_mode,
            )
            if breached_adv:
                warnings.append(
                    f"adv_cap_applied symbols={breached_adv}, "
                    f"fraction={adv_participation_fraction}, mode={constraints_mode}"
                )

        if risk_policy is not None and hasattr(self._adapter, "submit_orders"):
            prices: dict[str, float] = {}
            for t in fin_ts.ticker_list:
                key = (t, dt)
                if key not in df.index:
                    continue
                row = df.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                c = float(row.get("Close", np.nan))
                if np.isfinite(c) and c > 0:
                    prices[t] = c
            order_specs = OrderBuilder.build_many(
                deltas,
                prices,
                risk_policy,
                order_type=order_type,
                variety=order_variety,
                exchange=order_exchange,
                product=order_product,
                tag_prefix=correlation_id,
                min_order_notional=min_order_notional,
            )
            attempts = self._adapter.submit_orders(
                order_specs,
                dry_run=dry_run,
                correlation_id=correlation_id,
            )
        else:
            attempts = self._adapter.submit_delta_orders(
                deltas,
                min_order_notional=min_order_notional,
                dry_run=dry_run,
                correlation_id=correlation_id,
            )
        if observe_order_status and not dry_run:
            attempts = self._adapter.observe_submitted_orders(
                attempts,
                max_polls=status_max_polls,
                poll_interval_seconds=status_poll_interval_seconds,
            )

        partial = [a.symbol for a in attempts if a.final_status == "partially_filled"]
        unresolved = [
            a.symbol
            for a in attempts
            if a.success
            and a.final_status in (None, "new", "accepted", "pending_new")
        ]
        status_errors = [a.symbol for a in attempts if a.status_error]
        if partial:
            warnings.append(f"partially_filled_orders={partial}")
        if unresolved:
            warnings.append(f"unresolved_order_status={unresolved}")
        if status_errors:
            warnings.append(f"order_status_poll_errors={status_errors}")

        post_trade_current: dict[str, float] = {}
        residual_deltas: dict[str, float] = {}
        remediation_attempts = []
        if reconcile_after_submit and not dry_run:
            try:
                post_positions = self._client.get_all_positions()
                post_trade_current = {t: 0.0 for t in fin_ts.ticker_list}
                for p in post_positions:
                    sym = str(p.symbol)
                    if sym in post_trade_current:
                        post_trade_current[sym] = _signed_position_usd(p)
                residual_deltas = broker_deltas(targets, post_trade_current, fin_ts.ticker_list)
                residual_deltas = {
                    k: v for k, v in residual_deltas.items() if abs(float(v)) >= reconciliation_tolerance_notional
                }
                if residual_deltas:
                    warnings.append(
                        f"reconciliation_residuals symbols={sorted(residual_deltas)}, "
                        f"policy={reconciliation_policy}"
                    )
                    if reconciliation_policy == "retry_once":
                        remediation_attempts = self._adapter.submit_delta_orders(
                            residual_deltas,
                            min_order_notional=max(min_order_notional, reconciliation_tolerance_notional),
                            dry_run=False,
                            correlation_id=f"{correlation_id}-recon",
                        )
                    elif reconciliation_policy == "cancel_and_retarget":
                        self._adapter.cancel_open_orders()
                        remediation_attempts = self._adapter.submit_delta_orders(
                            residual_deltas,
                            min_order_notional=max(min_order_notional, reconciliation_tolerance_notional),
                            dry_run=False,
                            correlation_id=f"{correlation_id}-recon",
                        )
                    if remediation_attempts and observe_order_status:
                        remediation_attempts = self._adapter.observe_submitted_orders(
                            remediation_attempts,
                            max_polls=status_max_polls,
                            poll_interval_seconds=status_poll_interval_seconds,
                        )
            except Exception as e:
                warnings.append(f"reconciliation_failed error={e}")

        eff_gross = sum(abs(v) for v in targets.values())

        return ExecutionReport(
            correlation_id=correlation_id,
            as_of=dt,
            data_source=data_source,
            panel_tickers=list(names),
            targets_usd=targets,
            current_usd=current,
            deltas_usd=deltas,
            tradecapital_requested=float(tradecapital),
            tradecapital_effective_gross=eff_gross,
            buying_power=buying_power_obs,
            clock_is_open=clock_open,
            dry_run=dry_run,
            order_attempts=attempts,
            remediation_attempts=remediation_attempts,
            warnings=warnings,
            status_observation_enabled=bool(observe_order_status and not dry_run),
            reconciliation_enabled=bool(reconcile_after_submit and not dry_run),
            reconciliation_policy=reconciliation_policy if (reconcile_after_submit and not dry_run) else None,
            post_trade_current_usd=post_trade_current,
            residual_deltas_usd=residual_deltas,
        )
