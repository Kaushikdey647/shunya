from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .execution import OrderAttempt
from .orders import (
    ExecutionAdapter,
    OpenOrderView,
    OrderBuilder,
    OrderType,
    OrderVariety,
    RiskPolicy,
)
from .targets import broker_deltas


@dataclass
class ManagedOrderBatch:
    """Result of a stateful target submission cycle."""

    targets_usd: Dict[str, float]
    current_usd: Dict[str, float]
    deltas_usd: Dict[str, float]
    skipped_symbols: List[str] = field(default_factory=list)
    open_orders: List[OpenOrderView] = field(default_factory=list)
    order_attempts: List[OrderAttempt] = field(default_factory=list)


class OrderManager:
    """
    Stateful OMS helper for target-to-order cycles.

    For institutional parent/child flows (FSM ledger, EMS slicing), prefer
    :class:`shunya.oms.service.InstitutionalOMS` plus :class:`shunya.ems.runner.EMSParentRunner`.

    The manager keeps a small cache of positions and open orders so that repeated
    target submissions can suppress duplicate submits while orders are still working.
    """

    def __init__(
        self,
        execution_adapter: ExecutionAdapter,
        *,
        risk_policy: Optional[RiskPolicy] = None,
        min_order_notional: float = 1.0,
        retarget_threshold_notional: float = 1.0,
        order_type: OrderType = OrderType.MARKET,
        order_variety: OrderVariety = OrderVariety.REGULAR,
        observe_order_status: bool = True,
        status_max_polls: int = 1,
        status_poll_interval_seconds: float = 0.0,
    ) -> None:
        self._adapter = execution_adapter
        self._risk_policy = risk_policy
        self._min_order_notional = float(min_order_notional)
        self._retarget_threshold = float(retarget_threshold_notional)
        self._order_type = order_type
        self._order_variety = order_variety
        self._observe_order_status = bool(observe_order_status)
        self._status_max_polls = int(status_max_polls)
        self._status_poll_interval_seconds = float(status_poll_interval_seconds)
        self._positions_cache: Dict[str, float] = {}
        self._open_orders_cache: List[OpenOrderView] = []
        self._last_submitted_targets: Dict[str, float] = {}

    def refresh_state(self) -> tuple[Dict[str, float], List[OpenOrderView]]:
        self._positions_cache = dict(self._adapter.get_positions())
        self._open_orders_cache = list(self._adapter.list_open_orders())
        return dict(self._positions_cache), list(self._open_orders_cache)

    def open_order_symbols(self) -> set[str]:
        return {order.symbol for order in self._open_orders_cache}

    def compute_deltas(self, targets: Dict[str, float]) -> tuple[Dict[str, float], List[str]]:
        current = {symbol: float(self._positions_cache.get(symbol, 0.0)) for symbol in targets}
        deltas = broker_deltas(targets, current, list(targets))
        skipped: List[str] = []
        blocked = self.open_order_symbols()
        for symbol in list(deltas):
            last_target = self._last_submitted_targets.get(symbol)
            if symbol in blocked:
                skipped.append(symbol)
                del deltas[symbol]
                continue
            if last_target is not None and abs(float(targets[symbol]) - last_target) < self._retarget_threshold:
                skipped.append(symbol)
                del deltas[symbol]
        return deltas, skipped

    def submit_targets(
        self,
        targets: Dict[str, float],
        *,
        prices: Dict[str, float],
        correlation_id: str,
        dry_run: bool = False,
    ) -> ManagedOrderBatch:
        current, open_orders = self.refresh_state()
        deltas, skipped = self.compute_deltas(targets)

        attempts: List[OrderAttempt]
        if self._risk_policy is not None:
            specs = OrderBuilder.build_many(
                deltas,
                prices,
                self._risk_policy,
                order_type=self._order_type,
                variety=self._order_variety,
                tag_prefix=correlation_id,
                min_order_notional=self._min_order_notional,
            )
            attempts = self._adapter.submit_orders(
                specs,
                dry_run=dry_run,
                correlation_id=correlation_id,
            )
        else:
            submit_delta = getattr(self._adapter, "submit_delta_orders", None)
            if submit_delta is None:
                raise TypeError(
                    "execution adapter does not expose submit_delta_orders; "
                    "pass a RiskPolicy to submit explicit order specs instead"
                )
            attempts = submit_delta(
                deltas,
                min_order_notional=self._min_order_notional,
                dry_run=dry_run,
                correlation_id=correlation_id,
            )

        if self._observe_order_status and hasattr(self._adapter, "observe_submitted_orders"):
            attempts = self._adapter.observe_submitted_orders(
                attempts,
                max_polls=self._status_max_polls,
                poll_interval_seconds=self._status_poll_interval_seconds,
            )

        for attempt in attempts:
            if attempt.success:
                self._last_submitted_targets[attempt.symbol] = float(targets.get(attempt.symbol, 0.0))

        return ManagedOrderBatch(
            targets_usd=dict(targets),
            current_usd=current,
            deltas_usd=deltas,
            skipped_symbols=sorted(set(skipped)),
            open_orders=open_orders,
            order_attempts=attempts,
        )

    def cancel_all_open_orders(self) -> None:
        self._adapter.cancel_open_orders()
        self._open_orders_cache = []
