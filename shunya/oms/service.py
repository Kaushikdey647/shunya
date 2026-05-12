"""High-level OMS: ingest vetted USD targets, reconcile in shares, register parents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

from shunya.algorithm.orders import RiskPolicy

from .fills import ExecutionFill
from .ledger_memory import InMemoryLedger
from .parent_fsm import ParentOrder
from .reconciliation import usd_targets_to_share_targets


if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient


@dataclass
class ParentIntent:
    """A synthetic parent order the EMS should execute (shares, side)."""

    parent_id: str
    symbol: str
    side: str
    quantity: int


class InstitutionalOMS:
    """
    Institutional order management on top of :class:`InMemoryLedger`.

    Converts risk-vetted *USD* targets to share targets using the same ``prices``
    snapshot, then applies lot sizing via :class:`~shunya.algorithm.orders.RiskPolicy`.
    """

    def __init__(
        self,
        ledger: Optional[InMemoryLedger] = None,
        *,
        risk_policy: Optional[RiskPolicy] = None,
    ) -> None:
        self._ledger = ledger or InMemoryLedger()
        self._risk_policy = risk_policy or RiskPolicy()

    @property
    def ledger(self) -> InMemoryLedger:
        return self._ledger

    def ingest_vetted_usd_targets(
        self,
        targets_usd: Mapping[str, float],
        prices: Mapping[str, float],
        universe: Sequence[str],
    ) -> Dict[str, float]:
        """Return floating share targets (pre-lot) for diagnostics."""
        return usd_targets_to_share_targets(targets_usd, prices, universe)

    def propose_parent_intents(
        self,
        targets_usd: Mapping[str, float],
        prices: Mapping[str, float],
        universe: Sequence[str],
        *,
        min_abs_delta_shares: float = 1e-6,
    ) -> List[ParentIntent]:
        """
        Reconcile ledger vs targets and return BUY/SELL parent intents (whole shares).

        Does not mutate the ledger; caller registers :class:`ParentOrder` rows and
        hands execution to EMS.
        """
        float_targets = usd_targets_to_share_targets(targets_usd, prices, universe)
        deltas = self._ledger.reconcile_deltas(float_targets, universe)
        intents: List[ParentIntent] = []
        for sym in universe:
            d = float(deltas.get(str(sym), 0.0))
            if abs(d) < min_abs_delta_shares:
                continue
            px = float(prices.get(str(sym), 0.0))
            if px <= 0.0:
                continue
            side = "BUY" if d > 0 else "SELL"
            delta_usd = abs(d) * px
            qty = self._risk_policy.compute_quantity(delta_usd if side == "BUY" else -delta_usd, px)
            if qty <= 0:
                continue
            intents.append(
                ParentIntent(
                    parent_id=str(uuid.uuid4()),
                    symbol=str(sym),
                    side=side,
                    quantity=int(qty),
                )
            )
        return intents

    def create_parent_order(self, intent: ParentIntent) -> ParentOrder:
        p = ParentOrder(
            parent_id=intent.parent_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity_ordered=intent.quantity,
        )
        self._ledger.register_parent(p)
        return p

    def mark_parent_working(self, parent_id: str, client_order_id: Optional[str] = None) -> None:
        p = self._ledger.get_parent(parent_id)
        if p is None:
            raise KeyError(parent_id)
        p.client_order_id = client_order_id
        p.broker_accept()

    def record_fill(self, fill: ExecutionFill) -> None:
        """Append-only fill; updates parent FSM when ``parent_order_id`` matches."""
        self._ledger.append_fill(fill)

    def refresh_settled_from_broker(self, positions_shares: Mapping[str, float]) -> None:
        self._ledger.merge_settled_from_broker(positions_shares)

    def refresh_settled_shares_from_alpaca(self, client: "TradingClient") -> None:
        """Pull settled position quantities from Alpaca REST (stream reconnect hygiene)."""
        from .rest_snapshot import snapshot_from_alpaca_client

        settled, _, _ = snapshot_from_alpaca_client(client)
        self.refresh_settled_from_broker(settled)

    def apply_alpaca_trade_update(self, data: object, *, fee: float = 0.0) -> None:
        """
        Apply a :class:`alpaca.trading.TradeUpdate` to the in-memory ledger / parent FSM.

        Fills are deduped by ``trade_id`` (``execution_id``). Unknown ``client_order_id``
        roots are ignored.
        """
        from .stream_events import (
            execution_fill_from_trade_update,
            normalize_alpaca_trade_update,
            parent_id_from_client_order_id,
        )

        evt = normalize_alpaca_trade_update(data)
        pid = parent_id_from_client_order_id(evt.client_order_id)
        if not pid:
            return
        parent = self._ledger.get_parent(pid)
        if parent is None:
            return
        fill = execution_fill_from_trade_update(data, parent_order_id=pid, fee=fee)
        if fill is not None:
            self.record_fill(fill)
        if evt.kind == "canceled":
            try:
                parent.cancel()
            except Exception:
                pass
        if evt.kind == "rejected":
            try:
                parent.broker_reject()
            except Exception:
                pass
