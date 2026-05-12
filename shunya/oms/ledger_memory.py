"""In-memory OMS ledger for phase-1 development and tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence

from .fills import ExecutionFill
from .parent_fsm import TERMINAL_STATES, ParentOrder
from .reconciliation import required_delta_shares


@dataclass
class InMemoryLedger:
    """
    Authoritative settled position (shares) plus append-only fills and parent orders.

    Working exposure for reconciliation is derived from non-terminal parents only.
    """

    settled_shares: Dict[str, float] = field(default_factory=dict)
    fills: List[ExecutionFill] = field(default_factory=list)
    parents: Dict[str, ParentOrder] = field(default_factory=dict)

    def working_buy_sell(self) -> tuple[Dict[str, float], Dict[str, float]]:
        buy: DefaultDict[str, float] = defaultdict(float)
        sell: DefaultDict[str, float] = defaultdict(float)
        for p in self.parents.values():
            if p.state in TERMINAL_STATES:
                continue
            w = float(p.working_shares)
            if w <= 0:
                continue
            if p.side == "BUY":
                buy[p.symbol] += w
            else:
                sell[p.symbol] += w
        return dict(buy), dict(sell)

    def reconcile_deltas(
        self,
        target_shares: Mapping[str, float],
        universe: Sequence[str],
    ) -> Dict[str, float]:
        wb, ws = self.working_buy_sell()
        return required_delta_shares(target_shares, self.settled_shares, wb, ws, universe)

    def register_parent(self, parent: ParentOrder) -> None:
        if parent.parent_id in self.parents:
            raise KeyError(f"duplicate parent_id {parent.parent_id}")
        self.parents[parent.parent_id] = parent

    def get_parent(self, parent_id: str) -> Optional[ParentOrder]:
        return self.parents.get(parent_id)

    def append_fill(self, fill: ExecutionFill) -> None:
        if any(f.trade_id == fill.trade_id for f in self.fills):
            return
        self.fills.append(fill)
        parent = self.parents.get(fill.parent_order_id)
        if parent is None:
            return
        if fill.side != parent.side:
            return
        if parent.state == "PENDING_SUBMIT":
            parent.broker_accept()
        parent.apply_fill(int(fill.quantity))

    def merge_settled_from_broker(self, positions: Mapping[str, float]) -> None:
        for k, v in positions.items():
            self.settled_shares[str(k)] = float(v)
