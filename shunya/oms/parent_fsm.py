"""Parent order lifecycle as a formal finite state machine (``transitions``).

Fill aggregation does not use conditional transitions (``transitions`` evaluates
``conditions``/``unless`` before ``before`` callbacks), so fills use
:meth:`ParentOrder.apply_fill` and explicit ``set_state`` for FILLED/PARTIAL_FILL.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from transitions import Machine  # type: ignore[import-untyped]

TERMINAL_STATES: Tuple[str, ...] = ("FILLED", "CANCELED", "REJECTED")


def _unwired_fsm_trigger() -> bool:
    return False


@dataclass
class ParentOrder:
    """
    Parent intent (e.g. buy 100k shares) tracked by the OMS.

    Broker lifecycle edges use :class:`transitions.Machine`; fill quantity uses
    explicit logic for partial vs complete fills.
    """

    parent_id: str
    symbol: str
    side: str  # BUY | SELL
    quantity_ordered: int
    quantity_filled: int = 0
    client_order_id: Optional[str] = None
    _machine: Any = field(init=False, repr=False)
    broker_accept: Callable[[], bool] = field(
        init=False, repr=False, default_factory=lambda: _unwired_fsm_trigger
    )
    broker_reject: Callable[[], bool] = field(
        init=False, repr=False, default_factory=lambda: _unwired_fsm_trigger
    )
    cancel: Callable[[], bool] = field(
        init=False, repr=False, default_factory=lambda: _unwired_fsm_trigger
    )

    def __post_init__(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be BUY or SELL")
        if self.quantity_ordered <= 0:
            raise ValueError("quantity_ordered must be positive")
        transitions: List[dict[str, Any]] = [
            {"trigger": "broker_accept", "source": "PENDING_SUBMIT", "dest": "WORKING"},
            {"trigger": "broker_reject", "source": "PENDING_SUBMIT", "dest": "REJECTED"},
            {
                "trigger": "cancel",
                "source": ["PENDING_SUBMIT", "WORKING", "PARTIAL_FILL"],
                "dest": "CANCELED",
            },
            {"trigger": "broker_reject", "source": ["WORKING", "PARTIAL_FILL"], "dest": "REJECTED"},
        ]
        self._machine = Machine(
            model=self,
            states=[
                "PENDING_SUBMIT",
                "WORKING",
                "PARTIAL_FILL",
                "FILLED",
                "CANCELED",
                "REJECTED",
            ],
            transitions=transitions,
            initial="PENDING_SUBMIT",
            send_event=True,
            auto_transitions=False,
            ignore_invalid_triggers=False,
            model_attribute="machine_state",
        )

        def _broker_accept(m: ParentOrder) -> bool:
            return bool(m._machine.events["broker_accept"].trigger(m))

        def _broker_reject(m: ParentOrder) -> bool:
            return bool(m._machine.events["broker_reject"].trigger(m))

        def _cancel(m: ParentOrder) -> bool:
            return bool(m._machine.events["cancel"].trigger(m))

        object.__setattr__(self, "broker_accept", types.MethodType(_broker_accept, self))
        object.__setattr__(self, "broker_reject", types.MethodType(_broker_reject, self))
        object.__setattr__(self, "cancel", types.MethodType(_cancel, self))

    @property
    def state(self) -> str:
        return str(getattr(self, "machine_state", "PENDING_SUBMIT"))

    def apply_fill(self, qty: int) -> None:
        """Add executed quantity while in WORKING or PARTIAL_FILL; move to FILLED when done."""
        if qty <= 0:
            raise ValueError("fill qty must be positive")
        st = str(self.state)
        if st not in ("WORKING", "PARTIAL_FILL"):
            raise RuntimeError(f"cannot apply fill from state {st}")
        self.quantity_filled = min(self.quantity_ordered, self.quantity_filled + qty)
        if self.quantity_filled >= self.quantity_ordered:
            self._machine.set_state("FILLED", self)
        elif st == "WORKING":
            self._machine.set_state("PARTIAL_FILL", self)

    @property
    def working_shares(self) -> int:
        """Shares still working at the broker for this parent (best-effort)."""
        if self.state in TERMINAL_STATES:
            return 0
        return max(0, self.quantity_ordered - self.quantity_filled)

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
