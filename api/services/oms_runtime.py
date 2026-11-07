"""Process-wide OMS singleton for manually placed orders (separate from the paper-cycle desk).

``InstitutionalPaperDesk`` still builds its own throwaway :class:`InstitutionalOMS` per
paper cycle (see ``shunya/live/desk.py``); this module builds a second, persistent
instance for the manual order-entry / blotter surface. Both subscribe independently to
Alpaca trade updates; :meth:`InstitutionalOMS.apply_alpaca_trade_update` already no-ops
for parent ids it doesn't own, so the two ledgers never cross-contaminate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from shunya.integration.alpaca_settings import AlpacaRuntimeSettings
from shunya.oms.fills import ExecutionFill, utc_now
from shunya.oms.parent_fsm import ParentOrder
from shunya.oms.service import InstitutionalOMS, ParentIntent

from api.services.oms_event_hub import OrderEventHub

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient

    from shunya.oms.alpaca_stream import AlpacaOMSTradeStream
    from shunya.oms.db.repository import OMSRepository

_log = logging.getLogger(__name__)


def _parent_to_event_dict(p: ParentOrder) -> dict:
    """Serialize parent order fields for SSE/WebSocket event payloads."""
    return {
        "parent_id": p.parent_id,
        "symbol": p.symbol,
        "side": p.side,
        "quantity_ordered": p.quantity_ordered,
        "quantity_filled": p.quantity_filled,
        "state": p.state,
        "client_order_id": p.client_order_id,
    }


def _fill_to_event_dict(f: ExecutionFill) -> dict:
    """Serialize fill fields with ISO timestamp for event payloads."""
    return {
        "trade_id": f.trade_id,
        "parent_order_id": f.parent_order_id,
        "symbol": f.symbol,
        "side": f.side,
        "quantity": f.quantity,
        "price": f.price,
        "fee": f.fee,
        "ts": f.ts.isoformat(),
        "child_client_order_id": f.child_client_order_id,
    }


class PersistingOMS(InstitutionalOMS):
    """:class:`InstitutionalOMS` that mirrors parent/fill state to Postgres and the event hub."""

    def __init__(
        self,
        *,
        repo: Optional["OMSRepository"],
        hub: OrderEventHub,
        **kwargs: object,
    ) -> None:
        """Store optional Postgres repo, event hub, and broker id map on the OMS."""
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._repo = repo
        self._hub = hub
        self.broker_order_id_by_parent: dict[str, str] = {}

    def set_broker_order_id(self, parent_id: str, broker_order_id: str) -> None:
        """Remember Alpaca broker order id for a parent."""
        self.broker_order_id_by_parent[parent_id] = broker_order_id

    def get_broker_order_id(self, parent_id: str) -> Optional[str]:
        """Look up stored Alpaca broker order id for a parent."""
        return self.broker_order_id_by_parent.get(parent_id)

    def _persist_parent(self, p: ParentOrder, *, origin: Optional[str] = None) -> None:
        """Upsert parent snapshot to Postgres when a repo is configured."""
        if self._repo is None:
            return
        try:
            self._repo.upsert_parent_snapshot(
                parent_id=p.parent_id,
                symbol=p.symbol,
                side=p.side,
                quantity_ordered=p.quantity_ordered,
                quantity_filled=p.quantity_filled,
                state=p.state,
                client_order_id=p.client_order_id,
                created_at=utc_now(),
                origin=origin,
                broker_order_id=self.broker_order_id_by_parent.get(p.parent_id),
            )
        except Exception:  # noqa: BLE001
            _log.exception("Failed to persist OMS parent snapshot for %s", p.parent_id)

    def create_parent_order(self, intent: ParentIntent) -> ParentOrder:
        """Create parent in ledger, persist, and publish an order event."""
        p = super().create_parent_order(intent)
        self._persist_parent(p, origin="manual")
        self._hub.publish_threadsafe({"type": "order", "order": _parent_to_event_dict(p)})
        return p

    def mark_parent_working(self, parent_id: str, client_order_id: Optional[str] = None) -> None:
        """Mark parent working, persist, and publish updated order state."""
        super().mark_parent_working(parent_id, client_order_id)
        p = self.ledger.get_parent(parent_id)
        if p is not None:
            self._persist_parent(p)
            self._hub.publish_threadsafe({"type": "order", "order": _parent_to_event_dict(p)})

    def record_fill(self, fill: ExecutionFill) -> None:
        """Record fill in ledger, persist fill and parent, and publish fill event."""
        if fill.quantity != int(fill.quantity):
            _log.warning(
                "Fractional fill quantity %s for parent %s; ledger tracks whole shares, "
                "DB record keeps exact quantity",
                fill.quantity,
                fill.parent_order_id,
            )
        super().record_fill(fill)
        if self._repo is not None:
            try:
                self._repo.insert_fill_idempotent(fill)
            except Exception:  # noqa: BLE001
                _log.exception("Failed to persist OMS fill %s", fill.trade_id)
        p = self.ledger.get_parent(fill.parent_order_id)
        if p is not None:
            self._persist_parent(p)
        self._hub.publish_threadsafe({"type": "fill", "fill": _fill_to_event_dict(fill)})

    def apply_alpaca_trade_update(self, data: object, *, fee: float = 0.0) -> None:
        """Apply Alpaca trade update then persist and publish parent state when owned."""
        from shunya.oms.stream_events import parent_id_from_client_order_id, normalize_alpaca_trade_update

        evt = normalize_alpaca_trade_update(data)
        pid = parent_id_from_client_order_id(evt.client_order_id)
        super().apply_alpaca_trade_update(data, fee=fee)
        if not pid:
            return
        p = self.ledger.get_parent(pid)
        if p is not None:
            self._persist_parent(p)
            self._hub.publish_threadsafe({"type": "order", "order": _parent_to_event_dict(p)})


@dataclass(slots=True)
class OMSRuntime:
    """Process-wide manual-order OMS: singleton ledger + persistence + Alpaca stream + event hub."""

    oms: PersistingOMS
    repo: Optional["OMSRepository"]
    stream: "AlpacaOMSTradeStream"
    hub: OrderEventHub
    paper: bool


def _sqlalchemy_psycopg_url(raw: str) -> str:
    """``api.db.resolve_database_url()`` returns a plain ``postgres(ql)://`` URL for
    ``psycopg.connect``; SQLAlchemy's engine needs the explicit ``+psycopg`` dialect."""
    if raw.startswith("postgresql+psycopg://") or raw.startswith("postgresql+psycopg2://"):
        return raw
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://") :]
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://") :]
    return raw


def _try_build_repo() -> Optional["OMSRepository"]:
    """Create OMS Postgres repository or log and return ``None`` when unavailable."""
    try:
        from api.db import resolve_database_url
    except Exception:  # noqa: BLE001
        _log.warning("api.db.resolve_database_url unavailable; OMS persistence disabled")
        return None
    try:
        url = _sqlalchemy_psycopg_url(resolve_database_url())
    except Exception as exc:  # noqa: BLE001
        _log.warning("No database configured for OMS persistence (%s); running in-memory only", exc)
        return None
    try:
        from shunya.oms.db.repository import OMSRepository, create_all, create_engine

        engine = create_engine(url)
        create_all(engine)
        return OMSRepository(engine)
    except Exception:  # noqa: BLE001
        _log.exception("Failed to initialize OMS Postgres repository; running in-memory only")
        return None


def build_oms_runtime(alpaca_settings: AlpacaRuntimeSettings) -> OMSRuntime:
    """Wire PersistingOMS, optional repo, Alpaca trade stream, and event hub."""
    from shunya.oms.alpaca_stream import AlpacaOMSTradeStream

    repo = _try_build_repo()
    hub = OrderEventHub()
    oms = PersistingOMS(repo=repo, hub=hub)
    stream = AlpacaOMSTradeStream(settings=alpaca_settings, oms=oms)
    return OMSRuntime(oms=oms, repo=repo, stream=stream, hub=hub, paper=alpaca_settings.paper)


def rehydrate(runtime: OMSRuntime, trading_client: "TradingClient") -> None:
    """Reload open parents from Postgres, replay their FSM state, then reconcile vs Alpaca REST."""
    repo = runtime.repo
    ledger = runtime.oms.ledger
    if repo is not None:
        try:
            rows = repo.list_open_parents()
        except Exception:  # noqa: BLE001
            _log.exception("Failed to list open OMS parents at startup; skipping rehydrate")
            rows = []
        for row in rows:
            try:
                p = ParentOrder(
                    parent_id=row.id,
                    symbol=row.symbol,
                    side=row.side,
                    quantity_ordered=row.quantity_ordered,
                )
                p.client_order_id = row.client_order_id
                if row.state != "PENDING_SUBMIT":
                    p.broker_accept()
                if row.quantity_filled > 0:
                    p.apply_fill(row.quantity_filled)
                ledger.register_parent(p)
                if row.broker_order_id:
                    runtime.oms.set_broker_order_id(row.id, row.broker_order_id)
            except Exception:  # noqa: BLE001
                _log.exception("Failed to rehydrate OMS parent %s; skipping", row.id)
                continue

            broker_order_id = row.broker_order_id
            if broker_order_id and not p.is_terminal():
                try:
                    obs = trading_client.get_order_by_id(broker_order_id)
                    status = str(getattr(obs, "status", "") or "").lower()
                    filled_qty = float(getattr(obs, "filled_qty", 0.0) or 0.0)
                    if filled_qty > p.quantity_filled:
                        p.apply_fill(int(filled_qty) - p.quantity_filled)
                    if status in ("canceled", "cancelled") and not p.is_terminal():
                        p.cancel()
                    elif status == "rejected" and not p.is_terminal():
                        p.broker_reject()
                    runtime.oms._persist_parent(p)  # noqa: SLF001
                except Exception:  # noqa: BLE001
                    _log.exception(
                        "Failed to reconcile OMS parent %s against Alpaca REST at startup", row.id
                    )

    try:
        runtime.oms.refresh_settled_shares_from_alpaca(trading_client)
    except Exception:  # noqa: BLE001
        _log.exception("Failed to refresh settled shares from Alpaca at OMS startup")
