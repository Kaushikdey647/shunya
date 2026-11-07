"""In-memory fakes for the live runner: Alpaca-like broker and PostgresLiveStore stand-in."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional

from shunya.live.broker import TZ_NY, AccountView, KillSwitchEngaged, OrderView, PositionView, Session
from shunya.live.store import Book, Control, OrderRow, Run, Sleeve, StrategyVersion


def ny(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=TZ_NY)


def calendar(start: date, end: date) -> List[Session]:
    """Weekdays; Thanksgiving 2026-11-26 closed, 2026-11-27 closes 13:00."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d != date(2026, 11, 26):
            close = time(13) if d == date(2026, 11, 27) else time(16)
            out.append(Session(d, ny(d, 9, 30), datetime.combine(d, close, tzinfo=TZ_NY)))
        d += timedelta(days=1)
    return out


class FakeBroker:
    """Alpaca-like broker; ``fill_all`` simulates the opening auction."""

    def __init__(self, control_fn, *, equity: float = 100_000.0) -> None:
        self.control_fn = control_fn
        self.equity = equity
        self.pos: Dict[str, float] = {}
        self.price: Dict[str, float] = {}
        self.orders: Dict[str, OrderView] = {}
        self.submitted: List[str] = []
        self.cancelled = 0
        self.after_submit = None  # hook(broker) called after each submit

    def sessions(self, start: date, end: date) -> List[Session]:
        return calendar(start, end)

    def clock(self) -> dict:
        return {"timestamp": "2026-11-03T20:00:00-05:00", "is_open": False,
                "next_open": "2026-11-04T09:30:00-05:00", "next_close": "2026-11-04T16:00:00-05:00"}

    def account(self) -> AccountView:
        return AccountView(self.equity, self.equity, self.equity, 2 * self.equity, False, "ACTIVE")

    def positions(self) -> List[PositionView]:
        return [
            PositionView(s, q, q * self.price.get(s, 0.0), self.price.get(s, 0.0), self.price.get(s, 0.0))
            for s, q in self.pos.items()
            if q
        ]

    def open_orders(self) -> List[OrderView]:
        return [o for o in self.orders.values() if o.status in ("accepted", "new")]

    def order_by_client_id(self, coid: str) -> Optional[OrderView]:
        return self.orders.get(coid)

    def submit_opg(self, *, symbol: str, side: str, qty: int, client_order_id: str) -> OrderView:
        ctl = self.control_fn()
        if ctl.halted:
            raise KillSwitchEngaged(ctl.reason or "halted")
        if client_order_id in self.orders:
            return self.orders[client_order_id]
        v = OrderView(client_order_id, uuid.uuid4().hex, symbol, side, qty, "accepted", 0.0, None, "opg")
        self.orders[client_order_id] = v
        self.submitted.append(client_order_id)
        if self.after_submit:
            self.after_submit(self)
        return v

    def fill_all(self) -> None:
        """Simulate the opening auction filling every working order at the last price."""
        for k, o in list(self.orders.items()):
            if o.status == "accepted":
                self.orders[k] = replace(o, status="filled", filled_qty=o.qty, filled_avg_price=self.price[o.symbol])
                self.pos[o.symbol] = self.pos.get(o.symbol, 0.0) + (o.qty if o.side == "buy" else -o.qty)

    def cancel_all(self) -> int:
        n = len(self.open_orders())
        for k, o in list(self.orders.items()):
            if o.status == "accepted":
                self.orders[k] = replace(o, status="canceled")
        self.cancelled += n
        return n

    def close_all(self) -> int:
        n = len([q for q in self.pos.values() if q])
        self.pos.clear()
        return n


class MemStore:
    """In-memory stand-in for PostgresLiveStore (same method surface used by the runner)."""

    def __init__(self) -> None:
        self.control = Control(state="armed")
        self.audits: List[tuple] = []
        self.heartbeat: Optional[str] = None
        self.books: Dict[str, Book] = {}
        self.sleeves: Dict[str, Sleeve] = {}
        self.runs: Dict[str, Run] = {}
        self.targets: Dict[str, List[Dict[str, Any]]] = {}
        self.orders: Dict[str, OrderRow] = {}
        self.ledger: Dict[str, Dict[str, float]] = {}
        self.snapshots: Dict[tuple, float] = {}
        self.fail_control = False
        self.versions: Dict[str, StrategyVersion] = {}

    def get_control(self) -> Control:
        if self.fail_control:
            raise ConnectionError("db down")
        return self.control

    def set_control(self, *, state, level, reason, actor) -> None:
        self.control = Control(state=state, level=level, reason=reason, actor=actor)

    def audit(self, actor, action, detail=None) -> None:
        self.audits.append((actor, action, detail))

    def write_heartbeat(self, result: str) -> None:
        self.heartbeat = result

    def list_books(self, *, active_only=False) -> List[Book]:
        return [b for b in self.books.values() if not active_only or b.status == "active"]

    def update_book_equity(self, book_id, *, equity, hwm) -> None:
        self.books[book_id] = replace(self.books[book_id], last_equity=equity, hwm_equity=hwm)

    def list_sleeves(self, book_id) -> List[Sleeve]:
        return [s for s in self.sleeves.values() if s.book_id == book_id and s.status != "retired"]

    def increment_sleeve_runs(self, ids) -> None:
        for i in ids:
            self.sleeves[i] = replace(self.sleeves[i], runs_completed=self.sleeves[i].runs_completed + 1)

    def claim_run(self, *, book_id, session_date, signal_date) -> Optional[str]:
        for r in self.runs.values():
            if r.book_id == book_id and r.session_date == session_date:
                if r.status != "failed":
                    return None
                self.runs[r.id] = replace(r, status="running")
                return r.id
        rid = uuid.uuid4().hex
        self.runs[rid] = Run(rid, book_id, session_date, signal_date, "running")
        return rid

    def claim_reconcile(self, run_id) -> bool:
        if self.runs[run_id].status != "submitted":
            return False
        self.runs[run_id] = replace(self.runs[run_id], status="reconciling")
        return True

    def update_run(self, run_id, *, status, error=None, equity=None, detail=None) -> None:
        r = self.runs[run_id]
        self.runs[run_id] = replace(r, status=status, error=error or r.error, detail={**r.detail, **(detail or {})})

    def list_runs(self, *, book_id=None, status=None, limit=30) -> List[Run]:
        return [r for r in self.runs.values() if (not book_id or r.book_id == book_id) and (not status or r.status == status)]

    def save_sleeve_targets(self, run_id, rows) -> None:
        self.targets[run_id] = [dict(r) for r in rows]

    def list_sleeve_targets(self, run_id):
        return self.targets.get(run_id, [])

    def upsert_order(self, o: OrderRow) -> None:
        self.orders[o.client_order_id] = o

    def list_orders(self, run_id) -> List[OrderRow]:
        return [o for o in self.orders.values() if o.run_id == run_id]

    def get_sleeve_positions(self, book_id) -> Dict[str, Dict[str, float]]:
        ids = {s.id for s in self.sleeves.values() if s.book_id == book_id}
        return {k: dict(v) for k, v in self.ledger.items() if k in ids}

    def set_sleeve_positions(self, sleeve_id, shares) -> None:
        self.ledger[sleeve_id] = {k: v for k, v in shares.items() if abs(v) > 1e-9}

    def save_snapshot(self, sleeve_id, session_date, market_value) -> None:
        self.snapshots[(sleeve_id, session_date)] = market_value

    # --- extra surface used by the hosted API routes ---

    def get_heartbeat(self) -> Dict[str, Any]:
        return {"last_tick_at": None, "last_result": self.heartbeat}

    def list_audit(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [{"actor": a, "action": b, "detail": c} for a, b, c in reversed(self.audits)][:limit]

    def create_version(self, *, name, alpha_source, finstrat_config, tickers, lookback_days) -> str:
        vid = uuid.uuid4().hex
        self.versions[vid] = StrategyVersion(vid, name, alpha_source, dict(finstrat_config), list(tickers), lookback_days)
        return vid

    def list_versions(self) -> List[Dict[str, Any]]:
        return [{"id": v.id, "name": v.name, "n_tickers": len(v.tickers)} for v in self.versions.values()]

    def create_book(self, *, name, capital_fraction, risk_config, status="active") -> str:
        bid = str(uuid.uuid4())
        self.books[bid] = Book(bid, name, status, capital_fraction, dict(risk_config))
        return bid

    def get_book(self, book_id) -> Optional[Book]:
        return self.books.get(book_id)

    def update_book(self, book_id, **fields) -> None:
        self.books[book_id] = replace(self.books[book_id], **{k: v for k, v in fields.items() if v is not None})

    def create_sleeve(self, *, book_id, version_id, weight, ramp_days=1) -> str:
        sid = uuid.uuid4().hex
        self.sleeves[sid] = Sleeve(sid, book_id, self.versions[version_id], weight, "active", ramp_days, 0)
        return sid

    def update_sleeve(self, sleeve_id, **fields) -> None:
        self.sleeves[sleeve_id] = replace(self.sleeves[sleeve_id], **{k: v for k, v in fields.items() if v is not None})

    def get_run(self, run_id) -> Optional[Run]:
        return self.runs.get(run_id)
