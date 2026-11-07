"""Status snapshot for the dashboard: kill state, heartbeat, broker view, books, ledger drift."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, TypeVar

from shunya.live.broker import Broker
from shunya.live.control import read_control
from shunya.live.store import PostgresLiveStore

_log = logging.getLogger(__name__)
T = TypeVar("T")


def _safe(errors: List[str], label: str, fn: Callable[[], T], default: T) -> T:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — status must render even when a dependency is down
        errors.append(f"{label}: {exc}")
        return default


def build_status(store: PostgresLiveStore, broker: Broker) -> Dict[str, Any]:
    """Everything the local dashboard polls; partial results plus ``errors`` when a source fails."""
    errors: List[str] = []
    now = datetime.now(timezone.utc)
    control = read_control(store)
    hb = _safe(errors, "heartbeat", store.get_heartbeat, {})
    last_tick = hb.get("last_tick_at")
    account = _safe(errors, "account", lambda: asdict(broker.account()), None)
    positions = _safe(errors, "positions", broker.positions, [])
    open_orders = _safe(errors, "open_orders", broker.open_orders, [])
    clock = _safe(errors, "clock", broker.clock, None)
    pos_by_sym = {p.symbol: p for p in positions}

    books_out: List[Dict[str, Any]] = []
    tracked: Dict[str, float] = {}
    for b in _safe(errors, "books", store.list_books, []):
        sleeves = _safe(errors, f"sleeves:{b.name}", lambda b=b: store.list_sleeves(b.id), [])
        ledger = _safe(errors, f"ledger:{b.name}", lambda b=b: store.get_sleeve_positions(b.id), {})
        runs = _safe(errors, f"runs:{b.name}", lambda b=b: store.list_runs(book_id=b.id, limit=1), [])
        sl_out = []
        for s in sleeves:
            shares = ledger.get(s.id, {})
            for sym, q in shares.items():
                if b.status == "active":
                    tracked[sym] = tracked.get(sym, 0.0) + q
            mv = sum(q * pos_by_sym[sym].current_price for sym, q in shares.items() if sym in pos_by_sym)
            sl_out.append(
                {
                    "id": s.id,
                    "version_id": s.version.id,
                    "name": s.version.name,
                    "n_tickers": len(s.version.tickers),
                    "weight": s.weight,
                    "status": s.status,
                    "ramp_days": s.ramp_days,
                    "runs_completed": s.runs_completed,
                    "positions": shares,
                    "market_value": mv,
                }
            )
        books_out.append({**asdict(b), "sleeves": sl_out, "last_run": asdict(runs[0]) if runs else None})

    mismatches = {
        sym: {"ledger": q, "broker": pos_by_sym[sym].qty if sym in pos_by_sym else 0.0}
        for sym, q in tracked.items()
        if abs(q - (pos_by_sym[sym].qty if sym in pos_by_sym else 0.0)) > 0.5
    }
    untracked = [asdict(p) for p in positions if p.symbol not in tracked]
    return {
        "server_time": now.isoformat(),
        "control": asdict(control),
        "heartbeat": {
            "last_tick_at": last_tick.isoformat() if last_tick else None,
            "last_result": hb.get("last_result"),
            "age_seconds": (now - last_tick).total_seconds() if last_tick else None,
        },
        "clock": clock,
        "account": account,
        "positions": [asdict(p) for p in positions],
        "open_orders": [asdict(o) for o in open_orders],
        "books": books_out,
        "mismatches": mismatches,
        "untracked_positions": untracked,
        "errors": errors,
    }
