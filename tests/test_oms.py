"""Unit tests for OMS reconciliation and parent FSM."""

from __future__ import annotations

from datetime import datetime, timezone

from shunya.oms.fills import ExecutionFill, vwap_avg_price
from shunya.oms.ledger_memory import InMemoryLedger
from shunya.oms.parent_fsm import ParentOrder
from shunya.oms.reconciliation import required_delta_shares, usd_targets_to_share_targets
from shunya.oms.service import InstitutionalOMS


def test_required_delta_shares_with_working() -> None:
    d = required_delta_shares(
        {"AAPL": 100.0},
        {"AAPL": 40.0},
        {"AAPL": 50.0},
        {"AAPL": 0.0},
        ["AAPL"],
    )
    assert abs(d["AAPL"] - 10.0) < 1e-9


def test_usd_targets_to_share_targets() -> None:
    t = usd_targets_to_share_targets({"A": 1000.0}, {"A": 200.0}, ["A", "B"])
    assert abs(t["A"] - 5.0) < 1e-9
    assert t["B"] == 0.0


def test_parent_order_fsm() -> None:
    p = ParentOrder("x", "MSFT", "SELL", 50)
    p.broker_accept()
    p.apply_fill(20)
    assert p.state == "PARTIAL_FILL"
    p.apply_fill(30)
    assert p.state == "FILLED"


def test_vwap_avg_price() -> None:
    ts = datetime.now(timezone.utc)
    fills = [
        ExecutionFill("1", "p", "A", "BUY", 10, 100.0, 0.0, ts),
        ExecutionFill("2", "p", "A", "BUY", 10, 120.0, 0.0, ts),
    ]
    assert abs(vwap_avg_price(fills) - 110.0) < 1e-9


def test_ledger_append_fill_idempotent() -> None:
    led = InMemoryLedger()
    p = ParentOrder("p1", "AAPL", "BUY", 100)
    led.register_parent(p)
    p.broker_accept()
    ts = datetime.now(timezone.utc)
    f = ExecutionFill("tr1", "p1", "AAPL", "BUY", 10, 1.0, 0.0, ts)
    led.append_fill(f)
    led.append_fill(f)
    assert len(led.fills) == 1
    assert p.quantity_filled == 10


def test_institutional_oms_propose_intents() -> None:
    oms = InstitutionalOMS()
    oms.refresh_settled_from_broker({"AAPL": 0.0})
    intents = oms.propose_parent_intents(
        {"AAPL": 10_000.0},
        {"AAPL": 100.0},
        ["AAPL"],
    )
    assert len(intents) == 1
    assert intents[0].side == "BUY"
    assert intents[0].quantity >= 1
