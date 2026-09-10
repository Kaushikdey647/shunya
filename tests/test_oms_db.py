"""Integration tests for OMS SQL persistence (requires Docker for testcontainers)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from shunya.oms.db.repository import OMSRepository, create_all, create_engine
from shunya.oms.fills import ExecutionFill


def _dockerish() -> bool:
    return os.path.exists("/var/run/docker.sock") or os.environ.get("DOCKER_HOST")


@pytest.fixture(scope="module")
def pg_url() -> str:
    if not _dockerish():
        pytest.skip("Docker socket / DOCKER_HOST not available for testcontainers")
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not installed")
    try:
        with PostgresContainer("postgres:16-alpine") as c:
            raw = c.get_connection_url()
            if raw.startswith("postgresql://"):
                return "postgresql+psycopg://" + raw[len("postgresql://") :]
            return raw
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not start postgres container: {exc}")


@pytest.fixture(scope="module")
def repo(pg_url: str) -> OMSRepository:
    engine = create_engine(pg_url)
    create_all(engine)
    return OMSRepository(engine)


def test_insert_fill_idempotent(repo: OMSRepository) -> None:
    ts = datetime.now(timezone.utc)
    repo.upsert_parent_snapshot(
        parent_id="p1",
        symbol="AAPL",
        side="BUY",
        quantity_ordered=100,
        quantity_filled=0,
        state="WORKING",
        client_order_id="cid-1",
        created_at=ts,
    )
    fill = ExecutionFill(
        trade_id="t-unique-1",
        parent_order_id="p1",
        symbol="AAPL",
        side="BUY",
        quantity=10.0,
        price=150.0,
        fee=0.01,
        ts=ts,
        child_client_order_id="p1:0",
    )
    assert repo.insert_fill_idempotent(fill) is True
    assert repo.insert_fill_idempotent(fill) is False
    fills = repo.list_fills_for_parent("p1")
    assert len(fills) == 1
    assert fills[0].trade_id == "t-unique-1"


def test_upsert_parent_snapshot_persists_manual_order_fields(repo: OMSRepository) -> None:
    ts = datetime.now(timezone.utc)
    repo.upsert_parent_snapshot(
        parent_id="p-manual-1",
        symbol="MSFT",
        side="BUY",
        quantity_ordered=50,
        quantity_filled=0,
        state="PENDING_SUBMIT",
        client_order_id="p-manual-1:0",
        created_at=ts,
        order_type="LIMIT",
        limit_price=410.5,
        time_in_force="DAY",
        origin="manual",
    )
    row = repo.get_parent("p-manual-1")
    assert row is not None
    assert row.order_type == "LIMIT"
    assert row.limit_price == 410.5
    assert row.origin == "manual"
    assert row.broker_order_id is None

    repo.upsert_parent_snapshot(
        parent_id="p-manual-1",
        symbol="MSFT",
        side="BUY",
        quantity_ordered=50,
        quantity_filled=50,
        state="FILLED",
        client_order_id="p-manual-1:0",
        created_at=ts,
        broker_order_id="broker-abc",
    )
    row = repo.get_parent("p-manual-1")
    assert row is not None
    assert row.state == "FILLED"
    assert row.quantity_filled == 50
    assert row.broker_order_id == "broker-abc"
    # Fields not passed on the update call are preserved, not cleared.
    assert row.order_type == "LIMIT"
    assert row.limit_price == 410.5


def test_rehydrate_replays_parent_state_from_postgres(repo: OMSRepository) -> None:
    """Persist → rehydrate → state-equality round trip (mirrors api.services.oms_runtime.rehydrate)."""
    from shunya.oms.ledger_memory import InMemoryLedger
    from shunya.oms.parent_fsm import ParentOrder

    ts = datetime.now(timezone.utc)
    repo.upsert_parent_snapshot(
        parent_id="p-rehydrate-1",
        symbol="AAPL",
        side="SELL",
        quantity_ordered=100,
        quantity_filled=40,
        state="PARTIAL_FILL",
        client_order_id="p-rehydrate-1:0",
        created_at=ts,
        broker_order_id="broker-xyz",
        origin="manual",
    )

    ledger = InMemoryLedger()
    rows = repo.list_open_parents()
    row = next(r for r in rows if r.id == "p-rehydrate-1")

    p = ParentOrder(
        parent_id=row.id, symbol=row.symbol, side=row.side, quantity_ordered=row.quantity_ordered
    )
    p.client_order_id = row.client_order_id
    if row.state != "PENDING_SUBMIT":
        p.broker_accept()
    if row.quantity_filled > 0:
        p.apply_fill(row.quantity_filled)
    ledger.register_parent(p)

    rehydrated = ledger.get_parent("p-rehydrate-1")
    assert rehydrated is not None
    assert rehydrated.state == "PARTIAL_FILL"
    assert rehydrated.quantity_filled == 40
    assert rehydrated.working_shares == 60
