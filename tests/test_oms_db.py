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
