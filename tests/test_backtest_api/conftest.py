from __future__ import annotations

import os
from collections.abc import Generator

import pytest


def _yield_timescale_container_url() -> Generator[str, None, None]:
    pytest.importorskip("psycopg")
    from testcontainers.postgres import PostgresContainer

    try:
        with PostgresContainer("timescale/timescaledb:latest-pg16") as postgres:
            url = postgres.get_connection_url()
            if "+psycopg2" in url:
                url = url.replace("postgresql+psycopg2", "postgresql", 1)
            yield url
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker / testcontainer unavailable: {exc}")


@pytest.fixture(scope="module")
def api_database_url() -> str:
    """Postgres URL for API integration tests (local or testcontainers Timescale)."""
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if dsn:
        yield str(dsn)
        return
    if os.environ.get("SHUNYA_RUN_TIMESCALE_CONTAINER") != "1":
        pytest.skip(
            "Set DATABASE_URL / SHUNYA_DATABASE_URL, or SHUNYA_RUN_TIMESCALE_CONTAINER=1 for API DB tests."
        )
    yield from _yield_timescale_container_url()


@pytest.fixture(scope="module")
def api_database_url_queue_isolated() -> str:
    """Postgres URL for tests that claim ``api_backtest_jobs``.

    ``claim_next_queued_job`` competes with any other process using the same DSN (e.g. a
    local ``uvicorn`` worker). This fixture avoids shared ``DATABASE_URL`` whenever
    ``SHUNYA_RUN_TIMESCALE_CONTAINER=1`` so tests get a dedicated container.

    Override with ``SHUNYA_API_INTEGRATION_DATABASE_URL`` to point at any dedicated database.
    """
    dedicated = os.environ.get("SHUNYA_API_INTEGRATION_DATABASE_URL")
    if dedicated:
        yield str(dedicated).strip()
        return
    if os.environ.get("SHUNYA_RUN_TIMESCALE_CONTAINER") == "1":
        yield from _yield_timescale_container_url()
        return
    if os.environ.get("SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS") == "1":
        shared = os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
        if shared:
            yield str(shared)
            return
    pytest.skip(
        "Queue-isolated API integration requires a dedicated Postgres URL. "
        "Set SHUNYA_API_INTEGRATION_DATABASE_URL to a database no other API worker uses, "
        "or set SHUNYA_RUN_TIMESCALE_CONTAINER=1 (with Docker) to spin an isolated "
        "Timescale container. As a last resort only when no other worker uses the DB, "
        "set SHUNYA_TRUST_SHARED_DATABASE_FOR_QUEUE_TESTS=1 with DATABASE_URL."
    )
