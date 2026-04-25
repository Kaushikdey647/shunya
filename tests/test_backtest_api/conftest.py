from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def api_database_url() -> str:
    """Postgres URL for API integration tests (local or testcontainers Timescale)."""
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if dsn:
        return str(dsn)
    if os.environ.get("SHUNYA_RUN_TIMESCALE_CONTAINER") != "1":
        pytest.skip(
            "Set DATABASE_URL / SHUNYA_DATABASE_URL, or SHUNYA_RUN_TIMESCALE_CONTAINER=1 for API DB tests."
        )
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
