"""Optional integration tests for Timescale read/write (Docker or DATABASE_URL)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.timescale


@pytest.fixture(scope="module")
def timescale_dsn() -> str:
    """Fresh Timescale image via testcontainers, or your local DATABASE_URL."""
    dsn = os.environ.get("DATABASE_URL") or os.environ.get("SHUNYA_DATABASE_URL")
    if dsn:
        return str(dsn)
    if os.environ.get("SHUNYA_RUN_TIMESCALE_CONTAINER") != "1":
        pytest.skip(
            "Set DATABASE_URL / SHUNYA_DATABASE_URL for a live DB, or "
            "SHUNYA_RUN_TIMESCALE_CONTAINER=1 to start a Timescale testcontainer (needs Docker)."
        )
    pytest.importorskip("psycopg")
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("timescale/timescaledb:latest-pg16") as postgres:
        url = postgres.get_connection_url()
        if "+psycopg2" in url:
            url = url.replace("postgresql+psycopg2", "postgresql", 1)
        yield url


def test_migrate_and_ohlcv_roundtrip(timescale_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg

    pytest.importorskip("psycopg")
    monkeypatch.setenv("DATABASE_URL", timescale_dsn)

    from shunya.data.timescale.dbutil import apply_migrations
    from shunya.data.timescale.ingest_lib import UPSERT_OHLCV_SQL, ensure_symbols, rows_from_provider_ohlcv
    from shunya.data.timescale.market_provider import TimescaleMarketDataProvider

    apply_migrations()

    sym = "__SHUNYA_TS_TEST__"
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1_000_000.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-06-03")]),
    )
    with psycopg.connect(timescale_dsn) as conn:
        with conn.cursor() as cur:
            tmap = ensure_symbols(cur, [sym])
            rows = rows_from_provider_ohlcv(raw, tmap, interval="1d", source="yfinance")
            cur.executemany(UPSERT_OHLCV_SQL, rows)
        conn.commit()

    prov = TimescaleMarketDataProvider(dsn=timescale_dsn, source="yfinance")
    out = prov.download([sym], "2024-06-01", "2024-06-10")
    assert not out.empty
    assert float(out["Close"].iloc[-1]) == pytest.approx(100.5)
