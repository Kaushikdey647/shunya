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
        yield str(dsn)
        return
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


def test_fundamentals_wide_daily_roundtrip(timescale_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import psycopg
    from datetime import date, datetime, timezone

    pytest.importorskip("psycopg")
    monkeypatch.setenv("DATABASE_URL", timescale_dsn)

    from shunya.data.timescale.dbutil import apply_migrations
    from shunya.data.timescale.fundamental_provider import (
        TimescaleDailyFundamentalDataProvider,
        TimescaleFundamentalDataProvider,
    )
    from shunya.data.timescale.ingest_lib import (
        UPSERT_FUND_DAILY_SQL,
        UPSERT_FUND_QUARTERLY_SQL,
        ensure_symbols,
    )

    apply_migrations()

    sym = "__FUND_WIDE_TEST__"
    fe = date(2024, 6, 30)
    nums = [float(i + 1) for i in range(16)]

    with psycopg.connect(timescale_dsn) as conn:
        with conn.cursor() as cur:
            tmap = ensure_symbols(cur, [sym])
            sid = tmap[sym]
            row_q = (sid, fe, "unit_test", *nums)
            cur.execute(UPSERT_FUND_QUARTERLY_SQL, row_q)
            cur.execute(UPSERT_FUND_QUARTERLY_SQL, row_q)
            row_d = (
                sid,
                datetime(2024, 6, 3, tzinfo=timezone.utc),
                "unit_test",
                1e9,
                None,
                15.0,
                None,
                None,
                None,
                1.0,
                None,
                None,
            )
            cur.execute(UPSERT_FUND_DAILY_SQL, row_d)
            cur.execute(UPSERT_FUND_DAILY_SQL, row_d)
        conn.commit()

    fp = TimescaleFundamentalDataProvider(dsn=timescale_dsn, source="unit_test")
    out = fp.fetch([sym], "2024-01-01", "2024-12-31", quarterly=True, fields=["Revenue", "Net_Income"])
    assert not out.empty
    ts = pd.Timestamp(fe).normalize()
    key = (sym, ts)
    assert key in out.index
    assert float(out.loc[key, "Revenue"]) == pytest.approx(1.0)
    assert float(out.loc[key, "Net_Income"]) == pytest.approx(2.0)

    dp = TimescaleDailyFundamentalDataProvider(dsn=timescale_dsn, source="unit_test")
    dout = dp.fetch([sym], "2024-06-01", "2024-06-10", fields=["Market_Cap", "Trailing_PE"])
    assert not dout.empty
    dkey = (sym, pd.Timestamp("2024-06-03").normalize())
    assert dkey in dout.index
    assert float(dout.loc[dkey, "Market_Cap"]) == pytest.approx(1e9)
    assert float(dout.loc[dkey, "Trailing_PE"]) == pytest.approx(15.0)
