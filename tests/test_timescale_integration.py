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


def test_sec_reference_flag_and_view(timescale_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs inside one transaction and rolls back: the flag pass touches every symbols row."""
    import psycopg

    monkeypatch.setenv("DATABASE_URL", timescale_dsn)
    from shunya.data.timescale import reference_sync as rs
    from shunya.data.timescale.dbutil import apply_migrations

    apply_migrations()
    with psycopg.connect(timescale_dsn) as conn:
        try:
            with conn.cursor() as cur:
                rs.upsert_sec_tickers(
                    cur,
                    [
                        rs.SecTicker("__SHY_A__", "0009999991", "A CORP"),
                        rs.SecTicker("__SHY_B__", "0009999992", "B CORP"),
                    ],
                )
                rs.upsert_sec_company(
                    cur,
                    rs.parse_submissions(
                        {"cik": "0009999991", "sic": "3674", "sicDescription": "Semis", "ownerOrg": "04 Manufacturing"}
                    ),
                )
                rs.apply_alpaca_tradable(cur, {"__SHY_A__": "asset-a"})
                assert rs.tradable_symbols(cur, ["__SHY_A__", "__SHY_B__"]) == ["__SHY_A__"]
                cur.execute(
                    "SELECT ticker, sector, industry, is_alpaca_tradable FROM symbol_reference "
                    "WHERE ticker IN ('__SHY_A__', '__SHY_B__') ORDER BY ticker"
                )
                assert cur.fetchall() == [
                    ("__SHY_A__", "04 Manufacturing", "Semis", True),
                    ("__SHY_B__", None, None, False),
                ]
                pending = rs.ciks_needing_submissions(cur, refresh=False)
                assert "0009999992" in pending and "0009999991" not in pending
        finally:
            conn.rollback()


def test_alpaca_bars_coverage_refresh(timescale_dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage counts calendar gaps and sets has_alpaca_data; rolled back."""
    import psycopg

    monkeypatch.setenv("DATABASE_URL", timescale_dsn)
    from shunya.data.timescale.dbutil import apply_migrations

    apply_migrations()
    with psycopg.connect(timescale_dsn) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO symbols (ticker) VALUES ('__SHY_COV__') RETURNING id")
                sid = cur.fetchone()[0]
                cur.execute("INSERT INTO symbols (ticker) VALUES ('__SHY_NODATA__') RETURNING id")
                sid_none = cur.fetchone()[0]
                for d in ("2099-01-05", "2099-01-06", "2099-01-07"):
                    cur.execute(
                        "INSERT INTO market_sessions VALUES (%s, %s::date + time '14:30', %s::date + time '21:00') "
                        "ON CONFLICT DO NOTHING",
                        (d, d, d),
                    )
                for d in ("2099-01-05", "2099-01-07"):  # 01-06 missing
                    cur.execute(
                        "INSERT INTO alpaca_bars_daily (symbol_id, session_date, feed, open, high, low, close, volume, vwap, trade_count) "
                        "VALUES (%s, %s, 'sip', 1, 1, 1, 1, 100, 1, 5)",
                        (sid, d),
                    )
                cur.execute("SELECT refresh_symbol_data_coverage('sip', %s)", ([sid, sid_none],))
                assert cur.fetchone()[0] == 1
                cur.execute(
                    "SELECT first_session::text, last_session::text, n_sessions, missing_sessions "
                    "FROM symbol_data_coverage WHERE symbol_id = %s AND feed = 'sip'",
                    (sid,),
                )
                assert cur.fetchone() == ("2099-01-05", "2099-01-07", 2, 1)
                cur.execute("SELECT id, has_alpaca_data FROM symbols WHERE id = ANY(%s) ORDER BY id", ([sid, sid_none],))
                assert dict(cur.fetchall()) == {sid: True, sid_none: False}
        finally:
            conn.rollback()
