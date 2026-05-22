"""Unit tests for :func:`resolve_instrument_ohlcv_sync` branches (mocked providers)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from api.services.instrument_ohlcv import (
    _dataframe_to_bars,
    resolve_instrument_ohlcv_sync,
)
from shunya.data.ohlcv_multindex import flatten_ohlcv_for_symbol as _flatten_ohlcv_for_symbol


def _sample_ohlcv_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02", tz="America/New_York"), pd.Timestamp("2024-01-03", tz="America/New_York")],
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [1e6, 1e6],
        },
        index=idx,
    )


@pytest.fixture
def no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SHUNYA_DATABASE_URL", raising=False)


def test_case3_no_database_uses_yfinance(no_database: None, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_ohlcv_df()

    mock_prov = MagicMock()
    mock_prov.download.return_value = df
    monkeypatch.setattr(
        "shunya.data.market_router.YFinanceMarketDataProvider",
        lambda session=None, **kwargs: mock_prov,
    )

    out = resolve_instrument_ohlcv_sync("ZZZ", "1d", "1mo")
    assert out.response.provenance is not None
    assert out.response.provenance.read_path == "live_fetch"
    assert out.response.provenance.upstream_source_id == "yfinance"
    assert out.response.storage_skipped is True
    assert len(out.response.bars) == 2
    assert out.pending_deferred_writeback is None


def test_case1_timescale_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force DSN present and Timescale path returning valid coverage."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:59999/nope")

    df = _sample_ohlcv_df()

    inner = MagicMock()
    inner.execute = MagicMock()
    outer = MagicMock()
    outer.__enter__.return_value = inner
    outer.__exit__.return_value = None
    monkeypatch.setattr("psycopg.connect", MagicMock(return_value=outer))

    ts_prov = MagicMock()
    ts_prov.download.return_value = df
    monkeypatch.setattr(
        "shunya.data.timescale.market_provider.TimescaleMarketDataProvider",
        lambda dsn=None, source="yfinance": ts_prov,
    )

    monkeypatch.setattr(
        "shunya.data.market_router.validate_core_ohlcv_coverage",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "shunya.data.market_router.fetch_ohlcv_manifest_last_refresh_sync",
        lambda *a, **k: datetime.now(timezone.utc),
    )

    mock_yf = MagicMock()
    monkeypatch.setattr(
        "shunya.data.market_router.YFinanceMarketDataProvider",
        lambda session=None, **kwargs: mock_yf,
    )

    out = resolve_instrument_ohlcv_sync("ZZZ", "1d", "1mo")
    assert out.response.provenance is not None
    assert out.response.provenance.read_path == "timescale"
    assert len(out.response.bars) == 2
    mock_yf.download.assert_not_called()


def test_case2_timescale_incomplete_triggers_yfinance_writeback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:59999/nope")

    df_ok = _sample_ohlcv_df()
    df_incomplete = df_ok.iloc[:1]

    inner = MagicMock()
    inner.execute = MagicMock()
    outer = MagicMock()
    outer.__enter__.return_value = inner
    outer.__exit__.return_value = None
    monkeypatch.setattr("psycopg.connect", MagicMock(return_value=outer))

    ts_prov = MagicMock()
    ts_prov.download.return_value = df_incomplete
    monkeypatch.setattr(
        "shunya.data.timescale.market_provider.TimescaleMarketDataProvider",
        lambda dsn=None, source="yfinance": ts_prov,
    )

    def _manifest_now(*_a: object, **_k: object) -> datetime:
        return datetime.now(timezone.utc)

    monkeypatch.setattr(
        "shunya.data.market_router.fetch_ohlcv_manifest_last_refresh_sync",
        _manifest_now,
    )
    monkeypatch.setattr(
        "shunya.data.timescale.market_cache_lib.fetch_ohlcv_manifest_last_refresh_sync",
        _manifest_now,
    )

    def _validate_incomplete(*_a: object, **_k: object) -> None:
        raise ValueError("incomplete")

    monkeypatch.setattr(
        "shunya.data.market_router.validate_core_ohlcv_coverage",
        _validate_incomplete,
    )

    cols = pd.MultiIndex.from_tuples(
        [
            ("Open", "ZZZ"),
            ("High", "ZZZ"),
            ("Low", "ZZZ"),
            ("Close", "ZZZ"),
            ("Volume", "ZZZ"),
        ],
        names=["Price", "Ticker"],
    )
    df_yf_price_first = df_ok.copy()
    df_yf_price_first.columns = cols

    mock_yf = MagicMock()
    mock_yf.download.return_value = df_yf_price_first
    monkeypatch.setattr(
        "shunya.data.market_router.YFinanceMarketDataProvider",
        lambda session=None, **kwargs: mock_yf,
    )

    with patch("api.services.instrument_ohlcv.replace_ohlcv_range_sync") as wr:
        wr.return_value = (1, 2)
        out = resolve_instrument_ohlcv_sync("ZZZ", "1d", "1mo", defer_storage=False)
    assert out.response.provenance is not None
    assert out.response.provenance.read_path == "live_fetch"
    assert out.response.provenance.upstream_source_id == "yfinance"
    assert out.response.storage_status == "ok"
    assert len(out.response.bars) == 2
    wr.assert_called_once()
    passed = wr.call_args.kwargs["ohlcv_df"]
    assert not isinstance(passed.columns, pd.MultiIndex)
    assert list(passed.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.pending_deferred_writeback is None


def test_yfinance_price_first_multiindex_flattens_to_bars() -> None:
    """yfinance default group_by='column' uses (Field, Ticker) MultiIndex; bars must not be empty."""
    cols = pd.MultiIndex.from_tuples(
        [
            ("Open", "ZZZ"),
            ("High", "ZZZ"),
            ("Low", "ZZZ"),
            ("Close", "ZZZ"),
            ("Volume", "ZZZ"),
        ],
        names=["Price", "Ticker"],
    )
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")])
    df = pd.DataFrame(
        [[10.0, 11.0, 9.0, 10.5, 1e6], [11.0, 12.0, 10.0, 11.5, 1e6]],
        index=idx,
        columns=cols,
    )
    flat = _flatten_ohlcv_for_symbol(df, "ZZZ")
    assert not isinstance(flat.columns, pd.MultiIndex)
    assert len(_dataframe_to_bars(flat, max_rows=5000)) == 2
