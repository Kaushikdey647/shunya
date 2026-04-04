"""Tests for yfinance-based classification attachment in finTs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shunya.data.fints import finTs
from shunya.utils import indicators


class _StubMarketData:
    def download(self, ticker_list, start, end):
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")], name="Date")
        frames = {}
        for t in ticker_list:
            frames[t] = pd.DataFrame(
                {
                    "Open": [100.0, 101.0],
                    "High": [101.0, 102.0],
                    "Low": [99.0, 100.0],
                    "Close": [100.5, 101.5],
                    "Adj Close": [100.5, 101.5],
                    "Volume": [1_000_000, 1_100_000],
                },
                index=idx,
            )
        return pd.concat(frames, axis=1)


def test_fints_attaches_classification_columns(monkeypatch):
    def _fake_fetch(tickers, session=None):
        return {
            "AAPL": {
                "Sector": "Technology",
                "Industry": "Consumer Electronics",
                "SubIndustry": "Hardware",
            },
            "XOM": {
                "Sector": "Energy",
                "Industry": "Oil & Gas",
                "SubIndustry": "Integrated",
            },
        }

    monkeypatch.setattr("shunya.data.fints.fetch_yfinance_classifications", _fake_fetch)

    fts = finTs(
        "2024-01-01",
        "2024-01-10",
        ["AAPL", "XOM"],
        market_data=_StubMarketData(),
        attach_yfinance_classifications=True,
    )
    assert {"Sector", "Industry", "SubIndustry"}.issubset(set(fts.df.columns))
    aapl_sector = fts.df.loc[("AAPL", pd.Timestamp("2024-01-02")), "Sector"]
    xom_industry = fts.df.loc[("XOM", pd.Timestamp("2024-01-03")), "Industry"]
    assert aapl_sector == "Technology"
    assert xom_industry == "Oil & Gas"
    assert "VWAP" in fts.df.columns
    assert np.isfinite(fts.df["VWAP"].to_numpy(dtype=float)).all()
    assert indicators.feature_index("VWAP", live=True) >= 0


def test_fints_uses_unknown_fallbacks_when_missing(monkeypatch):
    monkeypatch.setattr(
        "shunya.data.fints.fetch_yfinance_classifications",
        lambda tickers, session=None: {"AAPL": {"Sector": "Technology"}},
    )
    fts = finTs(
        "2024-01-01",
        "2024-01-10",
        ["AAPL", "MSFT"],
        market_data=_StubMarketData(),
        attach_yfinance_classifications=True,
    )
    assert (
        fts.df.loc[("MSFT", pd.Timestamp("2024-01-03")), "Sector"] == "UnknownSector"
    )
    assert (
        fts.df.loc[("AAPL", pd.Timestamp("2024-01-03")), "SubIndustry"]
        == "UnknownSubIndustry"
    )


def test_fints_rejects_provider_missing_required_ohlcv_columns():
    class _BadMarketData:
        def download(self, ticker_list, start, end):
            del ticker_list, start, end
            return pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.5],
                    # Missing Volume on purpose.
                },
                index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")], name="Date"),
            )

    with pytest.raises(ValueError, match="Volume"):
        finTs(
            "2024-01-01",
            "2024-01-10",
            ["AAPL"],
            market_data=_BadMarketData(),
            attach_yfinance_classifications=False,
        )
