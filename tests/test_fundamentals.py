"""Tests for fundamental-data attachment and alpha-context access."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from shunya.algorithm.finstrat import FinStrat
from shunya.data.fints import finTs
from shunya.data.fundamentals import align_fundamental_panel_to_panel_index

from tests.conftest import make_stub_fints


class StaticMarketDataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def download(
        self,
        ticker_list,
        start,
        end,
        *,
        bar_spec=None,
        bar_index_policy=None,
    ) -> pd.DataFrame:
        return self._frame.copy()


class StaticFundamentalDataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def fetch(
        self,
        ticker_list,
        start,
        end,
        *,
        fields=None,
        quarterly=True,
        bar_spec=None,
    ) -> pd.DataFrame:
        if fields is None:
            return self._frame.copy()
        return self._frame.loc[:, list(fields)].copy()


def _make_raw_ohlcv(tickers: list[str], dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    cols = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Volume"]],
    )
    data: dict[tuple[str, str], list[float]] = {}
    for ticker in tickers:
        data[(ticker, "Open")] = [100.0] * len(idx)
        data[(ticker, "High")] = [101.0] * len(idx)
        data[(ticker, "Low")] = [99.0] * len(idx)
        data[(ticker, "Close")] = [100.0] * len(idx)
        data[(ticker, "Volume")] = [1_000_000.0] * len(idx)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_align_fundamental_panel_forward_fills_period_end_values() -> None:
    panel_index = pd.MultiIndex.from_product(
        [["AAA"], pd.DatetimeIndex(pd.to_datetime(["2020-03-30", "2020-03-31", "2020-04-01"]), name="Date")],
        names=["Ticker", "Date"],
    )
    periodic = pd.DataFrame(
        {"Revenue": [100.0, 120.0]},
        index=pd.MultiIndex.from_tuples(
            [
                ("AAA", pd.Timestamp("2019-12-31")),
                ("AAA", pd.Timestamp("2020-03-31")),
            ],
            names=["Ticker", "Date"],
        ),
    )

    aligned = align_fundamental_panel_to_panel_index(periodic, panel_index)

    assert float(aligned.loc[("AAA", pd.Timestamp("2020-03-30")), "Revenue"]) == 100.0
    assert float(aligned.loc[("AAA", pd.Timestamp("2020-03-31")), "Revenue"]) == 120.0
    assert float(aligned.loc[("AAA", pd.Timestamp("2020-04-01")), "Revenue"]) == 120.0


def test_fints_attaches_fundamentals_to_panel() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-03-30", "2020-03-31", "2020-04-01"]
    market = StaticMarketDataProvider(_make_raw_ohlcv(tickers, dates))
    periodic = pd.DataFrame(
        {
            "Revenue": [100.0, 120.0, 200.0, 210.0],
            "Return_On_Equity": [0.10, 0.15, 0.20, 0.22],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("AAA", pd.Timestamp("2019-12-31")),
                ("AAA", pd.Timestamp("2020-03-31")),
                ("BBB", pd.Timestamp("2019-12-31")),
                ("BBB", pd.Timestamp("2020-03-31")),
            ],
            names=["Ticker", "Date"],
        ),
    )
    fundamentals = StaticFundamentalDataProvider(periodic)

    fts = finTs(
        "2020-03-30",
        "2020-04-01",
        tickers,
        market_data=market,
        attach_yfinance_classifications=False,
        attach_fundamentals=True,
        fundamental_data=fundamentals,
        fundamental_fields=["Revenue", "Return_On_Equity"],
        feature_mode="ohlcv_only",
    )

    assert fts.fundamental_feature_columns == ("Revenue", "Return_On_Equity")
    assert float(fts.df.loc[("AAA", pd.Timestamp("2020-03-30")), "Revenue"]) == 100.0
    assert float(fts.df.loc[("AAA", pd.Timestamp("2020-03-31")), "Revenue"]) == 120.0
    assert float(fts.df.loc[("BBB", pd.Timestamp("2020-04-01")), "Return_On_Equity"]) == 0.22


def test_context_exposes_fundamental_features_and_respects_signal_delay() -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=10.0)
    fts.df["Revenue"] = [100.0, 150.0, 200.0, 250.0]
    fts._fundamental_feature_columns = ("Revenue",)
    fts.align_universe(("Close", "Volume", "Revenue"), on_bad_ticker="drop")

    def alpha(ctx) -> jnp.ndarray:
        return ctx.feature("Revenue").latest.astype(jnp.float32)

    fs = FinStrat(
        fts,
        alpha,
        signal_delay=1,
        neutralization="none",
    )
    ctx = fs.context_at("2020-01-03", tickers=fs.tickers_at("2020-01-03"))

    assert ctx.feature_names == ("Revenue",)
    assert ctx.features["Revenue"].shape == (1, 2)
    np.testing.assert_allclose(np.asarray(ctx.feature("Revenue").latest), np.array([100.0, 200.0]))
