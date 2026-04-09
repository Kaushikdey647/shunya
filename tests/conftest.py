"""Shared test helpers."""

from __future__ import annotations

from typing import Sequence, Union

import pandas as pd

from shunya.data.fints import finTs
from shunya.data.timeframes import default_bar_index_policy, default_bar_spec


def make_stub_fints(
    ticker_list: Sequence[str],
    dates: Sequence[Union[str, pd.Timestamp]],
    *,
    base_price: float = 100.0,
    volume: float = 1e6,
) -> finTs:
    """Minimal ``finTs`` stub with MultiIndex OHLCV rows only."""
    rows = [(t, pd.Timestamp(d).normalize()) for t in ticker_list for d in dates]
    idx = pd.MultiIndex.from_tuples(rows, names=["Ticker", "Date"])
    n = len(rows)
    df = pd.DataFrame(
        {
            "Open": [base_price] * n,
            "High": [base_price + 1.0] * n,
            "Low": [base_price - 1.0] * n,
            "Close": [base_price] * n,
            "Volume": [volume] * n,
        },
        index=idx,
    )
    stub = object.__new__(finTs)
    stub.start_date = dates[0]
    stub.end_date = dates[-1]
    stub.session = None
    stub.ticker_list = list(ticker_list)
    stub.df = df
    stub._aligned_calendar = None
    stub._fundamental_feature_columns = tuple()
    stub.bar_spec = default_bar_spec()
    stub._bar_index_policy = default_bar_index_policy()
    stub._trading_axis_mode = "observed"
    return stub
