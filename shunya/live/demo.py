"""Minimal ``finTs`` + :class:`~shunya.algorithm.portfolio_manager.PortfolioConstructionService` for local paper demos."""

from __future__ import annotations

from typing import Sequence, Union

import jax.numpy as jnp
import pandas as pd

from shunya.algorithm.finstrat import FinStrat
from shunya.algorithm.portfolio_manager import PortfolioConstructionService, TargetBlendConfig
from shunya.data.fints import finTs
from shunya.data.timeframes import default_bar_index_policy, default_bar_spec


def make_minimal_fints(
    ticker_list: Sequence[str],
    dates: Sequence[Union[str, pd.Timestamp]],
    *,
    base_price: float = 100.0,
    volume: float = 1e6,
) -> finTs:
    """Small OHLCV panel (same shape as tests ``make_stub_fints``) for demos and CLI."""
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


def build_demo_target_blend_pcs(
    tickers: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
) -> PortfolioConstructionService:
    """
    Two-ticker equal-weight style book using latest close ranks as raw scores.

    Defaults to ``SPY`` / ``QQQ`` so paper accounts can execute slices; override
    ``tickers`` when your Alpaca universe differs.
    """
    t = list(tickers or ("SPY", "QQQ"))
    d = list(dates or ("2024-01-02", "2024-01-03"))
    fts = make_minimal_fints(t, d, base_price=100.0)
    fs = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none")
    book = TargetBlendConfig(strategies=(("demo", fs, 1.0),))
    return PortfolioConstructionService(book=book)
