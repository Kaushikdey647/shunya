"""Price snapshots from ``finTs`` for live / paper desk wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

import pandas as pd

from shunya.data.fints import finTs
from shunya.data.timeframes import normalize_bar_timestamp

if TYPE_CHECKING:
    from shunya.algorithm.portfolio_manager import PortfolioConstructionService


def close_prices_at(fts: finTs, tickers: Sequence[str], execution_date: pd.Timestamp) -> Dict[str, float]:
    """
    Last ``Close`` from the aligned panel at the bar matching ``execution_date``.

    ``execution_date`` is normalized with :func:`~shunya.data.timeframes.normalize_bar_timestamp`
    using ``fts.bar_spec`` so it matches the MultiIndex ``Date`` level.
    """
    dt = normalize_bar_timestamp(execution_date, fts.bar_spec)
    out: Dict[str, float] = {}
    missing: list[str] = []
    for t in tickers:
        sym = str(t)
        key = (sym, dt)
        if key not in fts.df.index:
            missing.append(sym)
            continue
        row = fts.df.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        c = float(row["Close"])
        if c <= 0.0 or not (c == c):  # NaN check
            missing.append(sym)
            continue
        out[sym] = c
    if missing:
        raise ValueError(f"No valid Close for tickers {missing} at normalized date {dt!r}")
    return out


def fin_ts_from_portfolio_construction(pcs: PortfolioConstructionService) -> finTs:
    """Shared ``finTs`` from a target-blend or alpha-blend book."""
    from shunya.algorithm.portfolio_manager import AlphaBlendConfig, TargetBlendConfig

    book = pcs.book
    if isinstance(book, TargetBlendConfig):
        return book.strategies[0][1]._ts
    if isinstance(book, AlphaBlendConfig):
        return book.master._ts
    raise TypeError(f"Unsupported book type: {type(book)!r}")
