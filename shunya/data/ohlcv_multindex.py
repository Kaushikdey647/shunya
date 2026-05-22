"""Normalize multi-ticker OHLCV frames to a single-symbol flat OHLCV DataFrame."""

from __future__ import annotations

import pandas as pd

_OHLCV_FIELD_NAMES = frozenset({"Open", "High", "Low", "Close", "Volume", "Adj Close"})


def flatten_ohlcv_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    yfinance ``download`` may return MultiIndex columns either as ``(Ticker, Field)``
    (``group_by='ticker'``) or ``(Field, Ticker)`` with field names on level 0.
    Downstream expects flat ``Open``/… columns.
    """
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df
    sym_u = symbol.upper()
    lev0_names = {str(x) for x in df.columns.get_level_values(0).unique()}
    if lev0_names & _OHLCV_FIELD_NAMES:
        for lev in (1, 0):
            if lev >= df.columns.nlevels:
                continue
            for raw in df.columns.get_level_values(lev).unique():
                if str(raw).upper() != sym_u:
                    continue
                out = df.xs(raw, axis=1, level=lev, drop_level=True)
                if isinstance(out.columns, pd.MultiIndex):
                    return df
                return out.copy()
        return df
    tickers = [str(t) for t in df.columns.get_level_values(0).unique()]
    for t in tickers:
        if t.upper() == sym_u:
            return df[t].copy()
    if len(tickers) == 1:
        return df[tickers[0]].copy()
    return df
