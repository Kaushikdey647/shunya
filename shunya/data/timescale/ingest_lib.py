"""Upsert helpers for bootstrap CLIs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


def ensure_symbols(cur: Any, tickers: Sequence[str]) -> dict[str, int]:
    """Upsert tickers into ``symbols``; return mapping ``ticker -> symbol_id``."""
    out: dict[str, int] = {}
    for t in tickers:
        cur.execute(
            """
            INSERT INTO symbols (ticker) VALUES (%s)
            ON CONFLICT (ticker) DO UPDATE SET ticker = EXCLUDED.ticker
            RETURNING id
            """,
            (str(t),),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute("SELECT id FROM symbols WHERE ticker = %s", (str(t),))
            row = cur.fetchone()
        out[str(t)] = int(row[0])
    return out


def rows_from_provider_ohlcv(
    df: pd.DataFrame,
    ticker_to_id: Mapping[str, int],
    *,
    interval: str,
    source: str,
) -> list[tuple[Any, ...]]:
    """Flatten a ``MarketDataProvider`` OHLCV frame into DB rows."""
    rows: list[tuple[Any, ...]] = []
    if df.empty:
        return rows

    if isinstance(df.columns, pd.MultiIndex):
        tickers = [t for t in df.columns.get_level_values(0).unique() if str(t) in ticker_to_id]
        for t in tickers:
            part = df[str(t)].copy()
            _append_ticker_bars(rows, part, ticker_to_id[str(t)], interval, source)
    else:
        if len(ticker_to_id) != 1:
            raise ValueError("single-level OHLCV frame requires exactly one ticker mapping")
        tid = next(iter(ticker_to_id.values()))
        _append_ticker_bars(rows, df.copy(), tid, interval, source)
    return rows


def _append_ticker_bars(
    rows: list[tuple[Any, ...]],
    part: pd.DataFrame,
    symbol_id: int,
    interval: str,
    source: str,
) -> None:
    need = ("Open", "High", "Low", "Close", "Volume")
    miss = [c for c in need if c not in part.columns]
    if miss:
        raise KeyError(f"OHLCV frame missing columns {miss}")
    part = part.sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(part.index))
    for i in range(len(part)):
        ts = idx[i]
        o = float(part["Open"].iloc[i])
        h = float(part["High"].iloc[i])
        l = float(part["Low"].iloc[i])
        c = float(part["Close"].iloc[i])
        v = float(part["Volume"].iloc[i])
        rows.append((symbol_id, ts.to_pydatetime(), interval, o, h, l, c, v, source))


UPSERT_OHLCV_SQL = """
INSERT INTO ohlcv_bars (symbol_id, ts, interval, open, high, low, close, volume, source)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, ts, interval, source) DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    ingested_at = now()
"""


def fundamentals_eav_rows(
    periodic: pd.DataFrame,
    ticker_to_id: Mapping[str, int],
    *,
    freq: str,
    source: str,
) -> list[tuple[Any, ...]]:
    """Explode a wide periodic ``(Ticker, Date)`` frame into EAV rows."""
    if periodic.empty:
        return []
    if tuple(periodic.index.names) != ("Ticker", "Date"):
        raise ValueError(f"expected MultiIndex ('Ticker', 'Date'), got {periodic.index.names!r}")
    rows: list[tuple[Any, ...]] = []
    for (ticker, dt), ser in periodic.iterrows():
        sid = ticker_to_id.get(str(ticker))
        if sid is None:
            continue
        pe = pd.Timestamp(dt).date()
        for field, val in ser.items():
            if pd.isna(val):
                continue
            fv = float(val)
            rows.append((sid, pe, str(freq), str(field), fv, str(source)))
    return rows


UPSERT_FUND_SQL = """
INSERT INTO fundamentals_field_values (symbol_id, period_end, freq, field, value, source)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol_id, period_end, freq, field, source) DO UPDATE SET
    value = EXCLUDED.value,
    ingested_at = now()
"""
