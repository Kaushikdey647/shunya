"""Read OHLCV from TimescaleDB / Postgres using the :class:`MarketDataProvider` contract."""

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd

from ..timeframes import (
    BarIndexPolicy,
    BarSpec,
    default_bar_index_policy,
    default_bar_spec,
    normalize_history_index,
)
from .dbutil import get_database_url
from .intervals import bar_spec_to_interval_key


class TimescaleMarketDataProvider:
    """
    Load OHLCV previously ingested into ``ohlcv_bars`` (see bootstrap CLI).

    Requires optional dependency ``shunya-py[timescale]`` and ``DATABASE_URL``.
    """

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        source: str = "yfinance",
    ) -> None:
        self._dsn = dsn or get_database_url()
        self._source = str(source)

    def download(
        self,
        ticker_list: List[str],
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
    ) -> pd.DataFrame:
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Install the timescale extra: pip install 'shunya-py[timescale]'"
            ) from exc

        spec = bar_spec if bar_spec is not None else default_bar_spec()
        idx_policy = (
            bar_index_policy if bar_index_policy is not None else default_bar_index_policy()
        )
        interval = bar_spec_to_interval_key(spec)
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)

        if not ticker_list:
            return pd.DataFrame()

        sql = """
        SELECT s.ticker, b.ts, b.open, b.high, b.low, b.close, b.volume
        FROM ohlcv_bars b
        JOIN symbols s ON s.id = b.symbol_id
        WHERE s.ticker = ANY(%s)
          AND b.interval = %s
          AND b.source = %s
          AND b.ts >= %s
          AND b.ts < %s
        ORDER BY b.ts ASC
        """
        params = (list(str(t) for t in ticker_list), interval, self._source, t0, t1)

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                raw_rows = cur.fetchall()

        if not raw_rows:
            return pd.DataFrame()

        base = pd.DataFrame(
            raw_rows,
            columns=["ticker", "ts", "open", "high", "low", "close", "volume"],
        )
        base["ts"] = pd.to_datetime(base["ts"])

        parts: list[pd.DataFrame] = []
        keys: list[str] = []
        for t, sub in base.groupby("ticker", sort=True):
            p = sub.set_index("ts")[["open", "high", "low", "close", "volume"]].sort_index()
            p.columns = ["Open", "High", "Low", "Close", "Volume"]
            keys.append(str(t))
            parts.append(p)

        if len(parts) == 1:
            out = parts[0]
            out.index.name = "Date"
            return normalize_history_index(out, spec, policy=idx_policy)

        wide = pd.concat(parts, keys=keys, axis=1)
        wide.columns = wide.columns.set_names(["Ticker", None])
        wide.index.name = "Date"
        return normalize_history_index(wide, spec, policy=idx_policy)
