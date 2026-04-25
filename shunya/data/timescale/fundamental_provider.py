"""Read periodic fundamentals from ``fundamentals_field_values`` (EAV)."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..timeframes import BarSpec, default_bar_spec
from .dbutil import get_database_url


class TimescaleFundamentalDataProvider:
    """
    Reconstruct the wide periodic frame expected by :meth:`~shunya.data.fints.finTs._attach_fundamentals`.

    Requires ``shunya-py[timescale]`` and ``DATABASE_URL``.
    """

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        source: str = "yfinance_statements",
    ) -> None:
        self._dsn = dsn or get_database_url()
        self._source = str(source)

    def fetch(
        self,
        ticker_list: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        fields: Optional[Sequence[str]] = None,
        quarterly: bool = True,
        bar_spec: Optional[BarSpec] = None,
    ) -> pd.DataFrame:
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Install the timescale extra: pip install 'shunya-py[timescale]'"
            ) from exc

        _ = bar_spec if bar_spec is not None else default_bar_spec()
        if fields is None:
            from ..fundamentals import FUNDAMENTAL_FIELDS

            field_list = list(FUNDAMENTAL_FIELDS)
        else:
            field_list = [str(f) for f in fields]

        if not ticker_list or not field_list:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=field_list, dtype=float)

        freq = "quarterly" if quarterly else "yearly"
        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize()

        sql = """
        SELECT s.ticker, f.period_end, f.field, f.value
        FROM fundamentals_field_values f
        JOIN symbols s ON s.id = f.symbol_id
        WHERE s.ticker = ANY(%s)
          AND f.freq = %s
          AND f.source = %s
          AND f.period_end >= %s::date
          AND f.period_end <= %s::date
          AND f.field = ANY(%s)
        """
        params = (
            list(str(t) for t in ticker_list),
            freq,
            self._source,
            t0.date(),
            t1.date(),
            field_list,
        )

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                raw_rows = cur.fetchall()

        if not raw_rows:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=field_list, dtype=float)

        long_df = pd.DataFrame(raw_rows, columns=["Ticker", "Date", "field", "value"])
        long_df["Date"] = pd.to_datetime(long_df["Date"])
        wide = long_df.pivot_table(
            index=["Ticker", "Date"],
            columns="field",
            values="value",
            aggfunc="last",
        )
        wide = wide.reindex(columns=field_list)
        wide.columns.name = None
        return wide.astype(float)
