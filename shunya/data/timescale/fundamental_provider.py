"""Read fundamentals from wide Timescale tables (with EAV fallback for legacy data)."""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..timeframes import BarSpec, default_bar_spec, normalize_bar_timestamp
from .dbutil import get_database_url
from .fundamentals_relational_lib import DAILY_FIELD_TO_SQL, FUNDAMENTAL_SQL_COLS, SQL_TO_PY_PERIODIC


def _wide_periodic_fetch(
    cur: object,
    *,
    table: str,
    ticker_list: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    field_list: list[str],
    source: str,
) -> list[tuple]:
    selected_sql_cols = [c for c in FUNDAMENTAL_SQL_COLS if SQL_TO_PY_PERIODIC[c] in field_list]
    if not selected_sql_cols:
        return []
    col_sql = ", ".join(f"f.{c}" for c in selected_sql_cols)
    sql = f"""
    SELECT s.ticker, f.fiscal_period_end, {col_sql}
    FROM {table} f
    JOIN symbols s ON s.id = f.symbol_id
    WHERE s.ticker = ANY(%s)
      AND f.source = %s
      AND f.fiscal_period_end >= %s::date
      AND f.fiscal_period_end <= %s::date
    """
    cur.execute(
        sql,
        (list(str(t) for t in ticker_list), source, start.date(), end.date()),
    )
    return cur.fetchall()


def _eav_fetch(
    cur: object,
    *,
    ticker_list: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    field_list: list[str],
    source: str,
    freq: str,
) -> list[tuple]:
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
    cur.execute(
        sql,
        (list(str(t) for t in ticker_list), freq, source, start.date(), end.date(), field_list),
    )
    return cur.fetchall()


class TimescaleFundamentalDataProvider:
    """
    Reconstruct the wide periodic frame expected by :meth:`~shunya.data.fints.finTs._attach_fundamentals`.

    Reads ``fundamentals_quarterly`` / ``fundamentals_annual`` when populated; otherwise falls back
    to legacy ``fundamentals_field_values`` (EAV).

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

        use_spec = bar_spec if bar_spec is not None else default_bar_spec()
        _ = use_spec
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
        table = "fundamentals_quarterly" if quarterly else "fundamentals_annual"
        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize()

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                raw_rows = _wide_periodic_fetch(
                    cur,
                    table=table,
                    ticker_list=ticker_list,
                    start=t0,
                    end=t1,
                    field_list=field_list,
                    source=self._source,
                )
                if not raw_rows:
                    raw_rows = _eav_fetch(
                        cur,
                        ticker_list=ticker_list,
                        start=t0,
                        end=t1,
                        field_list=field_list,
                        source=self._source,
                        freq=freq,
                    )

        if not raw_rows:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=field_list, dtype=float)

        sample = raw_rows[0]
        is_eav = len(sample) == 4 and isinstance(sample[2], str)

        if not is_eav:
            selected_sql_cols = [c for c in FUNDAMENTAL_SQL_COLS if SQL_TO_PY_PERIODIC[c] in field_list]
            py_metrics = [SQL_TO_PY_PERIODIC[c] for c in selected_sql_cols]
            header = ["Ticker", "Date"] + py_metrics
            df = pd.DataFrame(raw_rows, columns=header)
            df["Date"] = pd.to_datetime(df["Date"]).map(lambda x: normalize_bar_timestamp(x, use_spec))
            wide = df.set_index(["Ticker", "Date"]).sort_index()
            for c in field_list:
                if c not in wide.columns:
                    wide[c] = float("nan")
            wide = wide.reindex(columns=field_list)
            wide.columns.name = None
            return wide.astype(float)

        long_df = pd.DataFrame(raw_rows, columns=["Ticker", "Date", "field", "value"])
        long_df["Date"] = pd.to_datetime(long_df["Date"]).map(lambda x: normalize_bar_timestamp(x, use_spec))
        wide = long_df.pivot_table(
            index=["Ticker", "Date"],
            columns="field",
            values="value",
            aggfunc="last",
        )
        wide = wide.reindex(columns=field_list)
        wide.columns.name = None
        return wide.astype(float)


class TimescaleDailyFundamentalDataProvider:
    """
    Read :data:`~shunya.data.timescale.fundamentals_relational_lib.DAILY_FUNDAMENTAL_FIELDS` from
    ``fundamentals_daily`` for bar dates.
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
        bar_spec: Optional[BarSpec] = None,
    ) -> pd.DataFrame:
        try:
            import psycopg
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Install the timescale extra: pip install 'shunya-py[timescale]'"
            ) from exc

        from ..fundamentals import validate_daily_fundamental_fields

        use_spec = bar_spec if bar_spec is not None else default_bar_spec()
        field_list = list(validate_daily_fundamental_fields(fields))

        if not ticker_list or not field_list:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=field_list, dtype=float)

        sql_parts = ["s.ticker", "f.as_of_ts"]
        for py in field_list:
            sql_parts.append(f"f.{DAILY_FIELD_TO_SQL[py]}")
        col_sql = ", ".join(sql_parts)

        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize()
        t1_exclusive = t1 + pd.Timedelta(days=1)
        t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        t1_ex_utc = (
            t1_exclusive.tz_localize("UTC")
            if t1_exclusive.tzinfo is None
            else t1_exclusive.tz_convert("UTC")
        )

        sql = f"""
        SELECT {col_sql}
        FROM fundamentals_daily f
        JOIN symbols s ON s.id = f.symbol_id
        WHERE s.ticker = ANY(%s)
          AND f.source = %s
          AND f.as_of_ts >= %s::timestamptz
          AND f.as_of_ts < %s::timestamptz
        """
        params = (
            list(str(t) for t in ticker_list),
            self._source,
            t0_utc.to_pydatetime(),
            t1_ex_utc.to_pydatetime(),
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

        header = ["Ticker", "Date"] + field_list
        df = pd.DataFrame(raw_rows, columns=header)
        ts = pd.to_datetime(df["Date"], utc=True)
        df["Date"] = ts.dt.tz_convert(None).dt.normalize().map(lambda x: normalize_bar_timestamp(x, use_spec))
        wide = df.set_index(["Ticker", "Date"]).sort_index()
        wide.columns.name = None
        return wide.astype(float)
