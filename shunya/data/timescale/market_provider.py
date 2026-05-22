"""Read OHLCV from TimescaleDB / Postgres using the :class:`MarketDataProvider` contract."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Union

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

from ..timeframes import (
    BarIndexPolicy,
    BarSpec,
    default_bar_index_policy,
    default_bar_spec,
    normalize_history_index,
)
from .dbutil import get_database_url
from shunya.data.market_data.constants import STORED_OHLCV_DEFAULT_UPSTREAM_ID

from .intervals import bar_spec_to_interval_key


def _drop_rows_nonfinite_ohlcv(
    p: pd.DataFrame,
    *,
    ticker: str | None = None,
    drop_events: list[tuple[str, int, int]] | None = None,
) -> pd.DataFrame:
    """
    Remove rows with non-finite OHLC, non-finite Volume (after repair), or negative Volume.

    Timescale allows NULL doubles on ``ohlcv_bars``; legacy rows may have NULL volume while OHLC
    is present. For those rows only, volume is coerced to ``0.0`` so valid price bars are kept.
    Rows with non-finite OHLC remain dropped (``finTs`` ``strict_ohlcv`` would reject them).

    When ``drop_events`` is passed, per-ticker drop counts are appended for one aggregated log
    in :meth:`TimescaleMarketDataProvider.download` (avoids hundreds of identical warnings on
    large universes).
    """
    if p.empty:
        return p
    work = p.copy()
    for c in ("Open", "High", "Low", "Close", "Volume"):
        work[c] = pd.to_numeric(work[c], errors="coerce")
    o = work[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    v = work["Volume"].to_numpy(dtype=float)
    o_ok = np.isfinite(o).all(axis=1)
    # NULL / NaN volume with finite OHLC: treat as zero (common in partial DB rows).
    v_rep = np.where(~np.isfinite(v) & o_ok, 0.0, v)
    ok = o_ok & np.isfinite(v_rep) & (v_rep >= 0.0)
    n_drop = int((~ok).sum())
    n_vol_repair = int((~np.isfinite(v) & o_ok).sum())
    if n_drop and drop_events is not None:
        label = ticker if ticker is not None else "?"
        drop_events.append((label, n_drop, len(work)))
    elif n_drop:
        label = ticker if ticker is not None else "?"
        _LOG.warning(
            "Dropped %d/%d Timescale OHLCV row(s) with non-finite or invalid OHLCV for %s",
            n_drop,
            len(work),
            label,
        )
    elif n_vol_repair and _LOG.isEnabledFor(logging.DEBUG):
        label = ticker if ticker is not None else "?"
        _LOG.debug(
            "Coerced NULL/NaN volume to 0 for %d/%d Timescale OHLCV row(s) (%s)",
            n_vol_repair,
            len(work),
            label,
        )
    work["Volume"] = v_rep
    return work.loc[ok].copy()


class TimescaleMarketDataProvider:
    """
    Load OHLCV previously ingested into ``ohlcv_bars`` (see bootstrap CLI).

    Requires optional dependency ``shunya-py[timescale]`` and ``DATABASE_URL``.
    """

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        source: str = STORED_OHLCV_DEFAULT_UPSTREAM_ID,
        enforce_cache_ttl: bool | None = None,
    ) -> None:
        self._dsn = dsn or get_database_url()
        self._source = str(source)
        if enforce_cache_ttl is None:
            self._enforce_cache_ttl = str(
                os.environ.get("SHUNYA_TIMESCALE_OHLCV_ENFORCE_CACHE_TTL", "")
            ).lower() in ("1", "true", "yes", "on")
        else:
            self._enforce_cache_ttl = bool(enforce_cache_ttl)

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
                if self._enforce_cache_ttl:
                    from shunya.data.timescale.market_cache_lib import (
                        default_market_cache_ttl_days,
                        ohlcv_manifests_all_fresh_for_universe_on_cursor,
                    )

                    tickers_u = list(dict.fromkeys(str(t) for t in ticker_list))
                    if not ohlcv_manifests_all_fresh_for_universe_on_cursor(
                        cur,
                        tickers=tickers_u,
                        interval=interval,
                        source=self._source,
                        ttl_days=default_market_cache_ttl_days(),
                        now=datetime.now(timezone.utc),
                    ):
                        return pd.DataFrame()
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
        drop_events: list[tuple[str, int, int]] = []
        for t, sub in base.groupby("ticker", sort=True):
            p = sub.set_index("ts")[["open", "high", "low", "close", "volume"]].sort_index()
            p.columns = ["Open", "High", "Low", "Close", "Volume"]
            p = _drop_rows_nonfinite_ohlcv(p, ticker=str(t), drop_events=drop_events)
            keys.append(str(t))
            parts.append(p)

        if drop_events:
            total_drop = sum(d for _, d, _ in drop_events)
            worst = sorted(drop_events, key=lambda x: -x[1])[:12]
            tail = ", ".join(f"{name}({d}/{n})" for name, d, n in worst)
            if len(drop_events) > 12:
                tail += ", …"
            _LOG.warning(
                "Timescale OHLCV: dropped %d bar(s) with non-finite OHLC or invalid volume "
                "across %d ticker(s) (see strict_ohlcv / legacy rows). Worst: %s",
                total_drop,
                len(drop_events),
                tail,
            )

        if len(parts) == 1:
            out = parts[0]
            out.index.name = "Date"
            return normalize_history_index(out, spec, policy=idx_policy)

        # Inner join: only timestamps where every ticker has a finite OHLCV row (outer
        # union would leave NaNs for missing symbol-dates and break finTs strict_ohlcv).
        wide = pd.concat(parts, keys=keys, axis=1, join="inner")
        wide.columns = wide.columns.set_names(["Ticker", None])
        wide.index.name = "Date"
        return normalize_history_index(wide, spec, policy=idx_policy)
