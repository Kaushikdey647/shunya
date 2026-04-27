"""
Strict checks on provider OHLCV frames before feature engineering.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from .timeframes import (
    BarIndexPolicy,
    BarSpec,
    build_trading_calendar,
    bar_spec_is_intraday,
    default_bar_index_policy,
    timestamp_is_on_trading_grid,
)

_OHLCV = ("Open", "High", "Low", "Close", "Volume")


def _midnight_in_policy_zone(
    day: Union[str, pd.Timestamp], policy: BarIndexPolicy
) -> pd.Timestamp:
    """Start of the calendar day ``day`` in ``policy.timezone``, naive or aware per policy."""
    d = pd.Timestamp(day).normalize()
    t = pd.Timestamp(d.date(), tz=policy.timezone)
    if policy.naive:
        return t.tz_localize(None)
    return t


def bounds_for_validation(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    spec: BarSpec,
    policy: BarIndexPolicy,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Return ``(t0, t1_incl, t1_open)`` for :func:`validate_core_ohlcv_coverage` and alignment.

    * Intraday: rows must satisfy ``t0 <= ts < t1_open`` (half-open window by calendar dates
      in ``policy.timezone``).
    * Daily-like: rows must satisfy ``t0 <= ts <= t1_incl`` at day resolution in that zone.
    """
    intraday = bar_spec_is_intraday(spec)
    raw_start = pd.Timestamp(start)
    raw_end = pd.Timestamp(end)
    t0 = _midnight_in_policy_zone(raw_start, policy)
    if intraday:
        end_mid = _midnight_in_policy_zone(raw_end, policy)
        t1_open = end_mid + pd.Timedelta(days=1)
        return t0, t0, t1_open
    t1_incl = _midnight_in_policy_zone(raw_end, policy)
    return t0, t1_incl, t1_incl


def validate_core_ohlcv_coverage(
    raw: pd.DataFrame,
    *,
    ticker_list: List[str],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    bar_spec: BarSpec,
    strict_provider_universe: bool,
    strict_ohlcv: bool,
    strict_empty: bool,
    bar_index_policy: BarIndexPolicy | None = None,
    strict_trading_grid: bool = False,
) -> None:
    """
    Validate provider output against requested symbols and finite OHLCV.

    Raises ``ValueError`` with actionable messages when strict flags are enabled.
    """
    if not ticker_list:
        return

    if raw.empty:
        if strict_empty:
            raise ValueError(
                "strict_empty: provider returned empty DataFrame but ticker_list is non-empty. "
                "Widen the date range, check symbols, or set strict_empty=False for exploratory loads."
            )
        return

    pol = (
        bar_index_policy
        if bar_index_policy is not None
        else default_bar_index_policy()
    )
    t0, t1_incl, t1_open = bounds_for_validation(start, end, bar_spec, pol)
    intraday = bar_spec_is_intraday(bar_spec)

    def _check_bounds(idx: pd.DatetimeIndex) -> None:
        if len(idx) == 0:
            return
        ts_min = pd.Timestamp(idx.min())
        ts_max = pd.Timestamp(idx.max())
        if intraday:
            if ts_min < t0 or ts_max >= t1_open:
                raise ValueError(
                    f"provider_index_out_of_range: expected rows in [{t0}, {t1_open}), "
                    f"got [{ts_min}, {ts_max}]. Check start_date/end_date vs provider output."
                )

    def _check_grid(idx: pd.DatetimeIndex, sym: str) -> None:
        if len(idx) == 0:
            return
        if not idx.is_monotonic_increasing:
            raise ValueError(f"strict_trading_grid: ticker {sym!r} index must be sorted ascending")
        if bool(idx.duplicated().any()):
            raise ValueError(f"strict_trading_grid: ticker {sym!r} index has duplicate timestamps")
        off_grid = [
            pd.Timestamp(ts)
            for ts in idx
            if not timestamp_is_on_trading_grid(ts, bar_spec, policy=pol)
        ]
        if off_grid:
            raise ValueError(
                "strict_trading_grid: off-grid timestamp(s) for ticker "
                f"{sym!r}; first={off_grid[0]!s}"
            )
        expected = build_trading_calendar(idx.min(), idx.max(), bar_spec, policy=pol)
        if len(expected) == 0:
            return
        missing = expected.difference(idx)
        if len(missing) > 0:
            raise ValueError(
                "strict_trading_grid: missing in-session bar(s) for ticker "
                f"{sym!r}; first_missing={pd.Timestamp(missing[0])!s}"
            )
        else:
            if ts_min < t0 or ts_max > t1_incl:
                raise ValueError(
                    f"provider_index_out_of_range: expected rows in [{t0}, {t1_incl}], "
                    f"got [{ts_min}, {ts_max}]. Check start_date/end_date vs provider output."
                )

    def _check_finite_ohlcv(part: pd.DataFrame, sym: str) -> None:
        for col in _OHLCV:
            if col not in part.columns:
                raise ValueError(f"strict_ohlcv: ticker {sym!r} missing column {col!r}")
            s = pd.to_numeric(part[col], errors="coerce")
            bad = ~np.isfinite(s.to_numpy(dtype=float))
            if col == "Volume":
                bad = bad | (s.to_numpy(dtype=float) < 0)
            if bad.any():
                first_idx = part.index[int(np.argmax(bad))]
                n_bad = int(bad.sum())
                raise ValueError(
                    f"strict_ohlcv: non-finite or invalid {col!r} for ticker {sym!r} "
                    f"(bad_rows={n_bad}, first_bad_index={first_idx!r})"
                )

    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0).unique().tolist())
        missing_syms = [t for t in ticker_list if t not in available]
        if missing_syms and strict_provider_universe:
            raise ValueError(
                "strict_provider_universe: provider output missing requested ticker(s): "
                + ", ".join(sorted(missing_syms))
            )
        for sym in ticker_list:
            if sym not in available:
                continue
            part = raw[sym].copy()
            part = part.sort_index()
            idx = pd.DatetimeIndex(pd.to_datetime(part.index))
            _check_bounds(idx)
            if strict_trading_grid:
                _check_grid(idx, sym)
            if strict_ohlcv:
                _check_finite_ohlcv(part, sym)
        return

    if len(ticker_list) != 1:
        if strict_provider_universe:
            raise ValueError(
                "strict_provider_universe: expected MultiIndex columns for multiple tickers, "
                f"got single-level columns for ticker_list={ticker_list!r}."
            )
        return

    sym = ticker_list[0]
    part = raw.copy().sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(part.index))
    _check_bounds(idx)
    if strict_trading_grid:
        _check_grid(idx, sym)
    if strict_ohlcv:
        _check_finite_ohlcv(part, sym)
