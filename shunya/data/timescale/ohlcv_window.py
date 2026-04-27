"""Map instrument API ``period`` / ``interval`` to :class:`~.timeframes.BarSpec` and UTC bounds."""

from __future__ import annotations

from datetime import timezone

import pandas as pd

from ..timeframes import BarSpec, BarUnit


def yfinance_interval_to_bar_spec(interval: str) -> BarSpec:
    """Inverse of :func:`~.timeframes.bar_spec_to_yfinance_interval` for supported API strings."""
    m: dict[str, BarSpec] = {
        "1m": BarSpec(BarUnit.MINUTES, 1),
        "2m": BarSpec(BarUnit.MINUTES, 2),
        "5m": BarSpec(BarUnit.MINUTES, 5),
        "15m": BarSpec(BarUnit.MINUTES, 15),
        "30m": BarSpec(BarUnit.MINUTES, 30),
        "60m": BarSpec(BarUnit.MINUTES, 60),
        "90m": BarSpec(BarUnit.MINUTES, 90),
        "1h": BarSpec(BarUnit.HOURS, 1),
        "1d": BarSpec(BarUnit.DAYS, 1),
        "5d": BarSpec(BarUnit.DAYS, 5),
        "1wk": BarSpec(BarUnit.WEEKS, 1),
        "1mo": BarSpec(BarUnit.MONTHS, 1),
        "3mo": BarSpec(BarUnit.MONTHS, 3),
    }
    if interval not in m:
        raise ValueError(f"unsupported yfinance interval for BarSpec: {interval!r}")
    return m[interval]


def period_to_utc_bounds(
    period: str,
    *,
    anchor: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return ``(start_inclusive, end_exclusive)`` in UTC for Timescale / yfinance ``download``.

    ``end_exclusive`` is midnight UTC strictly after the anchor calendar day so daily
    bars through ``anchor``'s date are included with ``ts < end_exclusive``.
    """
    now = anchor if anchor is not None else pd.Timestamp.now(tz=timezone.utc)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    end_exclusive = now.normalize() + pd.Timedelta(days=1)

    if period == "ytd":
        start = pd.Timestamp(year=now.year, month=1, day=1, tzinfo=timezone.utc)
        return start, end_exclusive

    if period == "max":
        start = pd.Timestamp("1970-01-01", tzinfo=timezone.utc)
        return start, end_exclusive

    offsets: dict[str, pd.DateOffset | pd.Timedelta] = {
        "1d": pd.Timedelta(days=3),
        "5d": pd.Timedelta(days=14),
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
    }
    if period not in offsets:
        raise ValueError(f"unsupported period: {period!r}")
    delta = offsets[period]
    if isinstance(delta, pd.Timedelta):
        start = now - delta
    else:
        start = now - delta
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    return start, end_exclusive
