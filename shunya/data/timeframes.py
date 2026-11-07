"""
Bar cadence types for :class:`~shunya.data.fints.finTs` and market data providers.

``BarUnit`` + :class:`BarSpec` map to yfinance ``interval`` and Alpaca :class:`~alpaca.data.timeframe.TimeFrame`
where supported. Unsupported combinations raise :class:`ValueError` at mapping time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

if TYPE_CHECKING:
    from alpaca.data.timeframe import TimeFrame


class BarUnit(StrEnum):
    """Named bar cadence units mapped to provider intervals."""
    SECONDS = "SECONDS"
    MINUTES = "MINUTES"
    HOURS = "HOURS"
    DAYS = "DAYS"
    WEEKS = "WEEKS"
    MONTHS = "MONTHS"
    YEARS = "YEARS"


@dataclass(frozen=True)
class BarSpec:
    """Provider bar cadence: ``unit`` and positive ``step`` (e.g. 5-minute bars)."""

    unit: BarUnit
    step: int = 1

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError(f"BarSpec.step must be >= 1, got {self.step!r}")


def default_bar_spec() -> BarSpec:
    """Daily bars (historical default)."""
    return BarSpec(BarUnit.DAYS, 1)


DailyAnchor = Literal["timezone", "utc"]
US_EQUITIES_SESSION_OPEN = (9, 30)
US_EQUITIES_SESSION_CLOSE = (16, 0)
US_EQUITIES_SESSION_SECONDS = 6.5 * 60.0 * 60.0


@dataclass(frozen=True)
class BarIndexPolicy:
    """
    How provider timestamps are converted before panels are built.

    Default targets **America/New_York** (US equities session clock). Yahoo and Alpaca
    emit UTC-aware instants; converting to this zone before optional ``naive`` stripping
    keeps RTH bar labels stable across DST.

    **Naive input:** if the index has no timezone, timestamps are treated as wall-clock
    values already in ``timezone`` (useful for tests and pre-localized CSVs).

    For migration from older Shunya (always UTC-naive indices), use
    ``BarIndexPolicy(timezone=\"UTC\", daily_anchor=\"utc\")``.
    """

    timezone: str = "America/New_York"
    naive: bool = True
    daily_anchor: DailyAnchor = "timezone"

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, TypeError) as exc:
            raise ValueError(
                f"BarIndexPolicy.timezone must be a valid IANA name, got {self.timezone!r}"
            ) from exc
        if self.daily_anchor not in ("timezone", "utc"):
            raise ValueError(
                f"BarIndexPolicy.daily_anchor must be 'timezone' or 'utc', got {self.daily_anchor!r}"
            )


def default_bar_index_policy() -> BarIndexPolicy:
    """Default: NY session clock, naive index, daily bars anchored to that zone."""
    return BarIndexPolicy()


def _maybe_strip_tz(idx: pd.DatetimeIndex, *, naive: bool) -> pd.DatetimeIndex:
    if not naive or idx.tz is None:
        return idx
    return idx.tz_localize(None)


def _normalize_history_index_core(
    idx: pd.DatetimeIndex, spec: BarSpec, policy: BarIndexPolicy
) -> pd.DatetimeIndex:
    if idx.tz is not None:
        if bar_spec_is_daily_like(spec) and policy.daily_anchor == "utc":
            idx_utc = idx.tz_convert("UTC")
            out = idx_utc.normalize()
            return _maybe_strip_tz(out, naive=policy.naive)
        idx = idx.tz_convert(policy.timezone)
    else:
        # Naive wall clock: interpret as already in ``policy.timezone``.
        pass

    if bar_spec_is_intraday(spec):
        return _maybe_strip_tz(idx, naive=policy.naive)

    if policy.daily_anchor == "utc":
        if idx.tz is not None:
            idx_u = idx.tz_convert("UTC").normalize()
        else:
            idx_u = idx.normalize()
        return _maybe_strip_tz(idx_u, naive=policy.naive)

    out = idx.normalize()
    return _maybe_strip_tz(out, naive=policy.naive)


def bar_spec_is_intraday(spec: BarSpec) -> bool:
    """
    True when bar timestamps must preserve clock time (no midnight normalization).

    Hourly bars are treated as intraday so session/time-of-day remains meaningful.
    """
    return spec.unit in (BarUnit.SECONDS, BarUnit.MINUTES, BarUnit.HOURS)


def bar_spec_is_daily_like(spec: BarSpec) -> bool:
    """True when bars are anchored to calendar days or coarser (midnight-normal index)."""
    return not bar_spec_is_intraday(spec)


def _to_policy_zone(ts: pd.Timestamp, policy: BarIndexPolicy) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(policy.timezone)
    return t.tz_convert(policy.timezone)


def _from_policy_zone(ts: pd.Timestamp, policy: BarIndexPolicy) -> pd.Timestamp:
    if policy.naive:
        return pd.Timestamp(ts.tz_localize(None))
    return pd.Timestamp(ts)


def _intraday_step(spec: BarSpec) -> pd.Timedelta:
    if spec.unit == BarUnit.SECONDS:
        return pd.Timedelta(seconds=spec.step)
    if spec.unit == BarUnit.MINUTES:
        return pd.Timedelta(minutes=spec.step)
    if spec.unit == BarUnit.HOURS:
        return pd.Timedelta(hours=spec.step)
    raise ValueError(f"intraday step only valid for intraday specs, got {spec!r}")


def build_trading_calendar(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    spec: BarSpec,
    *,
    policy: BarIndexPolicy | None = None,
) -> pd.DatetimeIndex:
    """
    Canonical US-equities trading bars for ``[start, end]`` in policy timezone.

    - Daily-like bars: business-day calendar (Mon-Fri) at midnight.
    - Intraday bars: regular-hours grid (09:30 inclusive to 16:00 exclusive).
    """
    pol = policy if policy is not None else default_bar_index_policy()
    s_loc = _to_policy_zone(pd.Timestamp(start), pol)
    e_loc = _to_policy_zone(pd.Timestamp(end), pol)
    if s_loc > e_loc:
        return pd.DatetimeIndex([], name="Date")

    day0 = s_loc.normalize()
    day1 = e_loc.normalize()
    bdays = pd.date_range(day0, day1, freq="B", tz=pol.timezone)
    if len(bdays) == 0:
        return pd.DatetimeIndex([], name="Date")

    if bar_spec_is_daily_like(spec):
        out = pd.DatetimeIndex([_from_policy_zone(d.normalize(), pol) for d in bdays], name="Date")
        return out.sort_values()

    step = _intraday_step(spec)
    open_h, open_m = US_EQUITIES_SESSION_OPEN
    close_h, close_m = US_EQUITIES_SESSION_CLOSE
    pts: list[pd.Timestamp] = []
    for d in bdays:
        session_open = d.normalize() + pd.Timedelta(hours=open_h, minutes=open_m)
        session_close = d.normalize() + pd.Timedelta(hours=close_h, minutes=close_m)
        last_open = session_close - step
        if last_open < session_open:
            continue
        bars = pd.date_range(session_open, last_open, freq=step)
        pts.extend(_from_policy_zone(pd.Timestamp(ts), pol) for ts in bars)

    return pd.DatetimeIndex(pts, name="Date").sort_values()


def timestamp_is_on_trading_grid(
    ts: pd.Timestamp | str,
    spec: BarSpec,
    *,
    policy: BarIndexPolicy | None = None,
) -> bool:
    """True when ``ts`` is on the canonical US-equities trading grid for ``spec``."""
    pol = policy if policy is not None else default_bar_index_policy()
    t = _to_policy_zone(pd.Timestamp(ts), pol)
    if t.weekday() >= 5:
        return False
    if bar_spec_is_daily_like(spec):
        return bool(t.hour == 0 and t.minute == 0 and t.second == 0 and t.microsecond == 0)

    open_h, open_m = US_EQUITIES_SESSION_OPEN
    close_h, close_m = US_EQUITIES_SESSION_CLOSE
    session_open = t.normalize() + pd.Timedelta(hours=open_h, minutes=open_m)
    session_close = t.normalize() + pd.Timedelta(hours=close_h, minutes=close_m)
    if t < session_open or t >= session_close:
        return False
    step_s = int(_intraday_step(spec).total_seconds())
    offset_s = int((t - session_open).total_seconds())
    return step_s > 0 and (offset_s % step_s) == 0


def trading_time_distance(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    spec: BarSpec,
    *,
    policy: BarIndexPolicy | None = None,
) -> tuple[int, float]:
    """
    Return ``(bar_steps, trading_seconds)`` between two bar timestamps.

    Both timestamps must lie on the canonical trading grid for ``spec``.
    """
    pol = policy if policy is not None else default_bar_index_policy()
    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    sign = 1
    if t1 < t0:
        t0, t1 = t1, t0
        sign = -1
    cal = build_trading_calendar(t0, t1, spec, policy=pol)
    pos0 = int(cal.searchsorted(t0, side="left"))
    pos1 = int(cal.searchsorted(t1, side="left"))
    if pos0 >= len(cal) or cal[pos0] != t0:
        raise ValueError(f"start timestamp {t0!s} is not on trading grid")
    if pos1 >= len(cal) or cal[pos1] != t1:
        raise ValueError(f"end timestamp {t1!s} is not on trading grid")
    bars = (pos1 - pos0) * sign
    if bar_spec_is_intraday(spec):
        seconds = float(abs(pos1 - pos0) * _intraday_step(spec).total_seconds())
    else:
        seconds = float(abs(pos1 - pos0) * US_EQUITIES_SESSION_SECONDS)
    return bars, float(sign) * seconds


def normalize_bar_timestamp(ts: pd.Timestamp, spec: BarSpec) -> pd.Timestamp:
    """
    Align a timestamp to how the panel indexes bars for ``spec``.

    For intraday panels, pass timestamps that match the index after applying
    :func:`normalize_history_index` (e.g. exchange-local bar opens), not a
    different timezone's wall clock.
    """
    t = pd.Timestamp(ts)
    if bar_spec_is_intraday(spec):
        return t
    return t.normalize()


def bar_spec_to_yfinance_interval(spec: BarSpec) -> str | Literal["__monthly_then_year_resample"]:
    """
    Map to yfinance ``download(..., interval=...)``.

    Returns a sentinel for :attr:`BarUnit.YEARS` — callers fetch monthly then resample to yearly bars.
    """
    u, s = spec.unit, spec.step

    if u == BarUnit.SECONDS:
        raise ValueError("yfinance does not support sub-minute intervals; use BarUnit.MINUTES or higher.")

    if u == BarUnit.MINUTES:
        allowed = {1, 2, 5, 15, 30, 60, 90}
        if s not in allowed:
            raise ValueError(
                f"yfinance minute interval: step must be one of {sorted(allowed)}, got {s}"
            )
        return f"{s}m"

    if u == BarUnit.HOURS:
        if s != 1:
            raise ValueError(f"yfinance hourly interval supports step=1 only, got step={s}")
        return "1h"

    if u == BarUnit.DAYS:
        if s == 1:
            return "1d"
        if s == 5:
            return "5d"
        raise ValueError(f"yfinance daily interval: step must be 1 or 5, got {s}")

    if u == BarUnit.WEEKS:
        if s != 1:
            raise ValueError(f"yfinance weekly interval supports step=1 only, got step={s}")
        return "1wk"

    if u == BarUnit.MONTHS:
        if s == 1:
            return "1mo"
        if s == 3:
            return "3mo"
        raise ValueError(f"yfinance monthly interval: step must be 1 or 3, got {s}")

    if u == BarUnit.YEARS:
        return "__monthly_then_year_resample"

    raise ValueError(f"unsupported BarSpec for yfinance: {spec!r}")


def bar_spec_to_alpaca_timeframe(
    spec: BarSpec,
) -> "TimeFrame" | Literal["__monthly_then_year_resample"]:
    """
    Map to ``alpaca.data.timeframe.TimeFrame(amount, unit)``.

    Returns a sentinel for yearly bars — fetch monthly then resample to year-end OHLCV.
    """
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    u, s = spec.unit, spec.step

    if u == BarUnit.SECONDS:
        raise ValueError("Alpaca stock bars do not expose sub-minute timeframes; use BarUnit.MINUTES+.")

    if u == BarUnit.YEARS:
        return "__monthly_then_year_resample"

    if u == BarUnit.MINUTES:
        return TimeFrame(s, TimeFrameUnit.Minute)

    if u == BarUnit.HOURS:
        return TimeFrame(s, TimeFrameUnit.Hour)

    if u == BarUnit.DAYS:
        return TimeFrame(s, TimeFrameUnit.Day)

    if u == BarUnit.WEEKS:
        return TimeFrame(s, TimeFrameUnit.Week)

    if u == BarUnit.MONTHS:
        return TimeFrame(s, TimeFrameUnit.Month)

    raise ValueError(f"unsupported BarSpec for Alpaca: {spec!r}")


def resample_ohlcv_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate OHLCV to calendar-year bars (year-end timestamp label).

    Expects a sorted datetime index (any tz or naive; not mutated in place).
    """
    if df.empty:
        out = df.copy()
        out.index.name = df.index.name
        return out
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.DatetimeIndex(pd.to_datetime(work.index))
    # Use year period end as bar label (normalized to end of year).
    ag = work.resample("YE-DEC", label="right", closed="right").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    ag = ag.dropna(how="all")
    return ag


def normalize_history_index(
    df: pd.DataFrame,
    spec: BarSpec,
    *,
    policy: BarIndexPolicy | None = None,
) -> pd.DataFrame:
    """
    Normalize provider output index for ``spec`` and :class:`BarIndexPolicy`.

    Converts tz-aware indices to ``policy.timezone`` (or UTC when ``daily_anchor='utc'``
    for daily-like bars), optionally strips to naive. Naive indices are left as wall-clock
    values in ``policy.timezone``.
    """
    pol = policy if policy is not None else default_bar_index_policy()
    if df.empty:
        out = df.copy()
        out.index.name = "Date"
        return out
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    idx = _normalize_history_index_core(idx, spec, pol)
    out = df.copy()
    out.index = idx
    out.index.name = "Date"
    return out.sort_index()
