"""US and India **wall clocks** plus US listed-equity **regular session** (RTH) rules.

All times use :class:`zoneinfo.ZoneInfo` (IANA). US RTH follows the usual **NYSE** calendar
**Monday–Friday**, **09:30–16:00** America/New_York (**end exclusive** at 16:00:00). US **exchange
holidays** and **early closes** are **not** modeled yet (session may read “open” on some closed
days); see module docstring if you need exchange-calendar accuracy.

``alpaca_l1_us_equities_stream_allowed`` matches RTH by default. Set ``SHUNYA_ALPACA_L1_IGNORE_US_RTH``
to ``1`` / ``true`` / ``yes`` (API process only) to allow the instrument L1 WebSocket outside RTH
for development.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

# IANA zones for primary listing venues we surface in the UI header.
TZ_US_LISTED = ZoneInfo("America/New_York")
TZ_INDIA_LISTED = ZoneInfo("Asia/Kolkata")

# US cash equity regular session (NYSE/Nasdaq typical; no early-close / holiday table here).
US_LISTED_RTH_OPEN = time(9, 30)
US_LISTED_RTH_CLOSE = time(16, 0)  # exclusive at 16:00:00 local


def utc_now() -> datetime:
    """Current instant in UTC (timezone-aware)."""
    return datetime.now(UTC)


def _as_utc(dt: datetime | None) -> datetime:
    if dt is None:
        return utc_now()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _local_on_calendar_day(d: date, t: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(d, t, tzinfo=tz)


def is_us_listed_equity_regular_session_open(at: datetime | None = None) -> bool:
    """True during US listed **RTH** window (weekday 09:30–16:00 America/New_York, end exclusive).

    Does **not** exclude NYSE holidays or early closes; treat as a **session window** guard.
    """
    dt = _as_utc(at).astimezone(TZ_US_LISTED)
    if dt.weekday() >= 5:
        return False
    start = _local_on_calendar_day(dt.date(), US_LISTED_RTH_OPEN, TZ_US_LISTED)
    end = _local_on_calendar_day(dt.date(), US_LISTED_RTH_CLOSE, TZ_US_LISTED)
    return start <= dt < end


def _truthy_env(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def alpaca_l1_us_equities_stream_allowed(at: datetime | None = None) -> bool:
    """Whether the API should allow Alpaca US equity L1 streaming (RTH unless bypass env is set)."""
    if _truthy_env("SHUNYA_ALPACA_L1_IGNORE_US_RTH"):
        return True
    return is_us_listed_equity_regular_session_open(at)


def format_market_clock_line(country_slug: str, tz: ZoneInfo, at: datetime | None = None) -> str:
    """Format ``[SLUG] DD-MM HH:MM:SS.mmm`` in the given zone (wall clock).

    ``country_slug`` is typically ``US`` or ``IN`` (short, ASCII, safe for UI).
    """
    dt = _as_utc(at).astimezone(tz)
    dd_mm = f"{dt.day:02d}-{dt.month:02d}"
    hms = dt.strftime("%H:%M:%S")
    ms = dt.microsecond // 1000
    return f"[{country_slug}] {dd_mm} {hms}.{ms:03d}"


@dataclass(frozen=True)
class MarketClockSnapshot:
    """Single point-in-time snapshot for API + UI (server is source of truth for this instant)."""

    utc_iso: str
    us_line: str
    in_line: str
    us_listed_rth_open: bool
    alpaca_l1_us_equities_stream_allowed: bool


def build_market_clock_snapshot(at: datetime | None = None) -> MarketClockSnapshot:
    """Build a :class:`MarketClockSnapshot` for ``at`` (default: now UTC)."""
    u = _as_utc(at)
    return MarketClockSnapshot(
        utc_iso=u.isoformat().replace("+00:00", "Z"),
        us_line=format_market_clock_line("US", TZ_US_LISTED, u),
        in_line=format_market_clock_line("IN", TZ_INDIA_LISTED, u),
        us_listed_rth_open=is_us_listed_equity_regular_session_open(u),
        alpaca_l1_us_equities_stream_allowed=alpaca_l1_us_equities_stream_allowed(u),
    )
