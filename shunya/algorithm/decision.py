"""Explicit decision-time semantics for live execution vs research panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import pandas as pd

DataSource = Literal["yfinance_research", "alpaca_bars"]

_DEFAULT_TZ = "America/New_York"


@dataclass(frozen=True)
class DecisionContext:
    """
    What data and calendar time the signal is defined on before orders go to the broker.

    Use ``data_source="yfinance_research"`` when the panel comes from :class:`finTs`
    backed by Yahoo (default). That history is **not** identical to Alpaca's tape;
    treat ``as_of`` as the bar date you believe is fully available (often **previous**
    session close for EOD rebalance), not "now".

    Use ``data_source="alpaca_bars"`` when the panel rows were built from Alpaca
    historical bars so clock/bar alignment is closer to what the broker uses.
    """

    as_of: pd.Timestamp
    data_source: DataSource = "yfinance_research"
    timezone: str = _DEFAULT_TZ
    bar_time_label: Optional[str] = "session_close_eod"
    """Human-readable label, e.g. ``session_close_eod`` or ``previous_regular_session_close``."""

    def as_of_date(self) -> pd.Timestamp:
        """Calendar-normalized timestamp for ``panel_at``."""
        return pd.Timestamp(self.as_of).normalize()


def resolve_panel_timestamp(
    *,
    decision: Optional[DecisionContext],
    explicit_as_of: Optional[Union[str, pd.Timestamp]],
    index_max_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Resolve the calendar date passed to :meth:`FinStrat.panel_at`.

    Precedence: ``decision.as_of`` if ``decision`` is set; else ``explicit_as_of``;
    else ``index_max_date``.
    """
    if decision is not None:
        return decision.as_of_date()
    if explicit_as_of is not None:
        return pd.Timestamp(explicit_as_of).normalize()
    return pd.Timestamp(index_max_date).normalize()


def validate_panel_timestamp(
    *,
    resolved_as_of: pd.Timestamp,
    index_max_date: pd.Timestamp,
    now_ts: Optional[pd.Timestamp] = None,
    timezone: str = _DEFAULT_TZ,
    enforce_weekday: bool = True,
    strict_same_session: bool = False,
    max_staleness_days: Optional[int] = None,
) -> Tuple[pd.Timestamp, List[str]]:
    """
    Validate resolved execution timestamp against simple session/date semantics.

    Returns normalized timestamp plus warning list.
    """
    dt = pd.Timestamp(resolved_as_of).normalize()
    idx_max = pd.Timestamp(index_max_date).normalize()
    warnings: List[str] = []

    if enforce_weekday and dt.weekday() >= 5:
        raise ValueError(f"as_of {dt.date()} is on weekend; choose a regular session date")

    if dt > idx_max:
        raise ValueError(f"as_of {dt.date()} is newer than panel max date {idx_max.date()}")

    tz = ZoneInfo(timezone)
    if now_ts is None:
        now_local = pd.Timestamp.now(tz=tz)
    else:
        n = pd.Timestamp(now_ts)
        now_local = n.tz_localize(tz) if n.tzinfo is None else n.tz_convert(tz)
    now_date = now_local.tz_localize(None).normalize()
    if dt > now_date:
        raise ValueError(f"as_of {dt.date()} is in the future relative to {timezone} date {now_date.date()}")

    if strict_same_session and dt != now_date:
        raise ValueError(
            f"strict_same_session=True requires as_of == current {timezone} date "
            f"({now_date.date()}), got {dt.date()}"
        )

    staleness = (idx_max - dt).days
    if staleness > 0:
        warnings.append(f"as_of_older_than_panel_max_by_days={staleness}")
    if max_staleness_days is not None and staleness > max_staleness_days:
        warnings.append(
            f"as_of_staleness_exceeds_limit days={staleness} limit={max_staleness_days}"
        )
    return dt, warnings
