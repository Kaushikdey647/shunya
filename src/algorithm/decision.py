"""Explicit decision-time semantics for live execution vs research panels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

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
