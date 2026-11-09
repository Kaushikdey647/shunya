"""Corporate-action adjustment of raw daily bars (stored raw, adjusted at read time).

Each action gets a price and volume factor once, at ingest, from the raw close of the session
before its ex date (:func:`event_factors`). A bar on date ``d`` is then multiplied by the product
of the factors of every action with ``ex_date > d`` (:func:`adjust_bars`). The latest bar is never
scaled, so the last close stays the tradable price; this matches Alpaca's ``adjustment=all``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

AdjustmentMode = Literal["raw", "split", "all"]

# Share-count changes: prices scale by 1/r, volume by r.
SPLIT_TYPES = frozenset({"forward_split", "reverse_split", "stock_dividend"})
DIVIDEND_TYPES = frozenset({"cash_dividend"})
_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "VWAP")


@dataclass(frozen=True)
class AdjustmentEvent:
    """Stored factors of one corporate action."""

    ex_date: date
    action_type: str
    price_factor: float
    volume_factor: float


def event_factors(
    action_type: str,
    *,
    old_rate: Optional[float] = None,
    new_rate: Optional[float] = None,
    rate: Optional[float] = None,
    prev_close: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """``(price_factor, volume_factor)`` for one action, or ``None`` when it has no price effect or is unknown.

    * splits: ``r = new_rate / old_rate`` (4-for-1: ``old=1, new=4``; 1-for-10 reverse: ``old=10, new=1``)
    * stock dividend: ``r = 1 + rate`` (``rate`` = new shares per share held)
    * cash dividend: ``1 - rate / prev_close`` with the raw close of the session before the ex date
    """
    if action_type in ("forward_split", "reverse_split"):
        if not old_rate or not new_rate or old_rate <= 0 or new_rate <= 0:
            return None
        r = float(new_rate) / float(old_rate)
        return 1.0 / r, r
    if action_type == "stock_dividend":
        if rate is None or rate <= 0:
            return None
        r = 1.0 + float(rate)
        return 1.0 / r, r
    if action_type == "cash_dividend":
        if rate is None or rate <= 0 or prev_close is None or not np.isfinite(prev_close) or prev_close <= rate:
            return None
        return 1.0 - float(rate) / float(prev_close), 1.0
    # spin_off needs the spun-off security's price; not adjusted yet (gap, see #3).
    return None


def _events_for_mode(events: Sequence[AdjustmentEvent], mode: AdjustmentMode) -> list[AdjustmentEvent]:
    if mode == "raw":
        return []
    allowed = SPLIT_TYPES | (DIVIDEND_TYPES if mode == "all" else frozenset())
    return sorted((e for e in events if e.action_type in allowed), key=lambda e: e.ex_date)


def cumulative_factors(
    dates: pd.DatetimeIndex, events: Sequence[AdjustmentEvent], mode: AdjustmentMode = "all"
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-date ``(price, volume)`` multipliers: product over events with ``ex_date > date``."""
    evs = _events_for_mode(events, mode)
    n = len(dates)
    if not evs:
        return np.ones(n), np.ones(n)
    ex = np.array([np.datetime64(e.ex_date, "D") for e in evs])
    # suffix[i] = product of factors of events i..end; suffix[len] = 1
    pf = np.append(np.cumprod([e.price_factor for e in evs][::-1])[::-1], 1.0)
    vf = np.append(np.cumprod([e.volume_factor for e in evs][::-1])[::-1], 1.0)
    idx = np.searchsorted(ex, dates.values.astype("datetime64[D]"), side="right")
    return pf[idx], vf[idx]


def adjust_bars(bars: pd.DataFrame, events: Sequence[AdjustmentEvent], mode: AdjustmentMode = "all") -> pd.DataFrame:
    """Single-symbol raw bars (``DatetimeIndex`` of session dates) → adjusted copy.

    Prices (``Open/High/Low/Close/VWAP``) scale by the price factor, ``Volume`` by the volume
    factor; ``TradeCount`` is a count of prints and stays raw.
    """
    out = bars.copy()
    pf, vf = cumulative_factors(pd.DatetimeIndex(out.index), events, mode)
    for col in _PRICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].to_numpy(dtype=float) * pf
    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].to_numpy(dtype=float) * vf
    return out
