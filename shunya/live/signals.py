"""Live sleeve weights: replay a FinStrat over the lookback window ending at the signal date.

Decay is stateful, so a single ``pass_`` on the last date would not match the backtest;
replaying from ``reset_pipeline_state()`` mirrors what FinBT does bar by bar. Weights are
computed with ``capital=1`` (``pass_`` is linear in capital) and sized by the runner.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from shunya.algorithm.finstrat import FinStrat
from shunya.algorithm.inline_alpha import resolve_alpha_from_source
from shunya.data.fints import finTs
from shunya.data.providers import MarketDataProvider
from shunya.schemas.transforms import merge_finstrat_runtime_dict

# Same detection as api.runner auto-fundamentals: hosted runner has no fundamentals store.
_FUN_USE = re.compile(r"\b(?:ctx\.)?fun\.")
# Sector / industry neutralization needs Timescale classifications the hosted runner lacks.
LIVE_NEUTRALIZATIONS = ("none", "market")


class StaleDataError(RuntimeError):
    """Bars unusable for trading: latest bar is not the signal date, or too many symbols missing."""


class UnsupportedForLive(ValueError):
    """Alpha or config uses a feature the hosted runner cannot provide."""


_VOLUME_USE = re.compile(r"\b(?:adj_)?volume\b", re.IGNORECASE)
# Hosted runner reads Alpaca IEX bars: IEX prints are a few percent of consolidated volume.
MAX_MISSING_FRACTION = 0.05


def live_strategy_warnings(alpha_source: str) -> List[str]:
    """Non-fatal caveats shown at deploy time."""
    out: List[str] = []
    if _VOLUME_USE.search(alpha_source or ""):
        out.append(
            "alpha reads volume: live bars come from the Alpaca IEX feed, whose volume is a small, "
            "venue-specific slice of consolidated volume; backtest on the same feed before trusting it"
        )
    return out


def validate_live_strategy(alpha_source: str, finstrat_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Compile the alpha and return the normalized FinStrat config; raise :class:`UnsupportedForLive`."""
    if _FUN_USE.search(alpha_source or ""):
        raise UnsupportedForLive("alphas using fundamentals (fun.*) are not supported live yet")
    cfg = merge_finstrat_runtime_dict(dict(finstrat_config or {}), None)
    if cfg.get("neutralization", "market") not in LIVE_NEUTRALIZATIONS:
        raise UnsupportedForLive(
            f"neutralization={cfg.get('neutralization')!r} not supported live; use one of {LIVE_NEUTRALIZATIONS}"
        )
    resolve_alpha_from_source(alpha_source)
    return cfg


def load_daily_fin_ts(
    tickers: Sequence[str],
    signal_date: date,
    lookback_days: int,
    *,
    market_data: Optional[MarketDataProvider] = None,
) -> finTs:
    """Daily bars ``[signal_date - lookback_days, signal_date]`` (Alpaca, split/dividend adjusted, by default).

    IEX has days without prints for thinly traded names, so missing symbols (up to
    :data:`MAX_MISSING_FRACTION`) and NaN bars are tolerated; FinStrat skips non-finite names per date.
    More missing than that raises :class:`StaleDataError` (the runner auto-halts).
    """
    if market_data is None:
        from shunya.data.providers import AlpacaHistoricalMarketDataProvider

        market_data = AlpacaHistoricalMarketDataProvider(max_missing_fraction=MAX_MISSING_FRACTION)
    start = pd.Timestamp(signal_date - timedelta(days=int(lookback_days)))
    end = pd.Timestamp(signal_date + timedelta(days=1))  # end exclusive
    try:
        return finTs(
            start,
            end,
            sorted(set(tickers)),
            market_data=market_data,
            attach_yfinance_classifications=False,
            strict_provider_universe=False,
            strict_ohlcv=False,
        )
    except ValueError as exc:
        raise StaleDataError(f"daily bars unusable: {exc}") from exc


def build_strategy(fts: finTs, alpha_source: str, finstrat_config: Mapping[str, Any]) -> FinStrat:
    """FinStrat over a (possibly shared) panel from a frozen snapshot."""
    algo = resolve_alpha_from_source(alpha_source)
    cfg = merge_finstrat_runtime_dict(dict(finstrat_config or {}), None)
    return FinStrat(fts, algo, **cfg)


def panel_dates(fts: finTs) -> List[pd.Timestamp]:
    """Sorted unique bar dates in the panel."""
    idx = fts.df.index.get_level_values("Date")
    return sorted(pd.DatetimeIndex(idx).unique())


def long_only_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Drop non-positive weights and renormalize the rest to gross 1."""
    pos = {k: float(v) for k, v in weights.items() if math.isfinite(float(v)) and float(v) > 0}
    s = sum(pos.values())
    return {k: v / s for k, v in pos.items()} if s > 0 else {}


def sleeve_weights(
    fs: FinStrat,
    signal_date: date,
    *,
    universe: Sequence[str],
    long_only: bool = True,
) -> Dict[str, float]:
    """Replay ``fs`` over every panel date up to ``signal_date``; return weights at ``signal_date``."""
    allowed = set(universe)
    dates = [d for d in panel_dates(fs._ts) if d.date() <= signal_date]
    if not dates or dates[-1].date() != signal_date:
        last = dates[-1].date().isoformat() if dates else "none"
        raise StaleDataError(f"latest bar {last} != signal date {signal_date.isoformat()}")
    fs.reset_pipeline_state()
    last_out: Optional[tuple[pd.Timestamp, List[str], np.ndarray]] = None
    for dt in dates:
        try:
            names = [n for n in fs.tickers_at(dt) if n in allowed]
        except ValueError:
            continue
        if not names:
            continue
        raw = fs.pass_(None, 1.0, tickers=names, execution_date=dt)
        last_out = (dt, names, np.asarray(raw, dtype=float))
    if last_out is None or last_out[0].date() != signal_date:
        raise StaleDataError(f"no signal produced for {signal_date.isoformat()}")
    _, names, arr = last_out
    w = {n: float(v) for n, v in zip(names, arr) if math.isfinite(float(v))}
    return long_only_weights(w) if long_only else w


def closes_on(fts: finTs, on: date) -> Dict[str, float]:
    """Close price per ticker on ``on`` (only finite, positive)."""
    df = fts.df
    dates = pd.DatetimeIndex(df.index.get_level_values("Date"))
    sub = df[dates.date == on]
    out: Dict[str, float] = {}
    for (ticker, _), row in sub.iterrows():
        c = float(row.get("Close", np.nan))
        if math.isfinite(c) and c > 0:
            out[str(ticker)] = c
    return out
