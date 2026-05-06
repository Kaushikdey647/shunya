"""Yahoo Finance predefined equity screeners (gainers / losers / most active)."""

from __future__ import annotations

import logging
from typing import Any, Literal

import yfinance as yf

from api.schemas.models import MarketMoverRow
from api.services.market_exceptions import MarketProviderError
from shunya.data.yfinance_session import build_yfinance_session

_log = logging.getLogger(__name__)

_PREDEFINED_SCREEN: dict[str, str] = {
    "gainers": "day_gainers",
    "losers": "day_losers",
    "active": "most_actives",
}


def fetch_movers(
    kind: Literal["gainers", "losers", "active"],
    limit: int,
    *,
    session: Any | None = None,
) -> list[MarketMoverRow]:
    """Run a predefined Yahoo screener and normalize rows."""
    key = _PREDEFINED_SCREEN.get(kind)
    if key is None:
        raise ValueError("invalid movers kind")
    cap = max(1, min(limit, 250))
    sess = session if session is not None else build_yfinance_session()
    try:
        result = yf.screen(key, count=cap, session=sess)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance screen failed kind=%s: %s", kind, exc)
        raise MarketProviderError("screener unavailable") from exc

    quotes = result.get("quotes") if isinstance(result, dict) else None
    if not isinstance(quotes, list):
        return []

    rows: list[MarketMoverRow] = []
    for q in quotes:
        if not isinstance(q, dict):
            continue
        row = _mover_from_quote(q)
        if row:
            rows.append(row)
        if len(rows) >= cap:
            break
    return rows


def _pick_float(q: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = q.get(k)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        return x
    return None


def _pick_str(q: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = q.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return None


def _mover_from_quote(q: dict[str, Any]) -> MarketMoverRow | None:
    sym = _pick_str(q, "symbol", "ticker")
    if not sym:
        return None
    price = _pick_float(
        q,
        "regularMarketPrice",
        "regularmarketprice",
        "navPrice",
    )
    pct = _pick_float(
        q,
        "regularMarketChangePercent",
        "regularmarketchangepercent",
        "percentChange",
    )
    vol = _pick_float(
        q,
        "regularMarketVolume",
        "regularmarketvolume",
        "averageDailyVolume3Month",
        "dayVolume",
    )
    return MarketMoverRow(ticker=sym, price=price, pct_change=pct, volume=vol)
