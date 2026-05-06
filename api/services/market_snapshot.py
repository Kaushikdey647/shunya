"""Batched daily OHLCV snapshot for macro strip / watchlist (yfinance ``download``)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from api.schemas.models import MarketSnapshotRow
from api.services.instrument_ohlcv import _flatten_ohlcv_for_symbol
from api.services.market_exceptions import MarketProviderError
from shunya.data.yfinance_session import build_yfinance_session

_log = logging.getLogger(__name__)

_SPARKLINE_MAX_BARS = 7


def build_snapshot(symbols: list[str], *, session: Any | None = None) -> list[MarketSnapshotRow]:
    """
    Download recent daily bars for all symbols in one batch and derive last price,
    session % change (last vs prior close), volume, and close series for sparklines.

    ``symbols`` must already be normalized (uppercase, validated).
    """
    if not symbols:
        return []
    sess = session if session is not None else build_yfinance_session()
    try:
        raw = yf.download(
            tickers=list(symbols),
            period="7d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=False,
            progress=False,
            session=sess,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("market snapshot yfinance download failed: %s", exc)
        raise MarketProviderError("market data unavailable") from exc

    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        raise MarketProviderError("empty market data")

    rows: list[MarketSnapshotRow] = []
    for sym in symbols:
        rows.append(_row_from_download_frame(raw, sym))
    return rows


def _row_from_download_frame(raw: pd.DataFrame, symbol: str) -> MarketSnapshotRow:
    flat = _flatten_ohlcv_for_symbol(raw, symbol)
    if flat is None or flat.empty:
        return MarketSnapshotRow(symbol=symbol, sparkline_close=[])

    flat = flat.sort_index()
    closes_s = flat["Close"].dropna()
    closes = [float(x) for x in closes_s.tail(_SPARKLINE_MAX_BARS).tolist()]
    last = float(closes_s.iloc[-1]) if len(closes_s) else None
    prev = float(closes_s.iloc[-2]) if len(closes_s) >= 2 else None
    pct: float | None = None
    if last is not None and prev is not None and prev != 0.0:
        pct = (last - prev) / prev * 100.0

    vol_raw = flat["Volume"].iloc[-1] if "Volume" in flat.columns and len(flat) else None
    volume = float(vol_raw) if vol_raw is not None and pd.notna(vol_raw) else None

    return MarketSnapshotRow(
        symbol=symbol,
        last=last,
        pct_change_1d=pct,
        volume=volume,
        sparkline_close=closes,
    )
