"""Broad financial headlines via Yahoo Search (yfinance ``Search``)."""

# TODO(market-data-router): Keep headlines on a non-OHLCV dataset lane; still centralize Yahoo session + labeling.

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from api.schemas.models import MarketHeadlineItem
from api.services.market_exceptions import MarketProviderError
from shunya.integration.yahoo_public import YahooPublicAdapter

_log = logging.getLogger(__name__)

_DEFAULT_QUERY = "stock market"


def _published_at_from_raw(pub_raw: Any, ppt_raw: Any) -> str | None:
    if isinstance(pub_raw, str) and pub_raw.strip():
        return pub_raw.strip()
    if isinstance(ppt_raw, (int, float)):
        ts = int(ppt_raw)
        if ts > 10_000_000_000:
            ts //= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return None


def _headline_from_raw(item: dict[str, Any]) -> MarketHeadlineItem | None:
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    link = item.get("link")
    pub = item.get("publisher")
    published_at = _published_at_from_raw(item.get("pubDate"), item.get("providerPublishTime"))
    return MarketHeadlineItem(
        title=title.strip(),
        link=link if isinstance(link, str) else None,
        publisher=pub if isinstance(pub, str) else None,
        published_at=published_at,
    )


def fetch_market_headlines(limit: int, *, session: Any | None = None) -> list[MarketHeadlineItem]:
    cap = max(1, min(limit, 100))
    adapter = YahooPublicAdapter(session=session)
    try:
        s = adapter.search_headlines(_DEFAULT_QUERY, news_count=cap)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance Search headlines failed: %s", exc)
        raise MarketProviderError("news provider unavailable") from exc

    raw_list = s.news or []
    out: list[MarketHeadlineItem] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        row = _headline_from_raw(item)
        if row:
            out.append(row)
        if len(out) >= cap:
            break
    return out
