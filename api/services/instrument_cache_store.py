"""Shared Timescale JSON cache helpers for yfinance instrument documents."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from api.settings import get_settings
from shunya.data.timescale.market_cache_lib import (
    FINSTRUMENT_DOC_SOURCE_YFINANCE,
    get_instrument_document_payload_if_fresh,
    try_market_cache_dsn,
    upsert_instrument_document_sync,
)

_log = logging.getLogger(__name__)

_TCacheModel = TypeVar("_TCacheModel")


def instrument_yfinance_document_get(
    model_cls: type[_TCacheModel],
    *,
    symbol: str,
    resource_type: str,
    resource_key: str = "",
) -> _TCacheModel | None:
    dsn = try_market_cache_dsn()
    if not dsn:
        return None
    ttl = get_settings().market_data_cache_ttl_days
    raw = get_instrument_document_payload_if_fresh(
        dsn,
        ticker=symbol,
        source=FINSTRUMENT_DOC_SOURCE_YFINANCE,
        resource_type=resource_type,
        resource_key=resource_key,
        ttl_days=ttl,
    )
    if raw is None:
        return None
    try:
        return model_cls.model_validate(raw)  # type: ignore[no-any-return]
    except Exception as exc:  # noqa: BLE001
        _log.warning("instrument cache invalid %s %s: %s", resource_type, symbol, exc)
        return None


def instrument_yfinance_document_put(
    *,
    symbol: str,
    resource_type: str,
    resource_key: str,
    obj: Any,
) -> None:
    dsn = try_market_cache_dsn()
    if not dsn:
        return
    try:
        upsert_instrument_document_sync(
            dsn,
            ticker=symbol,
            source=FINSTRUMENT_DOC_SOURCE_YFINANCE,
            resource_type=resource_type,
            resource_key=resource_key,
            payload=obj.model_dump(mode="json"),
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("instrument cache upsert failed %s %s: %s", resource_type, symbol, exc)
