"""Unit tests for market cache TTL helpers (no DB)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from shunya.data.timescale.market_cache_lib import (
    default_market_cache_ttl_days,
    ohlcv_manifest_is_fresh,
)


def test_ohlcv_manifest_is_fresh_none() -> None:
    now = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    assert ohlcv_manifest_is_fresh(None, ttl_days=30, now=now) is False


def test_ohlcv_manifest_is_fresh_within_ttl() -> None:
    now = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    lr = now - timedelta(days=10)
    assert ohlcv_manifest_is_fresh(lr, ttl_days=30, now=now) is True


def test_ohlcv_manifest_is_fresh_stale() -> None:
    now = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    lr = now - timedelta(days=31)
    assert ohlcv_manifest_is_fresh(lr, ttl_days=30, now=now) is False


def test_ohlcv_manifest_is_fresh_naive_timestamp() -> None:
    now = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    lr = datetime(2025, 6, 1, 0, 0, 0)
    assert ohlcv_manifest_is_fresh(lr, ttl_days=30, now=now) is True


def test_default_market_cache_ttl_days_is_positive() -> None:
    assert default_market_cache_ttl_days() >= 1


def test_ohlcv_manifest_ttl_minimum_one_day(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    lr = now - timedelta(hours=12)
    assert ohlcv_manifest_is_fresh(lr, ttl_days=0, now=now) is True
