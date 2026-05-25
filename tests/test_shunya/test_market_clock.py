"""Tests for :mod:`shunya.time.market_clock`."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shunya.time.market_clock import (
    TZ_INDIA_LISTED,
    TZ_US_LISTED,
    build_market_clock_snapshot,
    format_market_clock_line,
    is_us_listed_equity_regular_session_open,
)


def test_format_market_clock_line_shape() -> None:
    # Fixed instant: 2024-06-03 14:00:00 UTC = 10:00 Eastern on a Monday (EDT)
    at = datetime(2024, 6, 3, 14, 0, 0, 123_000, tzinfo=UTC)
    us = format_market_clock_line("US", TZ_US_LISTED, at)
    assert us.startswith("[US] ")
    assert "14:00:00" not in us  # not UTC wall
    assert us.endswith(".123")
    inn = format_market_clock_line("IN", TZ_INDIA_LISTED, at)
    assert inn.startswith("[IN] ")


def test_us_rth_weekday_open_window() -> None:
    # Monday 2024-06-03 14:00 New York (summer): 14:00 local = within 9:30–16:00
    at = datetime(2024, 6, 3, 18, 0, 0, tzinfo=UTC)  # 14:00 EDT
    assert is_us_listed_equity_regular_session_open(at) is True


def test_us_rth_weekend_closed() -> None:
    at = datetime(2024, 6, 1, 15, 0, 0, tzinfo=UTC)  # Saturday
    assert is_us_listed_equity_regular_session_open(at) is False


def test_us_rth_before_open() -> None:
    at = datetime(2024, 6, 3, 13, 0, 0, tzinfo=UTC)  # Mon ~09:00 Eastern
    assert is_us_listed_equity_regular_session_open(at) is False


def test_us_rth_at_close_boundary() -> None:
    # 16:00:00 Eastern exactly should be **closed** (end exclusive)
    at = datetime(2024, 6, 3, 20, 0, 0, tzinfo=UTC)  # 16:00 EDT
    assert is_us_listed_equity_regular_session_open(at) is False


def test_build_snapshot_has_all_fields() -> None:
    snap = build_market_clock_snapshot(datetime(2024, 6, 3, 18, 0, 0, tzinfo=UTC))
    assert snap.utc_iso.endswith("Z") or "+" in snap.utc_iso or "-" in snap.utc_iso
    assert "[US]" in snap.us_line
    assert "[IN]" in snap.in_line
    assert isinstance(snap.us_listed_rth_open, bool)
    assert isinstance(snap.alpaca_l1_us_equities_stream_allowed, bool)


def test_ignore_rth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from shunya.time import market_clock as mc

    monkeypatch.setenv("SHUNYA_ALPACA_L1_IGNORE_US_RTH", "1")
    at = datetime(2024, 6, 1, 15, 0, 0, tzinfo=UTC)
    assert mc.alpaca_l1_us_equities_stream_allowed(at) is True
