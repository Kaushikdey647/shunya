"""Tests for :mod:`shunya.data.yfinance_session`."""

from __future__ import annotations

import pytest

from shunya.data.yfinance_session import build_yfinance_session


def test_build_yfinance_session_strict_env_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YFINANCE_TLS_VERIFY", "1")
    assert build_yfinance_session() is None


def test_build_yfinance_session_returns_something_or_none() -> None:
    """With default env, either curl session or None if curl_cffi missing."""
    s = build_yfinance_session()
    assert s is None or hasattr(s, "get")
