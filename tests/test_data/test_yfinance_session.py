"""Tests for :mod:`shunya.data.yfinance_session`."""

from __future__ import annotations

import pytest

from shunya.data.yfinance_session import build_yfinance_session


def test_build_yfinance_session_strict_default_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TLS_VERIFY", raising=False)
    assert build_yfinance_session() is None


def test_build_yfinance_session_strict_explicit_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "1")
    assert build_yfinance_session() is None


def test_build_yfinance_session_relaxed_uses_curl_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "0")
    try:
        import curl_cffi  # noqa: F401
    except ImportError:
        pytest.skip("curl_cffi not installed")
    s = build_yfinance_session()
    assert s is not None
    assert getattr(s, "verify", None) is False
