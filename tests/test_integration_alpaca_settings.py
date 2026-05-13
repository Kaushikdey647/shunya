"""Tests for :mod:`shunya.integration.alpaca_settings`."""

from __future__ import annotations

import pytest

from shunya.integration.alpaca_settings import (
    AlpacaRuntimeSettings,
    build_trading_client,
    load_alpaca_settings_from_env,
    try_load_alpaca_settings_from_env,
)


def test_load_alpaca_settings_paper_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "kid")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "sec")
    monkeypatch.setenv("SHUNYA_ALPACA_PAPER", "false")
    s = load_alpaca_settings_from_env()
    assert s.api_key_id == "kid"
    assert s.secret_key == "sec"
    assert s.paper is False


def test_shunya_alpaca_key_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("SHUNYA_ALPACA_API_KEY_ID", "a")
    monkeypatch.setenv("SHUNYA_ALPACA_API_SECRET_KEY", "b")
    s = load_alpaca_settings_from_env()
    assert s.api_key_id == "a"
    assert s.secret_key == "b"


def test_try_load_returns_none_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_SECRET_KEY", raising=False)
    assert try_load_alpaca_settings_from_env() is None


def test_build_trading_client_uses_paper_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    c = build_trading_client(s)
    assert c is not None
