"""Tests for :mod:`shunya.integration.alpaca_settings`."""

from __future__ import annotations

import pytest

from shunya.integration.alpaca_settings import (
    AlpacaRuntimeSettings,
    build_stock_data_stream,
    build_stock_historical_data_client,
    build_trading_client,
    build_trading_stream,
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


def test_build_trading_client_default_tls_verify_on_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TLS_VERIFY", raising=False)
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    c = build_trading_client(s)
    assert c._session.verify is not False


def test_build_trading_client_shunya_tls_zero_sets_verify_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "0")
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    c = build_trading_client(s)
    assert c._session.verify is False


def test_build_stock_historical_data_client_shunya_tls_zero_sets_verify_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "0")
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    c = build_stock_historical_data_client(s)
    assert c._session.verify is False


def test_build_trading_stream_strict_has_no_ssl_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TLS_VERIFY", raising=False)
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    ts = build_trading_stream(s)
    assert "ssl" not in ts._websocket_params


def test_build_trading_stream_relaxed_merges_ssl_into_websocket_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "0")
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    ts = build_trading_stream(s)
    assert ts._websocket_params.get("ssl") is not None
    assert ts._websocket_params.get("ping_interval") == 10


def test_build_stock_data_stream_strict_has_no_ssl_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TLS_VERIFY", raising=False)
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    ds = build_stock_data_stream(s)
    assert "ssl" not in ds._websocket_params


def test_build_stock_data_stream_relaxed_merges_ssl_into_websocket_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "0")
    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    ds = build_stock_data_stream(s)
    assert ds._websocket_params.get("ssl") is not None
    assert ds._websocket_params.get("ping_interval") == 10


def test_build_stock_data_stream_explicit_iex_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TLS_VERIFY", raising=False)
    from alpaca.data.enums import DataFeed

    s = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
    ds = build_stock_data_stream(s, feed=DataFeed.IEX)
    assert "iex" in str(ds._endpoint).lower()
