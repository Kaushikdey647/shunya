"""Tests for :mod:`shunya.integration.tls_env`."""

from __future__ import annotations

import pytest

from shunya.integration.tls_env import tls_certificate_verification_enabled


def test_tls_unset_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_TLS_VERIFY", raising=False)
    assert tls_certificate_verification_enabled() is True


@pytest.mark.parametrize("v", ["1", "true", "YES", "On"])
def test_tls_truthy(monkeypatch: pytest.MonkeyPatch, v: str) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", v)
    assert tls_certificate_verification_enabled() is True


@pytest.mark.parametrize("v", ["0", "false", "NO", "off"])
def test_tls_falsy(monkeypatch: pytest.MonkeyPatch, v: str) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", v)
    assert tls_certificate_verification_enabled() is False


def test_tls_unknown_defaults_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHUNYA_TLS_VERIFY", "maybe")
    assert tls_certificate_verification_enabled() is True
