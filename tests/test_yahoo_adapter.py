"""Unit tests for :mod:`shunya.integration.yahoo_public`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shunya.integration.yahoo_public import YahooPublicAdapter


def test_yahoo_public_adapter_uses_injected_session() -> None:
    sess = MagicMock(name="session")
    adapter = YahooPublicAdapter(session=sess)
    assert adapter.session is sess


def test_yahoo_public_adapter_download_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import shunya.integration.yahoo_public as yp

    captured: dict = {}

    def _fake_download(*_a, **kwargs):
        captured["session"] = kwargs.get("session")
        import pandas as pd

        return pd.DataFrame()

    monkeypatch.setattr(yp.yf, "download", _fake_download)
    adapter = YahooPublicAdapter(session=MagicMock())
    df = adapter.download_daily_snapshot(["SPY"])
    assert df.empty
    assert captured["session"] is adapter.session
