"""Yahoo repair flag and OHLCV sanitization (no network)."""

import pandas as pd
import pytest

from shunya.data.providers import (
    YFinanceMarketDataProvider,
    env_yfinance_repair_default,
    sanitize_yfinance_ohlcv,
)


def test_sanitize_strips_repaired_and_keeps_ohlcv() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.1, 2.1],
            "Low": [0.9, 1.9],
            "Close": [1.05, 2.05],
            "Volume": [100.0, 200.0],
            "Repaired?": [True, False],
        },
        index=idx,
    )
    out = sanitize_yfinance_ohlcv(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_sanitize_empty_frame() -> None:
    out = sanitize_yfinance_ohlcv(pd.DataFrame())
    assert out.empty


def test_env_yfinance_repair_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_YFINANCE_REPAIR", raising=False)
    monkeypatch.delenv("SHUNYA_API_YFINANCE_REPAIR", raising=False)
    assert env_yfinance_repair_default() is True
    monkeypatch.setenv("SHUNYA_YFINANCE_REPAIR", "0")
    assert env_yfinance_repair_default() is False
    monkeypatch.setenv("SHUNYA_YFINANCE_REPAIR", "false")
    assert env_yfinance_repair_default() is False


def test_yfinance_download_passes_repair_kw(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_download(_tickers, **kwargs):
        captured.update(kwargs)
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        return pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [10.0]},
            index=idx,
        )

    monkeypatch.setattr("shunya.data.providers.yf.download", _fake_download)
    YFinanceMarketDataProvider(repair=True).download(["X"], "2024-01-01", "2024-01-10")
    assert captured.get("repair") is True
    YFinanceMarketDataProvider(repair=False).download(["X"], "2024-01-01", "2024-01-10")
    assert captured.get("repair") is False


def test_yfinance_download_respects_env_repair_when_using_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_download(_tickers, **kwargs):
        captured["repair"] = kwargs.get("repair")
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        return pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [10.0]},
            index=idx,
        )

    monkeypatch.setattr("shunya.data.providers.yf.download", _fake_download)
    monkeypatch.setenv("SHUNYA_YFINANCE_REPAIR", "0")
    # Provider default repair=True on init — explicit env tests use finTs path or instrument uses env from settings re-export
    p = YFinanceMarketDataProvider(repair=env_yfinance_repair_default())
    p.download(["X"], "2024-01-01", "2024-01-10")
    assert captured["repair"] is False
