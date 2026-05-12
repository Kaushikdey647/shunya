"""Tests for backtest analytics (FF factors, quantiles, tearsheet helpers)."""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from shunya.algorithm.backtest_analytics import (
    compute_ff_single_factor,
    compute_return_quantiles,
    compute_tearsheet_summary,
)
from shunya.data import ff_factors as ff_mod


def _minimal_ff_zip_bytes() -> bytes:
    lines = [
        "header",
        ",Mkt-RF,SMB,HML,RF",
    ]
    base = pd.Timestamp("2020-01-02")
    for i in range(15):
        d = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        lines.append(f"{d},   0.10,    0.05,    0.02,    0.01")
    csv = "\n".join(lines) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("F-F_Research_Data_Factors_daily.csv", csv)
    return buf.getvalue()


def test_compute_return_quantiles_basic() -> None:
    s = pd.Series([0.01, -0.02, 0.0, 0.015, -0.001])
    q = compute_return_quantiles(s)
    assert q["count"] == 5
    assert q["median"] is not None


def test_compute_tearsheet_summary_skew() -> None:
    idx = pd.date_range("2024-01-02", periods=30, freq="D")
    rng = pd.Series([0.02 * ((-1) ** i) for i in range(30)], dtype=float, index=idx)
    eq = pd.DataFrame({"Equity": 100_000.0 * (1.0 + rng).cumprod()}, index=idx)
    ts = compute_tearsheet_summary(eq, periods_per_year=252.0, max_drawdown_len=3)
    assert ts["ann_volatility_pct"] is not None
    assert len(ts["worst_bar_returns_pct"]) >= 1


def test_compute_ff_single_factor_with_mock_fetch() -> None:
    ff_mod.clear_ff_factors_cache()
    zbytes = _minimal_ff_zip_bytes()

    def _fake_fetch() -> pd.DataFrame:
        return ff_mod.load_ff_factors_daily_from_zip_bytes(zbytes)

    idx = pd.date_range("2020-01-02", periods=15, freq="D")
    r = pd.Series([0.001 * ((-1) ** i) for i in range(15)], index=idx)
    eqv = 100_000.0 * (1.0 + r).cumprod()
    eq = pd.DataFrame({"Equity": eqv.values}, index=idx)
    out = compute_ff_single_factor(eq, fetch_factors=_fake_fetch)
    assert "factors" in out
    fac = out["factors"]
    assert "Mkt-RF" in fac
    assert fac["Mkt-RF"]["beta"] is not None
    assert fac["Mkt-RF"]["n"] >= 10
    ff_mod.clear_ff_factors_cache()
