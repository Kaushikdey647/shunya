from __future__ import annotations

from backtest_api.data_service import compute_data_summary
from backtest_api.schemas.models import DataSummaryRequest
from tests.conftest import make_stub_fints


def test_compute_data_summary_with_stub(monkeypatch) -> None:
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)

    def _fake_build(_req) -> object:
        return fts

    monkeypatch.setattr("backtest_api.data_service.build_fin_ts", _fake_build)
    req = DataSummaryRequest(
        start_date="2020-01-02",
        end_date="2020-01-06",
        ticker_list=tickers,
        market_data_provider="yfinance",
    )
    out = compute_data_summary(req)
    assert out.tickers == tickers
    assert len(out.nan_counts) == 2
    assert len(out.per_ticker_metrics) == 2
    for row in out.per_ticker_metrics:
        assert row.ticker in tickers
