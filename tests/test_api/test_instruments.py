"""Instruments API: mocked yfinance / OHLCV resolver."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.schemas.models import (
    InstrumentFeatureAvailability,
    InstrumentFinancialLineRow,
    InstrumentFinancialStatementResponse,
    InstrumentHolderRow,
    InstrumentHoldersResponse,
    InstrumentOhlcvResponse,
    InstrumentOverviewResponse,
    InstrumentSearchNewsItem,
    InstrumentSearchQuote,
    InstrumentSearchResponse,
    InstrumentTickerNewsItem,
    InstrumentTickerNewsResponse,
    InstrumentValuationMetrics,
    OhlcvBar,
    OhlcvProvenance,
)
from api.services.instrument_ohlcv import InstrumentOhlcvResult


def test_search_requires_q() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/search")
    assert r.status_code == 422


def test_search_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(q: str) -> InstrumentSearchResponse:
        return InstrumentSearchResponse(
            quotes=[
                InstrumentSearchQuote(symbol="TEST", shortname="Test Co", exchange="NMS"),
            ],
            news=[InstrumentSearchNewsItem(title="Hello", link="https://example.com/n", publisher="X")],
            nav_links=[],
        )

    monkeypatch.setattr("api.routers.instruments._run_search", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/search?q=te")
    assert r.status_code == 200
    data = r.json()
    assert data["quotes"][0]["symbol"] == "TEST"
    assert data["news"][0]["title"] == "Hello"


def test_ohlcv_invalid_symbol() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/bad%20sym/ohlcv")
    assert r.status_code == 400


def test_ohlcv_returns_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(
        sym: str, interval: str, period: str, *, defer_storage: bool = False, route: str = "auto"
    ) -> InstrumentOhlcvResult:
        return InstrumentOhlcvResult(
            response=InstrumentOhlcvResponse(
                symbol=sym,
                interval=interval,
                period=period,
                bars=[
                    OhlcvBar(
                        time="2024-01-02T00:00:00+00:00",
                        open=1.0,
                        high=2.0,
                        low=0.5,
                        close=1.5,
                        volume=100.0,
                    ),
                ],
                provenance=OhlcvProvenance(
                    read_path="live_fetch",
                    upstream_source_id="yfinance",
                    route_rule_id="test",
                ),
                storage_status="none",
            )
        )

    monkeypatch.setattr("api.routers.instruments.resolve_instrument_ohlcv_sync", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/ohlcv?interval=1d&period=1mo")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    assert len(data["bars"]) == 1
    assert data["bars"][0]["close"] == 1.5
    assert data["provenance"]["read_path"] == "live_fetch"
    assert data["provenance"]["upstream_source_id"] == "yfinance"


def test_ingestion_run_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SHUNYA_DATABASE_URL", raising=False)
    client = TestClient(create_app())
    r = client.get("/instruments/ingestion-runs/1")
    assert r.status_code == 503


def test_ticker_news_returns_items(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(sym: str, limit: int) -> InstrumentTickerNewsResponse:
        assert sym == "ZZZ"
        assert limit == 40
        return InstrumentTickerNewsResponse(
            symbol=sym,
            news=[
                InstrumentTickerNewsItem(
                    title="Headline",
                    link="https://example.com/a",
                    publisher="Yahoo",
                    published_at="2024-06-01T12:00:00+00:00",
                ),
            ],
        )

    monkeypatch.setattr("api.routers.instruments._run_ticker_news", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/ZZZ/news")
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "ZZZ"
    assert data["news"][0]["title"] == "Headline"
    assert data["news"][0]["published_at"] == "2024-06-01T12:00:00+00:00"


def test_ticker_news_invalid_symbol() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/bad%20sym/news")
    assert r.status_code == 400


def test_normalize_instrument_kind() -> None:
    from api.services.instrument_dashboard import normalize_instrument_kind

    assert normalize_instrument_kind("EQUITY") == "equity"
    assert normalize_instrument_kind("ETF") == "etf"
    assert normalize_instrument_kind("MUTUALFUND") == "mutualfund"
    assert normalize_instrument_kind("OPTION") == "option"
    assert normalize_instrument_kind(None) == "unknown"
    assert normalize_instrument_kind("WEIRD") == "unknown"


def test_overview_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(sym: str) -> InstrumentOverviewResponse:
        assert sym == "AAPL"
        return InstrumentOverviewResponse(
            symbol=sym,
            instrument_kind="equity",
            yahoo_quote_type="EQUITY",
            short_name="Apple Inc.",
            market_cap=3e12,
            beta=1.2,
            valuation=InstrumentValuationMetrics(trailing_pe=28.5, trailing_eps=6.1),
            features=InstrumentFeatureAvailability(
                financials=True, holders=True, options_chain=True
            ),
        )

    monkeypatch.setattr("api.routers.instruments.fetch_instrument_overview", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/overview")
    assert r.status_code == 200
    data = r.json()
    assert data["instrument_kind"] == "equity"
    assert data["features"]["financials"] is True
    assert data["valuation"]["trailing_pe"] == 28.5


def test_financials_requires_statement(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/financials")
    assert r.status_code == 422


def test_financials_returns_table(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(
        symbol: str,
        *,
        statement: str,
        frequency: str,
        periods: int,
    ) -> InstrumentFinancialStatementResponse:
        return InstrumentFinancialStatementResponse(
            symbol=symbol,
            statement="income",  # type: ignore[arg-type]
            frequency="quarterly",  # type: ignore[arg-type]
            periods=["2024-01-01", "2024-04-01"],
            rows=[
                InstrumentFinancialLineRow(label="Total Revenue", values=[1e9, 1.1e9]),
            ],
        )

    monkeypatch.setattr("api.routers.instruments.fetch_instrument_financials", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/XYZ/financials?statement=income&frequency=quarterly&periods=4")
    assert r.status_code == 200
    data = r.json()
    assert data["rows"][0]["label"] == "Total Revenue"
    assert data["rows"][0]["values"] == [1e9, 1.1e9]


def test_holders_returns_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(symbol: str) -> InstrumentHoldersResponse:
        return InstrumentHoldersResponse(
            symbol=symbol,
            institutional=[
                InstrumentHolderRow(holder="Vanguard", percent_held=8.5, shares=1e9, value=1e11),
            ],
            mutual_funds=[],
            available_institutional=True,
            available_mutual_funds=False,
        )

    monkeypatch.setattr("api.routers.instruments.fetch_instrument_holders", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/IBM/holders")
    assert r.status_code == 200
    data = r.json()
    assert data["institutional"][0]["holder"] == "Vanguard"


def test_option_expirations(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.schemas.models import InstrumentOptionExpirationsResponse

    def _fake(symbol: str) -> InstrumentOptionExpirationsResponse:
        return InstrumentOptionExpirationsResponse(
            symbol=symbol, expirations=["2026-06-19", "2026-07-17"], available=True
        )

    monkeypatch.setattr("api.routers.instruments.fetch_option_expirations", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/options/expirations")
    assert r.status_code == 200
    assert r.json()["expirations"][0] == "2026-06-19"


def test_option_chain_bad_expiry() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/options/chain?expiry=short")
    assert r.status_code == 422


def test_option_chain_invalid_expiry_format() -> None:
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/options/chain?expiry=06-19-2026")
    assert r.status_code == 400


def test_option_chain_returns_legs(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.schemas.models import InstrumentOptionChainResponse, InstrumentOptionLegRow

    def _fake(symbol: str, expiry: str) -> InstrumentOptionChainResponse:
        return InstrumentOptionChainResponse(
            symbol=symbol,
            expiry=expiry,
            calls=[InstrumentOptionLegRow(strike=100.0, last=5.0, bid=4.9, ask=5.1, volume=10, open_interest=99)],
            puts=[],
            available=True,
        )

    monkeypatch.setattr("api.routers.instruments.fetch_option_chain", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/options/chain?expiry=2026-06-19")
    assert r.status_code == 200
    data = r.json()
    assert data["calls"][0]["strike"] == 100.0


def test_option_iv_heatmap(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.schemas.models import InstrumentIvHeatmapResponse

    def _fake(symbol: str, max_expirations: int) -> InstrumentIvHeatmapResponse:
        _ = max_expirations
        return InstrumentIvHeatmapResponse(
            symbol=symbol,
            expirations=["2026-06-19", "2026-07-17"],
            strikes=[100.0, 110.0],
            iv_calls=[[0.2, 0.22], [0.25, 0.28]],
            iv_puts=[[0.19, 0.21], [0.24, 0.27]],
            available=True,
        )

    monkeypatch.setattr("api.routers.instruments.fetch_option_iv_heatmap", _fake)
    client = TestClient(create_app())
    r = client.get("/instruments/AAPL/options/iv-heatmap?max_expirations=2")
    assert r.status_code == 200
    data = r.json()
    assert data["expirations"] == ["2026-06-19", "2026-07-17"]
    assert data["strikes"] == [100.0, 110.0]
    assert data["iv_calls"][0][0] == 0.2
    assert data["iv_puts"][1][1] == 0.27


def test_option_iv_heatmap_query_bounds() -> None:
    client = TestClient(create_app())
    assert client.get("/instruments/AAPL/options/iv-heatmap?max_expirations=0").status_code == 422
    assert client.get("/instruments/AAPL/options/iv-heatmap?max_expirations=41").status_code == 422
