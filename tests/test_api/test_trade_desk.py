"""Trade desk HTTP routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.settings import get_settings


@pytest.fixture(autouse=True)
def _clear_api_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def _worker_no_db(stop):  # noqa: ANN001
    await stop.wait()


def test_trade_paper_cycle_requires_configured_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/trade/paper/cycle",
            json={
                "capital": 1000.0,
                "execution_date": "2024-01-03",
                "use_demo_pcs": True,
            },
        )
    assert r.status_code == 503


def test_trade_paper_cycle_invalid_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "expected")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/trade/paper/cycle",
            headers={"X-Shunya-Trade-Desk-Token": "wrong"},
            json={
                "capital": 1000.0,
                "execution_date": "2024-01-03",
                "use_demo_pcs": True,
            },
        )
    assert r.status_code == 401


def test_trade_paper_cycle_disabled_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            "/trade/paper/cycle",
            headers={"X-Shunya-Trade-Desk-Token": "tok"},
            json={
                "capital": 1000.0,
                "execution_date": "2024-01-03",
                "use_demo_pcs": True,
            },
        )
    assert r.status_code == 503


def test_trade_paper_cycle_mock_desk(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app
    from api.trade_desk_runtime import TradeDeskRuntime
    from shunya.integration.alpaca_settings import AlpacaRuntimeSettings
    from shunya.live.desk import PaperCycleResult

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    fake = PaperCycleResult(
        correlation_id="c1",
        blend_mode="target",
        tickers=("SPY",),
        targets_constructed={"SPY": 100.0},
        targets_vetted={"SPY": 100.0},
        parent_intents=[],
        ems_parent_ids=[],
    )
    mock_desk = MagicMock()
    mock_desk.run_with_pcs = AsyncMock(return_value=fake)

    monkeypatch.setattr(
        "api.routers.trade_desk.InstitutionalPaperDesk",
        lambda *a, **k: mock_desk,
    )

    app = create_app()
    with TestClient(app) as client:
        settings = AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True)
        app.state.trade_desk_runtime = TradeDeskRuntime(
            settings=settings,
            trading_client=MagicMock(),
            data_client=MagicMock(),
        )
        r = client.post(
            "/trade/paper/cycle",
            headers={"X-Shunya-Trade-Desk-Token": "tok"},
            json={
                "capital": 1000.0,
                "execution_date": "2024-01-03",
                "use_demo_pcs": True,
            },
        )
    assert r.status_code == 200
    body = r.json()
    assert body["correlation_id"] == "c1"
    assert body["tickers"] == ["SPY"]


def test_lifespan_raises_when_alpaca_enabled_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "true")
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("SHUNYA_ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)
    with pytest.raises(RuntimeError, match="Alpaca keys"):
        with TestClient(create_app()):
            pass
    get_settings.cache_clear()


def _trade_client_headers() -> dict[str, str]:
    return {"X-Shunya-Trade-Desk-Token": "tok"}


def test_trade_account_equity_filters_crypto_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app
    from api.trade_desk_runtime import TradeDeskRuntime
    from shunya.integration.alpaca_settings import AlpacaRuntimeSettings

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    raw_account = {
        "id": "6430154a-744f-43d8-86e9-54b619b907a4",
        "account_number": "PA3JGZNW2HIJ",
        "status": "ACTIVE",
        "crypto_status": "ACTIVE",
        "options_approved_level": 3,
        "currency": "USD",
        "equity": "100000",
        "options_buying_power": "100000",
    }
    tc = MagicMock()
    tc.get = MagicMock(return_value=raw_account)

    app = create_app()
    with TestClient(app) as client:
        app.state.trade_desk_runtime = TradeDeskRuntime(
            settings=AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True),
            trading_client=tc,
            data_client=MagicMock(),
        )
        r = client.get("/trade/account/equity", headers=_trade_client_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["equity"] == "100000"
    assert "crypto_status" not in body
    assert "options_buying_power" not in body
    assert "options_approved_level" not in body


def test_trade_account_equity_401_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app
    from api.trade_desk_runtime import TradeDeskRuntime
    from shunya.integration.alpaca_settings import AlpacaRuntimeSettings

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    app = create_app()
    with TestClient(app) as client:
        app.state.trade_desk_runtime = TradeDeskRuntime(
            settings=AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True),
            trading_client=MagicMock(),
            data_client=MagicMock(),
        )
        r = client.get("/trade/account/equity")
    assert r.status_code == 401


def test_trade_account_equity_503_no_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    app = create_app()
    with TestClient(app) as client:
        r = client.get("/trade/account/equity", headers=_trade_client_headers())
    assert r.status_code == 503


def test_trade_account_activities_pagination_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from api.main import create_app
    from api.trade_desk_runtime import TradeDeskRuntime
    from shunya.integration.alpaca_settings import AlpacaRuntimeSettings

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    rows = [
        {
            "id": "20220101::11111111-1111-1111-1111-111111111111",
            "account_id": "6430154a-744f-43d8-86e9-54b619b907a4",
            "activity_type": "FILL",
            "transaction_time": "2026-05-01T15:00:00Z",
            "type": "fill",
            "price": 100.0,
            "qty": 1.0,
            "side": "buy",
            "symbol": "SPY",
            "leaves_qty": 0.0,
            "order_id": "22222222-2222-2222-2222-222222222222",
            "cum_qty": 1.0,
            "order_status": "filled",
        }
    ]
    tc = MagicMock()
    tc.get = MagicMock(return_value=rows)

    app = create_app()
    with TestClient(app) as client:
        app.state.trade_desk_runtime = TradeDeskRuntime(
            settings=AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True),
            trading_client=tc,
            data_client=MagicMock(),
        )
        r = client.get(
            "/trade/account/activities?page_size=1",
            headers=_trade_client_headers(),
        )
    assert r.status_code == 200
    body = r.json()
    assert len(body["activities"]) == 1
    assert body["activities"][0]["symbol"] == "SPY"
    assert body["next_page_token"] == rows[0]["id"]


def test_trade_account_portfolio_history(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpaca.trading.models import PortfolioHistory

    from api.main import create_app
    from api.trade_desk_runtime import TradeDeskRuntime
    from shunya.integration.alpaca_settings import AlpacaRuntimeSettings

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    ph = PortfolioHistory(
        timestamp=[1_700_000_000],
        equity=[100_000.0],
        profit_loss=[0.0],
        profit_loss_pct=[0.0],
        base_value=100_000.0,
        timeframe="1D",
        cashflow={},
    )
    tc = MagicMock()
    tc.get_portfolio_history = MagicMock(return_value=ph)

    app = create_app()
    with TestClient(app) as client:
        app.state.trade_desk_runtime = TradeDeskRuntime(
            settings=AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True),
            trading_client=tc,
            data_client=MagicMock(),
        )
        r = client.get(
            "/trade/account/portfolio-history?period=1M",
            headers=_trade_client_headers(),
        )
    assert r.status_code == 200
    body = r.json()
    assert body["equity"] == [100_000.0]
    assert body["timeframe"] == "1D"


def test_trade_account_configurations_get_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpaca.trading.enums import DTBPCheck, PDTCheck, TradeConfirmationEmail
    from alpaca.trading.models import AccountConfiguration

    from api.main import create_app
    from api.trade_desk_runtime import TradeDeskRuntime
    from shunya.integration.alpaca_settings import AlpacaRuntimeSettings

    get_settings.cache_clear()
    monkeypatch.setenv("SHUNYA_API_TRADE_DESK_TOKEN", "tok")
    monkeypatch.setenv("SHUNYA_API_ALPACA_ENABLED", "false")
    monkeypatch.setattr("api.main.backtest_worker_loop", _worker_no_db)

    cfg = AccountConfiguration(
        dtbp_check=DTBPCheck.BOTH,
        fractional_trading=True,
        max_margin_multiplier="2",
        no_shorting=False,
        pdt_check=PDTCheck.ENTRY,
        suspend_trade=False,
        trade_confirm_email=TradeConfirmationEmail.NONE,
        ptp_no_exception_entry=False,
        max_options_trading_level=3,
    )
    tc = MagicMock()
    tc.get_account_configurations = MagicMock(return_value=cfg)
    tc.set_account_configurations = MagicMock(side_effect=lambda c: c)

    app = create_app()
    with TestClient(app) as client:
        app.state.trade_desk_runtime = TradeDeskRuntime(
            settings=AlpacaRuntimeSettings(api_key_id="k", secret_key="s", paper=True),
            trading_client=tc,
            data_client=MagicMock(),
        )
        g = client.get("/trade/account/configurations", headers=_trade_client_headers())
        assert g.status_code == 200
        assert g.json()["fractional_trading"] is True

        p = client.patch(
            "/trade/account/configurations",
            headers=_trade_client_headers(),
            json={"suspend_trade": True},
        )
    assert p.status_code == 200
    assert p.json()["suspend_trade"] is True
    tc.set_account_configurations.assert_called_once()
