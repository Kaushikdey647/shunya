"""Backtest runner: auto-enable fundamentals when alpha source uses ``fun.*``."""

from __future__ import annotations

from api.runner import _backtest_fin_ts_auto_fundamentals
from api.schemas.models import BacktestCreate, FinTsRequest


def _minimal_body(*, attach_fundamentals: bool = False) -> BacktestCreate:
    return BacktestCreate(
        alpha_id="00000000-0000-4000-8000-000000000001",
        fin_ts=FinTsRequest(
            start_date="2020-01-01",
            end_date="2020-06-01",
            ticker_list=["AAA"],
            attach_fundamentals=attach_fundamentals,
        ),
    )


def test_auto_fundamentals_off_when_no_fun_reference() -> None:
    body = _minimal_body()
    out = _backtest_fin_ts_auto_fundamentals(
        body,
        "return cs.rank(ctx.close)",
    )
    assert out.fin_ts.attach_fundamentals is False


def test_auto_fundamentals_on_for_fun_dot() -> None:
    body = _minimal_body()
    out = _backtest_fin_ts_auto_fundamentals(
        body,
        "return cs.rank(fun.Revenue - fun.Debt_To_Equity)",
    )
    assert out.fin_ts.attach_fundamentals is True


def test_auto_fundamentals_on_for_ctx_fun() -> None:
    body = _minimal_body()
    out = _backtest_fin_ts_auto_fundamentals(
        body,
        "return cs.rank(ctx.fun.Revenue)",
    )
    assert out.fin_ts.attach_fundamentals is True


def test_auto_fundamentals_idempotent_when_already_enabled() -> None:
    body = _minimal_body(attach_fundamentals=True)
    out = _backtest_fin_ts_auto_fundamentals(body, "return fun.Revenue")
    assert out.fin_ts.attach_fundamentals is True
