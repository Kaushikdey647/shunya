"""Live runner: session windows, submit/reconcile, idempotency, kill switch (fake broker + in-memory store)."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import jax.numpy as jnp
import pandas as pd
import pytest

from shunya.algorithm.finbt import FinBT
from shunya.algorithm.finstrat import FinStrat
from shunya.live.broker import AlpacaBroker, KillSwitchEngaged
from shunya.live.control import engage_kill, read_control
from shunya.live.runner import (
    RunnerConfig,
    allocate_fills,
    client_order_id,
    plan_sessions,
    tick,
)
from shunya.live.signals import (
    StaleDataError,
    UnsupportedForLive,
    live_strategy_warnings,
    sleeve_weights,
    validate_live_strategy,
)
from shunya.live.store import Book, Control, Sleeve, StrategyVersion
from tests.conftest import make_stub_fints
from tests.test_live.fakes import FakeBroker, MemStore, calendar, ny

UTC = timezone.utc
ALPHA = "import jax.numpy as jnp\n\ndef alpha(ctx):\n    return ctx.close.latest.astype(jnp.float32)\n"
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _panel(signal_day: date, n: int = 30):
    days = [d.day for d in calendar(signal_day - timedelta(days=60), signal_day)][-n:]
    fts = make_stub_fints(TICKERS, days)
    prices = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}
    fts.df["Close"] = [prices[t] for t, _ in fts.df.index]
    return fts, prices


def _setup(*, risk: Optional[Dict[str, Any]] = None):
    store = MemStore()
    broker = FakeBroker(lambda: read_control(store))
    book = Book(id=str(uuid.uuid4()), name="main", status="active", capital_fraction=0.5,
                risk_config=risk or {"max_single_name_fraction": 0.5})
    store.books[book.id] = book
    v = StrategyVersion("v1", "close", ALPHA, {"neutralization": "none"}, TICKERS, 60)
    s = Sleeve(id="s1", book_id=book.id, version=v, weight=1.0, status="active", ramp_days=1, runs_completed=0)
    store.sleeves[s.id] = s
    loaded: Dict[str, Any] = {}

    def loader(tickers, signal_day, lookback):
        fts, prices = _panel(signal_day)
        broker.price.update(prices)
        loaded["tickers"] = list(tickers)
        return fts

    return store, broker, book, loader, loaded


# --- session windows ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("now", "signal", "nxt", "in_window"),
    [
        (ny(date(2026, 11, 3), 20, 30), date(2026, 11, 3), date(2026, 11, 4), True),
        (ny(date(2026, 11, 4), 9, 25), date(2026, 11, 3), date(2026, 11, 4), False),  # past OPG cutoff
        (ny(date(2026, 11, 4), 11), date(2026, 11, 3), date(2026, 11, 4), False),  # session open
        (ny(date(2026, 11, 3), 20), date(2026, 11, 3), date(2026, 11, 4), False),  # before 20:15
        (ny(date(2026, 11, 25), 20, 30), date(2026, 11, 25), date(2026, 11, 27), True),  # skips Thanksgiving
        (ny(date(2026, 11, 27), 17, 20), date(2026, 11, 27), date(2026, 11, 30), True),  # half-day close 13:00
        (ny(date(2026, 11, 28), 12), date(2026, 11, 27), date(2026, 11, 30), True),  # weekend
    ],
)
def test_plan_sessions_and_window(now, signal, nxt, in_window) -> None:
    broker = FakeBroker(lambda: Control("armed"))
    plan = plan_sessions(broker, now.astimezone(UTC))
    assert (plan.signal.day, plan.next.day) == (signal, nxt)
    start, end = plan.submit_window(RunnerConfig())
    assert (start <= now < end) is in_window


def test_window_follows_dst() -> None:
    # 00:20 UTC is 20:20 EDT (in window) before the DST switch, 19:20 EST (too early) after it.
    broker = FakeBroker(lambda: Control("armed"))
    for now, expected in ((datetime(2026, 10, 31, 0, 20, tzinfo=UTC), True), (datetime(2026, 11, 4, 0, 20, tzinfo=UTC), False)):
        start, end = plan_sessions(broker, now).submit_window(RunnerConfig())
        assert (start <= now < end) is expected


# --- submit / reconcile -----------------------------------------------------------------


def test_tick_submits_whole_share_opg_orders_once() -> None:
    store, broker, book, loader, _ = _setup()
    now = ny(date(2026, 11, 3), 20, 30).astimezone(UTC)
    out = tick(store, broker, now=now, load_fin_ts=loader)
    assert out["status"] == "ok", out
    # weights ∝ close (10/20/30/40 of 100) on 50k book → 5k/10k/15k/20k → 500 shares each
    qty = {broker.orders[c].symbol: broker.orders[c].qty for c in broker.submitted}
    assert qty == {"AAA": 500, "BBB": 500, "CCC": 500, "DDD": 500}
    assert all(c == client_order_id(book.id, date(2026, 11, 4), broker.orders[c].symbol) for c in broker.submitted)
    assert [r.status for r in store.runs.values()] == ["submitted"]

    tick(store, broker, now=now + timedelta(minutes=15), load_fin_ts=loader)
    assert len(broker.submitted) == 4  # idempotent


def test_reconcile_updates_ledger_then_next_run_trades_only_deltas() -> None:
    store, broker, book, loader, _ = _setup()
    tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    broker.fill_all()
    out = tick(store, broker, now=ny(date(2026, 11, 4), 9, 50).astimezone(UTC), load_fin_ts=loader)
    assert out["status"] == "ok", out
    assert [r.status for r in store.runs.values()] == ["reconciled"]
    assert store.ledger["s1"] == {"AAA": 500, "BBB": 500, "CCC": 500, "DDD": 500}

    tick(store, broker, now=ny(date(2026, 11, 4), 20, 30).astimezone(UTC), load_fin_ts=loader)
    assert len(broker.submitted) == 4  # unchanged targets → no new orders
    assert sorted(r.status for r in store.runs.values()) == ["reconciled", "submitted"]


def test_unreconciled_run_blocks_next_submission() -> None:
    store, broker, book, loader, _ = _setup()
    tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    # never reconciled (no tick during the 11-04 session) → the next evening run must wait
    out = tick(store, broker, now=ny(date(2026, 11, 5), 2).astimezone(UTC), load_fin_ts=loader)
    assert out["books"]["main"]["blocked_by_unreconciled"] == ["2026-11-04"]
    assert len(broker.submitted) == 4


def test_ledger_mismatch_auto_halts_and_cancels() -> None:
    store, broker, book, loader, _ = _setup()
    tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    broker.fill_all()
    broker.pos["AAA"] += 7  # someone traded outside the runner
    out = tick(store, broker, now=ny(date(2026, 11, 4), 9, 50).astimezone(UTC), load_fin_ts=loader)
    assert "auto-halt" in out["books"]["main"]["error"]
    assert store.control.halted and store.control.level == "cancel"


def test_stale_data_fails_run_and_halts() -> None:
    store, broker, book, _, _ = _setup()

    def stale_loader(tickers, signal_day, lookback):
        fts, prices = _panel(signal_day - timedelta(days=1))
        broker.price.update(prices)
        return fts

    out = tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=stale_loader)
    assert "auto-halt" in out["books"]["main"]["error"]
    assert [r.status for r in store.runs.values()] == ["failed"]
    assert broker.submitted == [] and store.control.halted


def test_drawdown_guard_halts_before_trading() -> None:
    store, broker, book, loader, _ = _setup(risk={"max_single_name_fraction": 0.5, "max_drawdown": 0.1})
    store.books[book.id] = replace(book, hwm_equity=120_000.0, last_equity=100_000.0, risk_config=book.risk_config)
    out = tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    assert "drawdown" in out["books"]["main"]["error"]
    assert broker.submitted == []


# --- kill switch ------------------------------------------------------------------------


def test_halted_flag_blocks_tick() -> None:
    store, broker, book, loader, _ = _setup()
    store.control = Control(state="halted", level="halt", reason="manual")
    out = tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    assert out["status"] == "halted" and broker.submitted == []
    assert store.heartbeat == "halted"


def test_unreadable_control_fails_closed() -> None:
    store, broker, book, loader, _ = _setup()
    store.fail_control = True
    out = tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    assert out["status"] == "halted" and broker.submitted == []


def test_kill_mid_run_stops_remaining_orders() -> None:
    store, broker, book, loader, _ = _setup()

    def kill_after_first(b: FakeBroker) -> None:
        engage_kill(store, b, level="cancel", reason="operator", actor="test")

    broker.after_submit = kill_after_first
    out = tick(store, broker, now=ny(date(2026, 11, 3), 20, 30).astimezone(UTC), load_fin_ts=loader)
    assert len(broker.submitted) == 1
    assert "kill switch" in out["books"]["main"]["error"]
    assert broker.orders[broker.submitted[0]].status == "canceled"
    assert [r.status for r in store.runs.values()] == ["failed"]


def test_flatten_closes_positions() -> None:
    store, broker, _, _, _ = _setup()
    broker.pos = {"AAA": 10}
    out = engage_kill(store, broker, level="flatten", reason="x", actor="test")
    assert out["positions_closed"] == 1 and broker.pos == {}
    assert store.control.level == "flatten" and store.audits[-1][1] == "kill"


# --- allocation / signals ---------------------------------------------------------------


def test_allocate_fills_internal_cross_and_partial() -> None:
    rows = [
        {"sleeve_id": "a", "symbol": "X", "target_shares": 10, "prev_shares": 0},
        {"sleeve_id": "b", "symbol": "X", "target_shares": 0, "prev_shares": 4},
    ]
    assert allocate_fills(rows, {"X": 6}) == {"a": {"X": 10}, "b": {"X": 0}}  # full fill: exact requests
    out = allocate_fills(rows, {"X": 3})  # partial: proportional
    assert out["a"]["X"] + out["b"]["X"] == pytest.approx(4 + 3)
    rows2 = [
        {"sleeve_id": "a", "symbol": "Y", "target_shares": 5, "prev_shares": 0},
        {"sleeve_id": "b", "symbol": "Y", "target_shares": 0, "prev_shares": 5},
    ]
    assert allocate_fills(rows2, {}) == {"a": {"Y": 5}, "b": {"Y": 0}}  # crossed internally, no order


def test_sleeve_weights_replay_matches_finbt_with_decay() -> None:
    days = [d.day for d in calendar(date(2026, 9, 1), date(2026, 10, 30))]
    fts = make_stub_fints(TICKERS, days)
    pos = {d: i for i, d in enumerate(days)}
    fts.df["Close"] = [100.0 + (pos[d.date()] % 7) * (1 + TICKERS.index(t)) for t, d in fts.df.index]

    def algo(ctx):
        return ctx.close.latest.astype(jnp.float32)

    fs_bt = FinStrat(fts, algo, neutralization="market", decay_mode="ema", decay=0.6)
    bt = FinBT(fs_bt, fts, cash=1_000_000.0).run()
    hist = bt.results(show=False)["target_history"]
    last_dt, last_targets = hist[-1]
    gross = sum(abs(v) for v in last_targets.values())

    fs_live = FinStrat(fts, algo, neutralization="market", decay_mode="ema", decay=0.6)
    w = sleeve_weights(fs_live, pd.Timestamp(last_dt).date(), universe=TICKERS, long_only=False)
    for t in TICKERS:
        assert w[t] == pytest.approx(last_targets[t] / gross, abs=1e-5)


def test_sleeve_weights_rejects_stale_panel() -> None:
    fts, _ = _panel(date(2026, 11, 2))
    fs = FinStrat(fts, lambda ctx: ctx.close.latest.astype(jnp.float32), neutralization="none")
    with pytest.raises(StaleDataError):
        sleeve_weights(fs, date(2026, 11, 3), universe=TICKERS)


def test_live_strategy_warns_on_volume() -> None:
    assert live_strategy_warnings(ALPHA) == []
    assert "IEX" in live_strategy_warnings(ALPHA.replace("ctx.close.latest", "ctx.adj_volume.latest"))[0]


def test_validate_live_strategy_rejects_unsupported() -> None:
    assert validate_live_strategy(ALPHA, {"neutralization": "market"})["neutralization"] == "market"
    with pytest.raises(UnsupportedForLive):
        validate_live_strategy(ALPHA, {"neutralization": "sector"})
    with pytest.raises(UnsupportedForLive):
        validate_live_strategy(ALPHA.replace("ctx.close.latest", "fun.roe"), {})


# --- AlpacaBroker guard -------------------------------------------------------------------


def test_alpaca_broker_refuses_when_halted_and_reuses_duplicate_ids() -> None:
    from alpaca.common.exceptions import APIError
    from alpaca.trading.enums import TimeInForce

    client = MagicMock()
    state = {"ctl": Control("halted", "halt", "x")}
    b = AlpacaBroker(client, lambda: state["ctl"])
    with pytest.raises(KillSwitchEngaged):
        b.submit_opg(symbol="AAA", side="buy", qty=1, client_order_id="c1")
    client.submit_order.assert_not_called()

    state["ctl"] = Control("armed")
    client.submit_order.side_effect = APIError('{"code":40010001,"message":"client_order_id must be unique"}')
    existing = MagicMock(client_order_id="c1", id="A1", symbol="AAA", side="buy", qty="1", status="accepted",
                         filled_qty="0", filled_avg_price=None, time_in_force="opg")
    client.get_order_by_client_id.return_value = existing
    v = b.submit_opg(symbol="AAA", side="buy", qty=1, client_order_id="c1")
    assert v.alpaca_id == "A1"
    req = client.submit_order.call_args.args[0]
    assert req.time_in_force == TimeInForce.OPG and req.qty == 1
