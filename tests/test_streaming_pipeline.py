"""Tests for the streaming tick-to-trade components."""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from shunya.algorithm.execution import OrderAttempt
from shunya.algorithm.finstrat import FinStrat
from shunya.algorithm.order_manager import OrderManager
from shunya.algorithm.streaming_runner import StreamingRunner
from shunya.algorithm.orders import OpenOrderView
from shunya.streaming.buffer import SymbolRingBuffer
from shunya.streaming.events import trade_event
from shunya.streaming.snapshot import MicroBarAggregator, SnapshotBuilder

from tests.conftest import make_stub_fints


class _FakeAdapter:
    def __init__(self, *, open_orders: list[OpenOrderView] | None = None) -> None:
        self._open_orders = list(open_orders or [])
        self.submissions: list[dict[str, float]] = []

    def submit_delta_orders(self, deltas, *, min_order_notional, dry_run, correlation_id):
        self.submissions.append(dict(deltas))
        return [
            OrderAttempt(
                symbol=symbol,
                client_order_id=f"{correlation_id}-{symbol}",
                side="BUY" if delta > 0 else "SELL",
                notional=abs(float(delta)),
                success=True,
                initial_status="dry_run" if dry_run else "new",
                final_status="dry_run" if dry_run else "new",
            )
            for symbol, delta in deltas.items()
            if abs(float(delta)) >= min_order_notional
        ]

    def observe_submitted_orders(self, attempts, *, max_polls, poll_interval_seconds):
        return attempts

    def submit_orders(self, orders, *, dry_run, correlation_id):
        raise AssertionError("submit_orders should not be used in these tests")

    def get_positions(self):
        return {}

    def is_market_open(self):
        return True

    def buying_power(self):
        return 100_000.0

    def cancel_open_orders(self):
        self._open_orders = []

    def list_open_orders(self):
        return list(self._open_orders)


def test_process_raw_scores_matches_pass_pipeline():
    tickers = ["AAA", "BBB"]
    dates = ["2020-01-02", "2020-01-03"]
    fts = make_stub_fints(tickers, dates, base_price=100.0)
    fts.df.loc[("AAA", dates[-1]), "Close"] = 110.0
    fts.df.loc[("BBB", dates[-1]), "Close"] = 90.0

    def alpha(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(fts, alpha, neutralization="market")
    names = fs.tickers_at(dates[-1])
    raw = fs.scores_at(dates[-1], tickers=names)

    fs.reset_pipeline_state()
    via_process = fs.process_raw_scores(
        raw,
        10_000.0,
        tickers=names,
        execution_date=dates[-1],
    )
    fs.reset_pipeline_state()
    via_pass = fs.pass_(None, 10_000.0, tickers=names, execution_date=dates[-1])
    np.testing.assert_allclose(np.asarray(via_process), np.asarray(via_pass))


def test_symbol_ring_buffer_is_fifo():
    buf = SymbolRingBuffer("AAPL", capacity=2)
    buf.append(trade_event("AAPL", "2024-01-01 09:30:00", 100.0))
    buf.append(trade_event("AAPL", "2024-01-01 09:30:01", 101.0))
    buf.append(trade_event("AAPL", "2024-01-01 09:30:02", 102.0))
    assert len(buf) == 2
    prices = [event.ltp for event in buf.events()]
    assert prices == [101.0, 102.0]


def test_snapshot_builder_aligns_asynchronous_micro_bars():
    aggregator = MicroBarAggregator(bar_interval="1s", lookback=3)
    aggregator.observe(trade_event("AAA", "2024-01-01 09:30:00.100", 10.0, size=1))
    aggregator.observe(trade_event("BBB", "2024-01-01 09:30:00.200", 20.0, size=1))
    aggregator.observe(trade_event("AAA", "2024-01-01 09:30:01.100", 11.0, size=2))
    aggregator.observe(trade_event("BBB", "2024-01-01 09:30:01.300", 21.0, size=3))

    snapshot = SnapshotBuilder(lookback=3, fill_policy="ffill").build(
        aggregator,
        ["AAA", "BBB"],
    )

    assert snapshot.close.shape == (2, 2)
    np.testing.assert_allclose(snapshot.close[:, 0], np.array([10.0, 11.0]))
    np.testing.assert_allclose(snapshot.close[:, 1], np.array([20.0, 21.0]))
    np.testing.assert_allclose(snapshot.adj_volume[:, 0], np.array([1.0, 2.0]))
    np.testing.assert_allclose(snapshot.adj_volume[:, 1], np.array([1.0, 3.0]))


def test_order_manager_skips_symbols_with_open_orders():
    adapter = _FakeAdapter(
        open_orders=[OpenOrderView(symbol="AAA", order_id="open-1", status="new")]
    )
    manager = OrderManager(adapter, min_order_notional=1.0)
    batch = manager.submit_targets(
        {"AAA": 1_000.0, "BBB": -1_000.0},
        prices={"AAA": 100.0, "BBB": 100.0},
        correlation_id="cid",
        dry_run=True,
    )
    assert batch.skipped_symbols == ["AAA"]
    assert [attempt.symbol for attempt in batch.order_attempts] == ["BBB"]
    assert adapter.submissions[-1] == {"BBB": -1000.0}


def test_streaming_runner_sector_neutralization_without_group_labels():
    fts = make_stub_fints(["AAA", "BBB"], ["2024-01-01"], base_price=100.0)
    fts.df.loc[("AAA", "2024-01-01"), "Sector"] = "Tech"
    fts.df.loc[("BBB", "2024-01-01"), "Sector"] = "Energy"

    def alpha(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(fts, alpha, neutralization="sector")
    adapter = _FakeAdapter()
    manager = OrderManager(adapter, min_order_notional=1.0)
    runner = StreamingRunner(fs, manager, lookback=4, bar_interval="1s", max_concurrent_symbols=2)
    runner.set_target_universe(["AAA", "BBB"])
    runner.on_event(
        trade_event("AAA", "2024-01-01 09:30:00.100", 110.0, size=1),
        capital=10_000.0,
        dry_run=True,
    )
    runner.on_event(
        trade_event("BBB", "2024-01-01 09:30:00.100", 90.0, size=1),
        capital=10_000.0,
        dry_run=True,
    )
    decision = runner.on_event(
        trade_event("AAA", "2024-01-01 09:30:01.100", 111.0, size=1),
        capital=10_000.0,
        dry_run=True,
    )
    assert decision is not None
    assert set(decision.targets_usd) == {"AAA", "BBB"}


def test_streaming_runner_reuses_finstrat_and_submits_dry_run_orders():
    fts = make_stub_fints(["AAA", "BBB"], ["2024-01-01"], base_price=100.0)

    def alpha(ctx) -> jnp.ndarray:
        return ctx.close.latest.astype(jnp.float32)

    fs = FinStrat(fts, alpha, neutralization="market")
    adapter = _FakeAdapter()
    manager = OrderManager(adapter, min_order_notional=1.0)
    runner = StreamingRunner(
        fs,
        manager,
        lookback=4,
        bar_interval="1s",
        max_concurrent_symbols=2,
    )
    runner.set_target_universe(["AAA", "BBB"])
    runner.on_event(
        trade_event("AAA", "2024-01-01 09:30:00.100", 110.0, size=1),
        capital=10_000.0,
        dry_run=True,
    )
    runner.on_event(
        trade_event("BBB", "2024-01-01 09:30:00.100", 90.0, size=1),
        capital=10_000.0,
        dry_run=True,
    )
    decision = runner.on_event(
        trade_event("AAA", "2024-01-01 09:30:01.100", 111.0, size=1),
        capital=10_000.0,
        dry_run=True,
    )

    assert decision is not None
    assert set(decision.targets_usd) == {"AAA", "BBB"}
    assert len(decision.order_batch.order_attempts) == 2
    assert decision.order_batch.order_attempts[0].initial_status == "dry_run"
