"""Tests for AlpacaExecutionAdapter submission and status observation."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.algorithm.execution import AlpacaExecutionAdapter


def _mock_client():
    client = MagicMock()
    asset = MagicMock()
    asset.tradable = True
    asset.fractionable = True
    asset.shortable = True
    client.get_asset.return_value = asset
    return client


def test_submit_delta_orders_dry_run_sets_statuses():
    client = _mock_client()
    adapter = AlpacaExecutionAdapter(client)
    attempts = adapter.submit_delta_orders(
        {"AAPL": 100.0},
        min_order_notional=1.0,
        dry_run=True,
        correlation_id="cid",
    )
    assert len(attempts) == 1
    assert attempts[0].success is True
    assert attempts[0].initial_status == "dry_run"
    assert attempts[0].final_status == "dry_run"


def test_observe_submitted_orders_updates_final_status_and_fill_fields():
    client = _mock_client()
    submit_order = MagicMock()
    submit_order.id = "oid-1"
    submit_order.status = "new"
    client.submit_order.return_value = submit_order

    observed = MagicMock()
    observed.status = "filled"
    observed.filled_qty = "5"
    observed.filled_avg_price = "123.45"
    client.get_order_by_id.return_value = observed

    adapter = AlpacaExecutionAdapter(client)
    attempts = adapter.submit_delta_orders(
        {"AAPL": 100.0},
        min_order_notional=1.0,
        dry_run=False,
        correlation_id="cid",
    )
    out = adapter.observe_submitted_orders(
        attempts,
        max_polls=1,
        poll_interval_seconds=0.0,
    )
    assert out[0].success is True
    assert out[0].final_status == "filled"
    assert out[0].filled_qty == 5.0
    assert out[0].filled_avg_price == 123.45
