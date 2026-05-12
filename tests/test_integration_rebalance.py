"""End-to-end mocked rebalance integration tests (adapter layer, no FinTrade)."""

from __future__ import annotations

from unittest.mock import MagicMock

from shunya.algorithm.execution import AlpacaExecutionAdapter


def _mock_client():
    client = MagicMock()
    asset = MagicMock()
    asset.tradable = True
    asset.fractionable = True
    asset.shortable = True
    client.get_asset.return_value = asset
    return client


def _client_with_status(status: str):
    client = _mock_client()
    order = MagicMock()
    order.id = "oid-x"
    order.status = "new"
    client.submit_order.return_value = order
    obs = MagicMock()
    obs.status = status
    obs.filled_qty = "0"
    obs.filled_avg_price = None
    client.get_order_by_id.return_value = obs
    return client


def test_integration_partial_fill_flow_surfaces_status():
    client = _client_with_status("partially_filled")
    adapter = AlpacaExecutionAdapter(client)
    attempts = adapter.submit_delta_orders(
        {"AAA": 1000.0},
        min_order_notional=1.0,
        dry_run=False,
        correlation_id="cid",
    )
    out = adapter.observe_submitted_orders(attempts, max_polls=1, poll_interval_seconds=0.0)
    assert out[0].final_status == "partially_filled"


def test_integration_rejected_flow_surfaces_status():
    client = _client_with_status("rejected")
    adapter = AlpacaExecutionAdapter(client)
    attempts = adapter.submit_delta_orders(
        {"AAA": 1000.0},
        min_order_notional=1.0,
        dry_run=False,
        correlation_id="cid",
    )
    out = adapter.observe_submitted_orders(attempts, max_polls=1, poll_interval_seconds=0.0)
    assert all(a.final_status == "rejected" for a in out)
