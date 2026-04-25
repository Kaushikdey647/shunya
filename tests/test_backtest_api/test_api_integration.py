"""API + DB integration: requires Postgres (see tests/test_backtest_api/conftest.py)."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.timescale


def _fake_run(
    _payload: dict[str, Any],
    _imp: str,
    _cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ser = {
        "metrics": {
            "total_return_pct": 1.23,
            "sharpe_ratio": 0.5,
            "max_drawdown_pct": 2.0,
            "end_value": 101230.0,
            "bar_unit": "DAYS",
            "bar_step": 1,
        },
        "equity_curve": [],
        "turnover_history": [],
        "returns_analysis": None,
        "drawdown_analysis": None,
        "sharpe_analysis": None,
        "target_history": [],
    }
    summary = {
        "total_return_pct": 1.23,
        "sharpe_ratio": 0.5,
        "max_drawdown_pct": 2.0,
        "end_value": 101230.0,
        "bar_unit": "DAYS",
        "bar_step": 1,
    }
    return ser, summary


def test_alphas_crud_and_backtest_job(
    api_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", api_database_url)
    from shunya.data.timescale.dbutil import apply_migrations

    apply_migrations()

    monkeypatch.setattr("backtest_api.worker.run_backtest_from_payload", _fake_run)

    from backtest_api.main import create_app

    with TestClient(create_app()) as client:
        r = client.post(
            "/alphas",
            json={
                "name": "api-test-alpha",
                "import_ref": "examples.alphas.sma_ratio_20:alpha",
                "finstrat_config": {"neutralization": "market"},
            },
        )
        assert r.status_code == 201, r.text
        alpha_id = r.json()["id"]

        r2 = client.post(
            "/backtests",
            json={
                "alpha_id": alpha_id,
                "fin_ts": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-02-01",
                    "ticker_list": ["AAPL"],
                    "market_data_provider": "yfinance",
                },
                "finbt": {"cash": 100000.0},
            },
        )
        assert r2.status_code == 201, r2.text
        job_id = r2.json()["id"]

        deadline = time.time() + 10.0
        status = ""
        while time.time() < deadline:
            g = client.get(f"/backtests/{job_id}")
            assert g.status_code == 200
            status = g.json()["status"]
            if status == "succeeded":
                break
            if status == "failed":
                pytest.fail(g.json().get("error_message", g.text))
            time.sleep(0.05)
        assert status == "succeeded"

        res = client.get(f"/backtests/{job_id}/result")
        assert res.status_code == 200
        body = res.json()
        assert body["job_id"] == job_id
        assert body["metrics"]["total_return_pct"] == 1.23

        dl = client.delete(f"/alphas/{alpha_id}")
        assert dl.status_code == 409

        # Remove job so alpha can be deleted (no DELETE /backtests in v1).
        import psycopg

        with psycopg.connect(api_database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM api_backtest_jobs WHERE id = %s::uuid", (job_id,))
            conn.commit()

        dl2 = client.delete(f"/alphas/{alpha_id}")
        assert dl2.status_code == 204
