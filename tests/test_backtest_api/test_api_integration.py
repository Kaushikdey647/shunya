"""API + DB integration: requires Postgres (see tests/test_backtest_api/conftest.py).

The backtest job test uses ``api_database_url_queue_isolated`` so no second process
(e.g. uvicorn) shares the same job queue on ``DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backtest_api.repositories import alphas as alphas_repo
from backtest_api.repositories import backtests as jobs_repo
from backtest_api.settings import get_settings

pytestmark = pytest.mark.timescale

_log = logging.getLogger(__name__)


def _fake_backtest_result(
    _payload: dict[str, Any],
    _imp: str | None,
    _source: str | None,
    _finstrat: dict[str, Any],
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


async def _integration_worker_loop(stop: asyncio.Event) -> None:
    """Same control flow as ``backtest_worker_loop`` but writes fake results (no FinTs/yfinance)."""
    settings = get_settings()
    interval = max(0.2, float(settings.worker_poll_interval_seconds))
    while not stop.is_set():
        row = await asyncio.to_thread(jobs_repo.claim_next_queued_job)
        if row is None:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass
            continue
        job_id, payload = row
        try:
            alpha_id = str(payload["alpha_id"])
            ar = await asyncio.to_thread(alphas_repo.get_alpha_raw, alpha_id)
            if ar is None:
                await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, "Alpha not found.")
                continue
            ir = ar.get("import_ref")
            sc = ar.get("source_code")
            serialized, summary = _fake_backtest_result(
                payload,
                ir if ir is not None else None,
                sc if sc is not None else None,
                dict(ar["finstrat_config"]),
            )
            await asyncio.to_thread(jobs_repo.mark_job_succeeded, job_id, serialized, summary)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            _log.exception("backtest job %s failed", job_id)
            msg = f"{exc!s}\n{tb}"[:8000]
            await asyncio.to_thread(jobs_repo.mark_job_failed, job_id, msg)


def test_alphas_crud_and_backtest_job(
    api_database_url_queue_isolated: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", api_database_url_queue_isolated)
    from shunya.data.timescale.dbutil import apply_migrations

    apply_migrations()

    import psycopg

    with psycopg.connect(api_database_url_queue_isolated) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM api_backtest_jobs WHERE status IN ('queued', 'running')"
            )
        conn.commit()

    monkeypatch.setattr(
        "backtest_api.main.backtest_worker_loop",
        _integration_worker_loop,
    )

    import backtest_api.main as _main_mod

    assert _main_mod.backtest_worker_loop is _integration_worker_loop

    from backtest_api.main import create_app

    alpha_name = f"api-test-alpha-{uuid.uuid4().hex[:12]}"
    with TestClient(create_app()) as client:
        r = client.post(
            "/alphas",
            json={
                "name": alpha_name,
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
        assert r2.json().get("include_test_period_in_results") is False

        deadline = time.time() + 10.0
        status = ""
        while time.time() < deadline:
            g = client.get(f"/backtests/{job_id}")
            assert g.status_code == 200
            assert g.json().get("include_test_period_in_results") is False
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

        with psycopg.connect(api_database_url_queue_isolated) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM api_backtest_jobs WHERE id = %s::uuid", (job_id,))
            conn.commit()

        dl2 = client.delete(f"/alphas/{alpha_id}")
        assert dl2.status_code == 204
