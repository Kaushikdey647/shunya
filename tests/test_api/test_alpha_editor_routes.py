from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.alpha_assist import sanitize_metrics_blob
from api.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_lint_body_ok(client: TestClient) -> None:
    r = client.post("/alphas/lint-body", json={"source_body": "return cs.rank(ctx.close)"})
    assert r.status_code == 200
    data = r.json()
    assert "diagnostics" in data
    assert isinstance(data["diagnostics"], list)


def test_assist_body_no_ollama_returns_empty_markers(client: TestClient) -> None:
    r = client.post(
        "/alphas/assist-body",
        json={"source_body": "return cs.rank(ctx.close)", "alpha_name": "t"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("markers") == []
    assert data.get("issues") == []


def test_assist_backtest_review_no_ollama(client: TestClient) -> None:
    r = client.post(
        "/alphas/assist-backtest-review",
        json={
            "source_body": "return cs.rank(ctx.close)",
            "metrics": {"total_return_pct": 1.2},
            "result_summary": {"sharpe_ratio": 0.5},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "summary_markdown" in data
    assert "Ollama" in data["summary_markdown"]
    assert isinstance(data.get("summary_points"), list)
    assert len(data["summary_points"]) >= 1


def test_sanitize_metrics_blob_smoke() -> None:
    """Regression: depth must be passed as second positional arg to inner trim."""
    s = sanitize_metrics_blob({"total_return_pct": 1.2}, {"nested": {"x": [1, 2, 3]}})
    assert "metrics" in s
    assert "result_summary" in s
