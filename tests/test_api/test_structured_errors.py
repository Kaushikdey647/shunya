"""Structured error responses from the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app


def test_shunya_error_returns_structured_detail() -> None:
    client = TestClient(create_app())
    r = client.get("/data/dashboard", params={"interval": "bogus"})
    assert r.status_code == 400
    body = r.json()
    assert "detail" in body
    d = body["detail"]
    assert isinstance(d, dict)
    assert d.get("code") == "DATA_INVALID_INTERVAL"
    assert "message" in d
