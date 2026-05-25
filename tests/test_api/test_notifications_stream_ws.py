"""Notifications WebSocket: hello and in-process fan-out."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services.notification_hub import publish_notification


def test_notifications_stream_hello() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/notifications/stream") as ws:
        raw = ws.receive_json()
        assert raw["type"] == "hello"
        assert raw["schema"] == 1


def test_notifications_stream_receives_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    async def quiet_worker(stop: asyncio.Event) -> None:
        await stop.wait()

    monkeypatch.setattr("api.main.backtest_worker_loop", quiet_worker)

    with TestClient(create_app()) as client:
        portal = client.portal
        with client.websocket_connect("/notifications/stream") as ws:
            assert ws.receive_json()["type"] == "hello"

            async def _publish() -> None:
                await publish_notification(
                    level="info",
                    message="unit test ping",
                    code="test.ping",
                    context={"job_id": "00000000-0000-0000-0000-000000000001"},
                )

            portal.call(_publish)
            note = ws.receive_json()
            assert note["type"] == "notification"
            assert note["schema"] == 1
            assert note["level"] == "info"
            assert note["message"] == "unit test ping"
            assert note["code"] == "test.ping"
            assert note["context"]["job_id"] == "00000000-0000-0000-0000-000000000001"
            assert "id" in note and "ts" in note
