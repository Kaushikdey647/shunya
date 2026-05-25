"""Schedule :func:`~api.services.notification_hub.publish_notification` from sync FastAPI routes."""

from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks

from api.services.notification_hub import publish_notification


def schedule_notification(background_tasks: BackgroundTasks, **kwargs: Any) -> None:
    """Run ``publish_notification`` after the response is sent (non-blocking for the client)."""
    background_tasks.add_task(publish_notification, **kwargs)
