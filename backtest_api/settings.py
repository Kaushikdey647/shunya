from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHUNYA_API_", extra="ignore")

    database_url: str | None = None
    """If unset, falls back to DATABASE_URL / SHUNYA_DATABASE_URL via dbutil."""

    worker_poll_interval_seconds: float = 1.0
    """How often the background worker polls for queued backtest jobs."""

    max_target_history_points: int = 500
    """Cap serialized target_history rows per job result."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
