from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from shunya.data.providers import env_yfinance_repair_default

__all__ = ["Settings", "get_settings", "env_yfinance_repair_default"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHUNYA_API_", extra="ignore")

    database_url: str | None = None
    """If unset, falls back to DATABASE_URL / SHUNYA_DATABASE_URL via dbutil."""

    ollama_host: str | None = None
    """Base URL for Ollama (e.g. ``http://127.0.0.1:11434``). When unset, alpha assist returns no markers."""

    ollama_model: str = "llama3.2"
    """Model id for ``/api/chat``."""

    ollama_timeout_seconds: float = 120.0
    """HTTP timeout for Ollama assist requests."""

    worker_poll_interval_seconds: float = 1.0
    """How often the background worker polls for queued backtest jobs."""

    max_target_history_points: int = 500
    """Cap serialized target_history rows per job result."""

    max_group_exposure_history_points: int = 500
    """Cap serialized group_exposure_history rows per job result."""

    max_exposure_history_points: int = 500
    """Cap serialized exposure_history rows per job result."""

    max_trade_events: int = 2000
    """Cap serialized trade_events per job result."""

    index_ohlcv_backfill_batch_size: int = 40
    """Tickers per yfinance download when backfilling OHLCV for failed index backtests."""

    market_data_cache_ttl_days: int = 30
    """Ignore Timescale OHLCV manifest and instrument yfinance JSON cache older than this many days."""

    # Yahoo price repair uses env_yfinance_repair_default():
    # SHUNYA_YFINANCE_REPAIR or SHUNYA_API_YFINANCE_REPAIR — see shunya.data.providers.


@lru_cache
def get_settings() -> Settings:
    return Settings()
