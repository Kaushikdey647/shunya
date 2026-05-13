"""Central Alpaca-py client construction from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream


def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _parse_bool(raw: Optional[str], *, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class AlpacaRuntimeSettings:
    """
    Credentials and paper/live mode for Alpaca-py clients.

    Keys are read from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` first, then optional
    ``SHUNYA_ALPACA_API_KEY_ID`` / ``SHUNYA_ALPACA_API_SECRET_KEY`` overrides.

    Paper mode defaults to ``True`` (``SHUNYA_ALPACA_PAPER`` unset or truthy).
    """

    api_key_id: str
    secret_key: str
    paper: bool = True

    @staticmethod
    def from_env(*, default_paper: bool = True) -> "AlpacaRuntimeSettings":
        """Load settings from environment; raises ``ValueError`` if keys are missing."""
        key = _env_first("APCA_API_KEY_ID", "SHUNYA_ALPACA_API_KEY_ID")
        sec = _env_first("APCA_API_SECRET_KEY", "SHUNYA_ALPACA_API_SECRET_KEY")
        if not key or not sec:
            raise ValueError(
                "Alpaca API keys are required: set APCA_API_KEY_ID and APCA_API_SECRET_KEY "
                "(or SHUNYA_ALPACA_API_KEY_ID / SHUNYA_ALPACA_API_SECRET_KEY)."
            )
        paper = _parse_bool(os.environ.get("SHUNYA_ALPACA_PAPER"), default=default_paper)
        return AlpacaRuntimeSettings(api_key_id=key, secret_key=sec, paper=paper)


def load_alpaca_settings_from_env(*, default_paper: bool = True) -> AlpacaRuntimeSettings:
    """Same as :meth:`AlpacaRuntimeSettings.from_env`."""
    return AlpacaRuntimeSettings.from_env(default_paper=default_paper)


def try_load_alpaca_settings_from_env(*, default_paper: bool = True) -> Optional[AlpacaRuntimeSettings]:
    """Return settings when keys exist; otherwise ``None`` (for optional Alpaca in API startup)."""
    try:
        return AlpacaRuntimeSettings.from_env(default_paper=default_paper)
    except ValueError:
        return None


def build_trading_client(settings: AlpacaRuntimeSettings) -> "TradingClient":
    from alpaca.trading.client import TradingClient

    return TradingClient(settings.api_key_id, settings.secret_key, paper=settings.paper)


def build_stock_historical_data_client(settings: AlpacaRuntimeSettings) -> "StockHistoricalDataClient":
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(
        api_key=settings.api_key_id,
        secret_key=settings.secret_key,
        sandbox=settings.paper,
    )


def build_trading_stream(settings: AlpacaRuntimeSettings) -> "TradingStream":
    from alpaca.trading.stream import TradingStream

    return TradingStream(settings.api_key_id, settings.secret_key, paper=settings.paper)
