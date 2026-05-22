"""Central Alpaca-py client construction from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from shunya.integration.tls_env import (
    alpaca_trading_stream_websocket_params_relaxed,
    disable_insecure_request_warnings_once,
    tls_certificate_verification_enabled,
)

_LOG = logging.getLogger(__name__)

# Log once per process when Alpaca clients skip TLS verification (SHUNYA_TLS_VERIFY off).
_alpaca_insecure_tls_logged = False


def _warn_alpaca_insecure_tls_once() -> None:
    global _alpaca_insecure_tls_logged
    if _alpaca_insecure_tls_logged:
        return
    _LOG.warning(
        "Alpaca REST/WebSocket TLS certificate verification is disabled (SHUNYA_TLS_VERIFY). "
        "API secrets are vulnerable to MITM; use only in controlled dev environments."
    )
    _alpaca_insecure_tls_logged = True

if TYPE_CHECKING:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live.stock import StockDataStream
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


def _relax_alpaca_rest_tls_verify(client: object) -> None:
    """
    When ``SHUNYA_TLS_VERIFY`` disables verification, patch Alpaca-py's private ``requests.Session``.

    Relies on ``RESTClient._session`` (alpaca-py). Prefer fixing CA trust over using this path.
    """
    if tls_certificate_verification_enabled():
        return
    _warn_alpaca_insecure_tls_once()
    disable_insecure_request_warnings_once()
    session = getattr(client, "_session", None)
    if session is not None:
        setattr(session, "verify", False)


def build_trading_client(settings: AlpacaRuntimeSettings) -> "TradingClient":
    from alpaca.trading.client import TradingClient

    client = TradingClient(settings.api_key_id, settings.secret_key, paper=settings.paper)
    _relax_alpaca_rest_tls_verify(client)
    return client


def build_stock_historical_data_client(settings: AlpacaRuntimeSettings) -> "StockHistoricalDataClient":
    from alpaca.data.historical import StockHistoricalDataClient

    client = StockHistoricalDataClient(
        api_key=settings.api_key_id,
        secret_key=settings.secret_key,
        sandbox=settings.paper,
    )
    _relax_alpaca_rest_tls_verify(client)
    return client


def build_trading_stream(settings: AlpacaRuntimeSettings) -> "TradingStream":
    from alpaca.trading.stream import TradingStream

    ws_params = None
    if not tls_certificate_verification_enabled():
        _warn_alpaca_insecure_tls_once()
        ws_params = alpaca_trading_stream_websocket_params_relaxed()
    return TradingStream(
        settings.api_key_id,
        settings.secret_key,
        paper=settings.paper,
        websocket_params=ws_params,
    )


def build_stock_data_stream(
    settings: AlpacaRuntimeSettings,
    *,
    feed: "DataFeed | None" = None,
) -> "StockDataStream":
    """
    Live stock market WebSocket client (quotes, trades, bars, etc.).

    ``feed`` defaults to :func:`shunya.data.providers.default_alpaca_data_feed_from_env`
    (``SHUNYA_ALPACA_BAR_FEED``). Pass an explicit :class:`~alpaca.data.enums.DataFeed`
    (e.g. ``DataFeed.IEX``) to pin the stream. Run with :meth:`StockDataStream._run_forever`
    on the app's asyncio loop (do not call :meth:`StockDataStream.run`, which uses ``asyncio.run``).
    """
    from alpaca.data.live.stock import StockDataStream

    from shunya.data.providers import default_alpaca_data_feed_from_env

    chosen = feed if feed is not None else default_alpaca_data_feed_from_env()
    ws_params = None
    if not tls_certificate_verification_enabled():
        _warn_alpaca_insecure_tls_once()
        ws_params = alpaca_trading_stream_websocket_params_relaxed()
    return StockDataStream(
        settings.api_key_id,
        settings.secret_key,
        feed=chosen,
        websocket_params=ws_params,
    )
