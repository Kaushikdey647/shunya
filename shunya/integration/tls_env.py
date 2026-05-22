"""TLS certificate verification policy from ``SHUNYA_TLS_VERIFY`` (yfinance + Alpaca)."""

from __future__ import annotations

import logging
import os
import ssl
from typing import Any, Final

_LOG = logging.getLogger(__name__)

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_FALSY: Final = frozenset({"0", "false", "no", "off"})

_urllib3_warnings_disabled = False


def tls_certificate_verification_enabled() -> bool:
    """
    Whether outbound HTTPS/WSS should verify server certificates.

    - **Unset or empty:** ``True`` (verify; secure default).
    - **Falsy:** ``0``, ``false``, ``no``, ``off`` (case-insensitive) → ``False``.
    - **Truthy:** ``1``, ``true``, ``yes``, ``on`` → ``True``.
    - **Other values:** log a warning and default to ``True``.
    """
    raw = os.environ.get("SHUNYA_TLS_VERIFY")
    if raw is None or str(raw).strip() == "":
        return True
    v = str(raw).strip().lower()
    if v in _FALSY:
        return False
    if v in _TRUTHY:
        return True
    _LOG.warning(
        "SHUNYA_TLS_VERIFY=%r is not a recognized value; defaulting to TLS verification enabled",
        raw,
    )
    return True


def disable_insecure_request_warnings_once() -> None:
    """Suppress urllib3 ``InsecureRequestWarning`` when using ``requests`` with ``verify=False``."""
    global _urllib3_warnings_disabled
    if _urllib3_warnings_disabled:
        return
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    _urllib3_warnings_disabled = True


def insecure_ssl_context_for_websockets() -> ssl.SSLContext:
    """SSL context that does not verify certificates (use only with ``SHUNYA_TLS_VERIFY=0``)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def alpaca_trading_stream_websocket_params_relaxed() -> dict[str, Any]:
    """
    ``websocket_params`` for :class:`alpaca.trading.stream.TradingStream` when TLS verification is off.

    Must include the same keys Alpaca-py defaults to, because passing ``websocket_params``
    replaces the internal dict entirely.
    """
    return {
        "ping_interval": 10,
        "ping_timeout": 180,
        "max_queue": 1024,
        "ssl": insecure_ssl_context_for_websockets(),
    }
