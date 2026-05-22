"""
Canonical HTTP session for Yahoo Finance requests in Shunya.

Use :func:`build_yfinance_session` anywhere yfinance accepts ``session=`` (e.g.
``yf.Search``, :class:`~.providers.YFinanceMarketDataProvider`) so TLS behavior
is consistent with :func:`~shunya.integration.tls_env.tls_certificate_verification_enabled`
(see ``SHUNYA_TLS_VERIFY``).

Returns a **new** session per call (avoid sharing one curl_cffi session across threads).
"""

from __future__ import annotations

import logging

from shunya.integration.tls_env import tls_certificate_verification_enabled

_log = logging.getLogger(__name__)


def build_yfinance_session() -> object | None:
    """
    When :func:`~shunya.integration.tls_env.tls_certificate_verification_enabled` is ``True``,
    returns ``None`` so yfinance uses its default TLS-verifying stack.

    When verification is disabled (``SHUNYA_TLS_VERIFY=0`` / ``false`` / ``no`` / ``off``),
    prefer a ``curl_cffi`` Chrome session with ``verify=False`` (corporate TLS inspection).

    If ``curl_cffi`` is not installed, logs a warning and returns ``None``.
    """
    if tls_certificate_verification_enabled():
        return None
    try:
        from curl_cffi import requests as curl_requests

        return curl_requests.Session(impersonate="chrome", verify=False)
    except ImportError:
        _log.warning("curl_cffi not installed; yfinance will use default HTTP (no TLS workaround)")
        return None
