"""
Canonical HTTP session for Yahoo Finance requests in Shunya.

Use :func:`build_yfinance_session` anywhere yfinance accepts ``session=`` (e.g.
``yf.Search``, :class:`~.providers.YFinanceMarketDataProvider`) so TLS behavior
is consistent—especially behind corporate TLS inspection where default
verification fails.

Returns a **new** session per call (avoid sharing one curl_cffi session across threads).
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)


def build_yfinance_session() -> object | None:
    """
    Prefer ``curl_cffi`` Chrome impersonation with ``verify=False``.

    If ``YFINANCE_TLS_VERIFY`` is set to a truthy value (``1``, ``true``, ``yes``, ``on``),
    returns ``None`` so yfinance uses its default TLS-verifying stack.

    If ``curl_cffi`` is not installed, logs a warning and returns ``None``.
    """
    v = os.environ.get("YFINANCE_TLS_VERIFY", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return None
    try:
        from curl_cffi import requests as curl_requests

        return curl_requests.Session(impersonate="chrome", verify=False)
    except ImportError:
        _log.warning("curl_cffi not installed; yfinance will use default HTTP (no TLS workaround)")
        return None
