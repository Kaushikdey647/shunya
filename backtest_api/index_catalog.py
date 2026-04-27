"""Yahoo-style **raw index** symbols per equity index code (matches ``equity_indexes`` migration codes)."""

from __future__ import annotations

from typing import Final

# Keys must match ``equity_indexes.code`` (see migrations 004/005).
# Values are standard index tickers (often with ``^`` prefix) used for benchmarks and OHLCV ingest.
RAW_INDEX_TICKER_BY_CODE: Final[dict[str, str]] = {
    "AEX": "^AEX",
    "BEL20": "^BFX",
    "CAC40": "^FCHI",
    "CACMID60": "^CACMD",
    "DAX": "^GDAXI",
    "DOWJONES": "^DJI",
    "EUROSTOXX50": "^STOXX50E",
    "FTSE100": "^FTSE",
    "IBEX35": "^IBEX",
    "MDAX": "^MDAXI",
    "NASDAQ100": "^NDX",
    "NIKKEI225": "^N225",
    "OMXHEL25": "^OMXH25",
    "OMXSTO30": "^OMX",
    "SDAX": "^SDAXI",
    "SP100": "^OEX",
    "SP500": "^GSPC",
    "SP600": "^SP600",
    "SWISS20": "^SSMI",
    "TECDAX": "^TECDAX",
}


def benchmark_for_index(index_code: str) -> str:
    """Return the raw index ticker (e.g. ``^GSPC``) used as the strategy benchmark series."""
    key = index_code.strip().upper()
    try:
        return RAW_INDEX_TICKER_BY_CODE[key]
    except KeyError as exc:
        raise KeyError(f"Unknown index_code for raw index mapping: {index_code!r}") from exc


def known_index_codes() -> frozenset[str]:
    return frozenset(RAW_INDEX_TICKER_BY_CODE.keys())
