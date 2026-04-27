"""PyTickerSymbols → ``symbol_index_membership`` (and ``symbols`` names) without OHLCV."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from shunya.data.timescale.ingest_lib import ensure_symbols

_LOG = logging.getLogger(__name__)

# ``display_name`` must match ``pytickersymbols.PyTickerSymbols.get_stocks_by_index`` exactly.
INDEX_SOURCE_TO_CODE = (
    ("AEX", "AEX"),
    ("BEL 20", "BEL20"),
    ("CAC_40", "CAC40"),
    ("CAC Mid 60", "CACMID60"),
    ("DAX", "DAX"),
    ("DOW JONES", "DOWJONES"),
    ("EURO STOXX 50", "EUROSTOXX50"),
    ("FTSE 100", "FTSE100"),
    ("IBEX 35", "IBEX35"),
    ("MDAX", "MDAX"),
    ("NASDAQ 100", "NASDAQ100"),
    ("NIKKEI 225", "NIKKEI225"),
    ("OMX Helsinki 25", "OMXHEL25"),
    ("OMX Stockholm 30", "OMXSTO30"),
    ("SDAX", "SDAX"),
    ("S&P 100", "SP100"),
    ("S&P 500", "SP500"),
    ("S&P 600", "SP600"),
    ("Switzerland 20", "SWISS20"),
    ("TecDAX", "TECDAX"),
)


def yahoo_preferred_symbol(stock: dict[str, Any]) -> str | None:
    """Prefer USD Yahoo listing; else first Yahoo symbol in ``symbols[]``; else top-level ``symbol``."""
    raw_syms = stock.get("symbols")
    fallback: str | None = None
    if isinstance(raw_syms, list):
        for entry in raw_syms:
            if not isinstance(entry, dict):
                continue
            y = entry.get("yahoo")
            if not isinstance(y, str) or not y.strip():
                continue
            ys = y.strip()
            if str(entry.get("currency", "")).upper() == "USD":
                return ys
            if fallback is None:
                fallback = ys
        if fallback is not None:
            return fallback
    sym = stock.get("symbol")
    if isinstance(sym, str) and sym.strip():
        return sym.strip()
    return None


def load_py_ticker_index_union() -> tuple[list[str], dict[str, set[str]], dict[str, str]]:
    """
    Union of Yahoo tickers across configured indices, membership map
    (upper ticker -> set of index codes), and PyTickerSymbols ``name`` per Yahoo ticker.
    """
    from pytickersymbols import PyTickerSymbols

    pts = PyTickerSymbols()
    membership: dict[str, set[str]] = {}
    display_names: dict[str, str] = {}
    seen_upper: set[str] = set()
    out: list[str] = []
    for display_name, code in INDEX_SOURCE_TO_CODE:
        stocks = list(pts.get_stocks_by_index(display_name))
        for stock in stocks:
            st = stock if isinstance(stock, dict) else {}
            y = yahoo_preferred_symbol(st)
            if not y:
                continue
            u = y.upper()
            membership.setdefault(u, set()).add(code)
            nm = st.get("name")
            if isinstance(nm, str) and nm.strip():
                display_names[str(y)] = nm.strip()[:2048]
            if u not in seen_upper:
                seen_upper.add(u)
                out.append(y)
    out.sort(key=str.upper)
    return out, membership, display_names


def sync_symbol_index_memberships(
    psycopg_module: Any,
    dsn: str,
    tickers: Sequence[str],
    membership_sets: dict[str, set[str]],
    ticker_display_names: dict[str, str],
) -> None:
    """``ensure_symbols`` + replace ``symbol_index_membership`` rows for ``tickers``."""
    tickers_list = list(tickers)
    if not tickers_list:
        return
    with psycopg_module.connect(dsn) as conn:
        with conn.cursor() as cur:
            tmap = ensure_symbols(
                cur,
                tickers_list,
                display_names={t: ticker_display_names.get(str(t)) for t in tickers_list},
            )
            cur.execute(
                """
                DELETE FROM symbol_index_membership m
                USING symbols s
                WHERE m.symbol_id = s.id AND s.ticker = ANY(%s)
                """,
                (tickers_list,),
            )
            for t in tickers_list:
                sid = tmap[str(t)]
                for code in sorted(membership_sets.get(str(t).upper(), set())):
                    cur.execute(
                        """
                        INSERT INTO symbol_index_membership (symbol_id, index_code)
                        VALUES (%s, %s)
                        ON CONFLICT (symbol_id, index_code) DO NOTHING
                        """,
                        (sid, code),
                    )
        conn.commit()
    _LOG.info("Synced symbol_index_membership for %d tickers", len(tickers_list))
