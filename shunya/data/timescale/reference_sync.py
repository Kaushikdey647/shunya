"""SEC reference data and the Alpaca tradability flag (``shunya-timescale sync-reference``).

One-time job, re-runnable for new tickers: ``company_tickers.json`` gives ticker and CIK, the
per-CIK ``submissions`` JSON gives SIC / owner org (``filings`` ignored), and Alpaca
``/v2/assets`` sets ``symbols.is_alpaca_tradable``. Backtests and order cycles read the result
from the DB only (:func:`tradable_symbols`, view ``symbol_reference``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
# SEC fair-access limit is 10 requests/second.
SEC_MIN_INTERVAL_S = 0.11


@dataclass(frozen=True)
class SecTicker:
    """One ``company_tickers.json`` entry, ticker normalized to Alpaca form."""

    ticker: str
    cik: str
    title: str


@dataclass(frozen=True)
class SecCompany:
    """Profile fields from a ``submissions`` document (``filings`` ignored)."""

    cik: str
    name: Optional[str]
    entity_type: Optional[str]
    sic: Optional[str]
    sic_description: Optional[str]
    owner_org: Optional[str]
    category: Optional[str]
    fiscal_year_end: Optional[str]
    state_of_incorporation: Optional[str]
    ein: Optional[str]
    exchanges: List[str]


def pad_cik(cik: Any) -> str:
    """``1045810`` → ``"0001045810"``."""
    return f"{int(cik):010d}"


def normalize_sec_ticker(ticker: str) -> str:
    """SEC class-share separator ``-`` → Alpaca ``.`` (``BRK-B`` → ``BRK.B``)."""
    return str(ticker).strip().upper().replace("-", ".")


def _blank_to_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def parse_company_tickers(doc: Mapping[str, Any]) -> List[SecTicker]:
    """Parse ``{"0": {"cik_str": .., "ticker": .., "title": ..}, ...}``; first entry wins per ticker."""
    out: dict[str, SecTicker] = {}
    for row in doc.values():
        t = normalize_sec_ticker(row["ticker"])
        if t and t not in out:
            out[t] = SecTicker(ticker=t, cik=pad_cik(row["cik_str"]), title=str(row.get("title") or "").strip())
    return list(out.values())


def parse_submissions(doc: Mapping[str, Any]) -> SecCompany:
    """Profile and classification from one ``CIK##########.json`` document."""
    return SecCompany(
        cik=pad_cik(doc["cik"]),
        name=_blank_to_none(doc.get("name")),
        entity_type=_blank_to_none(doc.get("entityType")),
        sic=_blank_to_none(doc.get("sic")),
        sic_description=_blank_to_none(doc.get("sicDescription")),
        owner_org=_blank_to_none(doc.get("ownerOrg")),
        category=_blank_to_none(doc.get("category")),
        fiscal_year_end=_blank_to_none(doc.get("fiscalYearEnd")),
        state_of_incorporation=_blank_to_none(doc.get("stateOfIncorporation")),
        ein=_blank_to_none(doc.get("ein")),
        exchanges=[str(x) for x in (doc.get("exchanges") or []) if x],
    )


def sec_user_agent() -> str:
    """``SHUNYA_SEC_USER_AGENT`` (SEC rejects requests without a contact User-Agent)."""
    ua = (os.environ.get("SHUNYA_SEC_USER_AGENT") or "").strip()
    if not ua:
        raise RuntimeError('set SHUNYA_SEC_USER_AGENT, e.g. "Shunya research you@example.com" (required by SEC)')
    return ua


class SecClient:
    """Throttled JSON GETs against sec.gov."""

    def __init__(self, user_agent: Optional[str] = None, *, min_interval_s: float = SEC_MIN_INTERVAL_S) -> None:
        self._ua = user_agent or sec_user_agent()
        self._min_interval = float(min_interval_s)
        self._last = 0.0

    def get_json(self, url: str) -> Any:
        """GET ``url`` and decode JSON, keeping under SEC's request rate."""
        wait = self._last + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": self._ua, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        finally:
            self._last = time.monotonic()

    def company_tickers(self) -> List[SecTicker]:
        """Download and parse ``company_tickers.json``."""
        return parse_company_tickers(self.get_json(COMPANY_TICKERS_URL))

    def submissions(self, cik: str) -> SecCompany:
        """Download and parse one company's ``submissions`` document."""
        return parse_submissions(self.get_json(SUBMISSIONS_URL.format(cik=pad_cik(cik))))


def alpaca_tradable_assets(trading_client: Any) -> dict[str, str]:
    """``symbol -> asset id`` for active, tradable US equities."""
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    assets = trading_client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    return {str(a.symbol).upper(): str(a.id) for a in assets if getattr(a, "tradable", False)}


# --- DB writes -------------------------------------------------------------------------------


def upsert_sec_tickers(cur: Any, rows: Sequence[SecTicker]) -> int:
    """Upsert companies (CIK + title) and symbols; mark symbols absent from the file as delisted."""
    cur.executemany(
        "INSERT INTO sec_companies (cik, name) VALUES (%s, %s) ON CONFLICT (cik) DO NOTHING",
        [(r.cik, r.title or None) for r in rows],
    )
    cur.executemany(
        """
        INSERT INTO symbols (ticker, name, cik, in_sec_tickers) VALUES (%s, %s, %s, true)
        ON CONFLICT (ticker) DO UPDATE SET
            cik = EXCLUDED.cik,
            in_sec_tickers = true,
            name = COALESCE(symbols.name, EXCLUDED.name)
        """,
        [(r.ticker, r.title or None, r.cik) for r in rows],
    )
    # Kept (bars stay joinable), only flagged.
    cur.execute(
        "UPDATE symbols SET in_sec_tickers = false WHERE in_sec_tickers AND NOT (ticker = ANY(%s))",
        ([r.ticker for r in rows],),
    )
    return len(rows)


def ciks_needing_submissions(cur: Any, *, refresh: bool, limit: Optional[int] = None) -> List[str]:
    """CIKs never fetched (or all with ``refresh``), tradable symbols' CIKs first."""
    sql = """
        SELECT c.cik
        FROM sec_companies c
        LEFT JOIN symbols s ON s.cik = c.cik AND s.is_alpaca_tradable
        WHERE %s OR c.fetched_at IS NULL
        GROUP BY c.cik
        ORDER BY bool_or(s.id IS NOT NULL) DESC, c.cik
    """
    params: list[Any] = [bool(refresh)]
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    cur.execute(sql, params)
    return [str(r[0]) for r in cur.fetchall()]


def upsert_sec_company(cur: Any, c: SecCompany) -> None:
    """Write ``sic_codes`` (when present) and the ``sec_companies`` profile row."""
    if c.sic:
        cur.execute(
            """
            INSERT INTO sic_codes (sic, description, owner_org) VALUES (%s, %s, %s)
            ON CONFLICT (sic) DO UPDATE SET
                description = COALESCE(EXCLUDED.description, sic_codes.description),
                owner_org = COALESCE(EXCLUDED.owner_org, sic_codes.owner_org)
            """,
            (c.sic, c.sic_description, c.owner_org),
        )
    cur.execute(
        """
        INSERT INTO sec_companies (
            cik, name, entity_type, sic, category, fiscal_year_end, state_of_incorporation, ein, exchanges, fetched_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (cik) DO UPDATE SET
            name = COALESCE(EXCLUDED.name, sec_companies.name),
            entity_type = EXCLUDED.entity_type,
            sic = EXCLUDED.sic,
            category = EXCLUDED.category,
            fiscal_year_end = EXCLUDED.fiscal_year_end,
            state_of_incorporation = EXCLUDED.state_of_incorporation,
            ein = EXCLUDED.ein,
            exchanges = EXCLUDED.exchanges,
            fetched_at = now()
        """,
        (
            c.cik,
            c.name,
            c.entity_type,
            c.sic,
            c.category,
            c.fiscal_year_end,
            c.state_of_incorporation,
            c.ein,
            c.exchanges,
        ),
    )


def apply_alpaca_tradable(cur: Any, assets: Mapping[str, str]) -> int:
    """Set ``is_alpaca_tradable`` on every ``symbols`` row from the Alpaca asset map; return tradable count."""
    syms = list(assets.keys())
    ids = [assets[s] for s in syms]
    cur.execute(
        """
        UPDATE symbols s SET
            is_alpaca_tradable = (m.asset_id IS NOT NULL),
            alpaca_asset_id = m.asset_id,
            alpaca_checked_at = now()
        FROM (
            SELECT s2.id, a.asset_id
            FROM symbols s2
            LEFT JOIN unnest(%s::text[], %s::text[]) AS a(symbol, asset_id) ON a.symbol = s2.ticker
        ) m
        WHERE s.id = m.id
        """,
        (syms, ids),
    )
    cur.execute("SELECT count(*) FROM symbols WHERE is_alpaca_tradable")
    return int(cur.fetchone()[0])


# --- DB reads (the only path backtests / order cycles use) ------------------------------------


def tradable_symbols(cur: Any, tickers: Optional[Iterable[str]] = None) -> List[str]:
    """Sorted tickers flagged ``is_alpaca_tradable`` (optionally restricted to ``tickers``)."""
    if tickers is None:
        cur.execute("SELECT ticker FROM symbols WHERE is_alpaca_tradable ORDER BY ticker")
    else:
        cur.execute(
            "SELECT ticker FROM symbols WHERE is_alpaca_tradable AND ticker = ANY(%s) ORDER BY ticker",
            (sorted({str(t) for t in tickers}),),
        )
    return [str(r[0]) for r in cur.fetchall()]


def sync_reference(
    conn: Any,
    *,
    sec: Optional[SecClient] = None,
    trading_client: Any = None,
    skip_submissions: bool = False,
    skip_alpaca: bool = False,
    refresh: bool = False,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Run the three passes: SEC tickers → Alpaca flag → SEC submissions (missing CIKs only).

    The Alpaca pass runs before submissions so tradable companies are fetched first.
    Each submissions row commits on its own, so an interrupted run resumes where it stopped.
    """
    sec = sec or SecClient()
    stats: dict[str, int] = {}
    with conn.cursor() as cur:
        stats["sec_tickers"] = upsert_sec_tickers(cur, sec.company_tickers())
    conn.commit()
    log(f"sec tickers: {stats['sec_tickers']}")

    if not skip_alpaca:
        if trading_client is None:
            from shunya.integration.alpaca_settings import build_trading_client, load_alpaca_settings_from_env

            trading_client = build_trading_client(load_alpaca_settings_from_env())
        with conn.cursor() as cur:
            stats["alpaca_tradable"] = apply_alpaca_tradable(cur, alpaca_tradable_assets(trading_client))
        conn.commit()
        log(f"alpaca tradable: {stats['alpaca_tradable']}")

    stats["submissions"] = 0
    stats["submissions_failed"] = 0
    if not skip_submissions:
        with conn.cursor() as cur:
            ciks = ciks_needing_submissions(cur, refresh=refresh, limit=limit)
        log(f"submissions to fetch: {len(ciks)}")
        for i, cik in enumerate(ciks, 1):
            try:
                company = sec.submissions(cik)
            except Exception as exc:  # one bad CIK must not stop a 10k-company run
                stats["submissions_failed"] += 1
                log(f"submissions {cik} failed: {exc}")
                continue
            with conn.cursor() as cur:
                upsert_sec_company(cur, company)
            conn.commit()
            stats["submissions"] += 1
            if i % 500 == 0:
                log(f"submissions: {i}/{len(ciks)}")
    return stats
