"""SEC reference parsing and sync orchestration (no database, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from shunya.data.timescale import reference_sync as rs

NVDA_SUBMISSIONS = {
    "cik": "0001045810",
    "entityType": "operating",
    "sic": "3674",
    "sicDescription": "Semiconductors & Related Devices",
    "ownerOrg": "04 Manufacturing",
    "name": "NVIDIA CORP",
    "tickers": ["NVDA"],
    "exchanges": ["Nasdaq"],
    "ein": "943177549",
    "lei": None,
    "description": "",
    "category": "Large accelerated filer",
    "fiscalYearEnd": "0131",
    "stateOfIncorporation": "DE",
    "filings": {"recent": {"accessionNumber": ["0001197647-26-000013"]}},
}


def test_pad_cik_and_normalize_ticker() -> None:
    assert rs.pad_cik(1045810) == "0001045810"
    assert rs.pad_cik("320193") == "0000320193"
    assert rs.normalize_sec_ticker("brk-b") == "BRK.B"
    assert rs.normalize_sec_ticker(" NVDA ") == "NVDA"


def test_parse_company_tickers_keeps_multi_ticker_ciks_and_dedupes() -> None:
    doc = {
        "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
        "1": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
        "2": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
        "3": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
        "4": {"cik_str": 9999999, "ticker": "BRK.B", "title": "duplicate after normalization"},
    }
    rows = {r.ticker: r for r in rs.parse_company_tickers(doc)}
    assert set(rows) == {"NVDA", "GOOGL", "GOOG", "BRK.B"}
    assert rows["GOOG"].cik == rows["GOOGL"].cik == "0001652044"
    assert rows["BRK.B"].cik == "0001067983"
    assert rows["NVDA"].title == "NVIDIA CORP"


def test_parse_submissions_ignores_filings_and_blanks() -> None:
    c = rs.parse_submissions(NVDA_SUBMISSIONS)
    assert c.cik == "0001045810"
    assert (c.sic, c.sic_description, c.owner_org) == ("3674", "Semiconductors & Related Devices", "04 Manufacturing")
    assert c.exchanges == ["Nasdaq"]
    assert c.fiscal_year_end == "0131"
    assert not hasattr(c, "filings")


def test_parse_submissions_without_sic() -> None:
    c = rs.parse_submissions({"cik": "0000000001", "sic": "", "sicDescription": None, "exchanges": None})
    assert c.sic is None and c.sic_description is None and c.exchanges == []


def test_sec_user_agent_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHUNYA_SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="SHUNYA_SEC_USER_AGENT"):
        rs.SecClient()


def test_sec_client_sends_user_agent_and_pads_cik(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    class _Resp:
        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def read(self, *a: Any) -> bytes:
            return b'{"cik": "0001045810", "sic": "3674"}'

    def fake_urlopen(req: Any, timeout: float) -> _Resp:
        seen.append(req)
        return _Resp()

    monkeypatch.setattr(rs.urllib.request, "urlopen", fake_urlopen)
    c = rs.SecClient("Test agent test@example.com", min_interval_s=0).submissions("1045810")
    assert c.sic == "3674"
    assert seen[0].full_url == "https://data.sec.gov/submissions/CIK0001045810.json"
    assert seen[0].get_header("User-agent") == "Test agent test@example.com"


def test_alpaca_tradable_assets_filters_non_tradable() -> None:
    assets = [
        SimpleNamespace(symbol="NVDA", id="a1", tradable=True),
        SimpleNamespace(symbol="BRK.B", id="a2", tradable=True),
        SimpleNamespace(symbol="XYZ", id="a3", tradable=False),
    ]
    client = SimpleNamespace(get_all_assets=lambda req: assets)
    assert rs.alpaca_tradable_assets(client) == {"NVDA": "a1", "BRK.B": "a2"}


class _Cur:
    def __init__(self, log: list[str], ciks: list[str]) -> None:
        self.log, self.ciks, self._last = log, ciks, ""

    def __enter__(self) -> "_Cur":
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._last = " ".join(sql.split())
        self.log.append(self._last.split(" ")[0] + ":" + ("sec_companies" if "sec_companies" in sql else "symbols"))

    def executemany(self, sql: str, rows: Any) -> None:
        self.execute(sql)

    def fetchall(self) -> list[tuple]:
        return [(c,) for c in self.ciks]

    def fetchone(self) -> tuple:
        return (1,)


class _Conn:
    def __init__(self, ciks: list[str]) -> None:
        self.log: list[str] = []
        self.ciks = ciks
        self.commits = 0

    def cursor(self) -> _Cur:
        return _Cur(self.log, self.ciks)

    def commit(self) -> None:
        self.commits += 1


class _Sec:
    def __init__(self, fail: set[str]) -> None:
        self.fail = fail
        self.fetched: list[str] = []

    def company_tickers(self) -> list[rs.SecTicker]:
        return [rs.SecTicker("NVDA", "0001045810", "NVIDIA CORP")]

    def submissions(self, cik: str) -> rs.SecCompany:
        self.fetched.append(cik)
        if cik in self.fail:
            raise OSError("boom")
        return rs.parse_submissions({**NVDA_SUBMISSIONS, "cik": cik})


def test_sync_reference_order_and_failure_isolation() -> None:
    conn = _Conn(["0001045810", "0000000002"])
    sec = _Sec(fail={"0000000002"})
    client = SimpleNamespace(get_all_assets=lambda req: [SimpleNamespace(symbol="NVDA", id="a1", tradable=True)])
    stats = rs.sync_reference(conn, sec=sec, trading_client=client, log=lambda m: None)
    assert stats == {"sec_tickers": 1, "alpaca_tradable": 1, "submissions": 1, "submissions_failed": 1}
    assert sec.fetched == ["0001045810", "0000000002"]
    # tickers pass, then the Alpaca flag, then per-CIK submissions
    first_update = conn.log.index("UPDATE:symbols")
    assert conn.log.index("INSERT:sec_companies") < first_update
    assert conn.commits == 3  # tickers, alpaca, one successful submission


def test_sync_reference_skips() -> None:
    conn = _Conn([])
    sec = _Sec(fail=set())
    stats = rs.sync_reference(conn, sec=sec, skip_alpaca=True, skip_submissions=True, log=lambda m: None)
    assert stats == {"sec_tickers": 1, "submissions": 0, "submissions_failed": 0}
    assert sec.fetched == []
