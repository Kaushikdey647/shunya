"""Kenneth French Fama–French factor data (daily) with allow-listed HTTPS fetch."""

from __future__ import annotations

import io
import zipfile
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

# Daily 3 factors + RF; official file from Dartmouth.
_FF_DAILY_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

_cache: Optional[pd.DataFrame] = None


def _assert_allowed_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("https",):
        raise ValueError("ff_factors: only https URLs are allowed")
    host = (p.hostname or "").lower()
    if host != "mba.tuck.dartmouth.edu":
        raise ValueError("ff_factors: unexpected host")
    if not (p.path or "").startswith("/pages/faculty/ken.french/ftp/"):
        raise ValueError("ff_factors: unexpected path")


def _parse_ff_daily_csv(text: str) -> pd.DataFrame:
    """Parse F-F daily factors CSV (skip preamble until header row with Mkt-RF)."""
    lines = text.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "HML" in line and "RF" in line:
            start = i
            break
    if start >= len(lines):
        raise ValueError("ff_factors: could not find header in CSV")
    body = "\n".join(lines[start:])
    df = pd.read_csv(io.StringIO(body), engine="python")
    cols = [str(c).strip() if str(c).strip() else "Date" for c in df.columns]
    df.columns = cols
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df["Date"] = pd.to_datetime(df[date_col].astype(str).str.strip(), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in ("Mkt-RF", "SMB", "HML", "RF"):
        if c not in df.columns:
            raise ValueError(f"ff_factors: missing column {c!r}")
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df[["Mkt-RF", "SMB", "HML", "RF"]].dropna(how="any")


def fetch_ff_factors_daily(*, url: str = _FF_DAILY_ZIP_URL) -> pd.DataFrame:
    """
    Download and parse daily Fama–French 3 factors + RF (returns as decimals, e.g. 0.01 = 1%).

    Cached in-process after first successful load.
    """
    global _cache
    if _cache is not None:
        return _cache.copy()

    _assert_allowed_url(url)
    req = Request(url, headers={"User-Agent": "shunya-finbt/1.0"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 — URL is fixed allow-list
        raw = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    inner = next((n for n in names if n.endswith(".CSV") or n.endswith(".csv")), names[0])
    text = zf.read(inner).decode("utf-8", errors="replace")
    _cache = _parse_ff_daily_csv(text)
    return _cache.copy()


def load_ff_factors_daily_from_zip_bytes(data: bytes) -> pd.DataFrame:
    """Parse factor CSV from zip bytes (tests / offline fixtures)."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = zf.namelist()
    inner = next((n for n in names if n.endswith(".CSV") or n.endswith(".csv")), names[0])
    text = zf.read(inner).decode("utf-8", errors="replace")
    return _parse_ff_daily_csv(text)


def clear_ff_factors_cache() -> None:
    global _cache
    _cache = None
