"""
Example FundamentalDataProvider backed by yfinance financial statements.

Use with ``finTs(..., attach_fundamentals=True, fundamental_data=...)`` and the same
``curl_cffi`` session + ``verify=False`` pattern as the OEX benchmark notebook when
default TLS verification fails behind corporate proxies.

``Price_To_Earnings`` and ``Free_Cash_Flow_Yield`` are left as NaN here (statement-only
pull); use FinanceToolkit + FMP if you need those time series.

Rows use **as-of** reads per statement (latest column with period end ``<=`` the union
date), a per-date **carry** of the last fully-known quarter within the ticker (no future
filings), then a per-ticker **forward fill** on the long panel. Union dates that still
lack any core line item (revenue, net income, assets, or operating cash flow) are
**dropped** so the periodic index is not padded with lookahead from later reports.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from shunya.data.fundamentals import validate_fundamental_fields
from shunya.data.timeframes import BarSpec, default_bar_spec, normalize_bar_timestamp


def _make_thread_session(shared: Any | None) -> Any:
    """yfinance + curl_cffi: prefer a fresh session per thread (not shared)."""
    if shared is not None:
        return shared
    try:
        from curl_cffi import requests as curl_requests

        return curl_requests.Session(impersonate="chrome", verify=False)
    except Exception:
        return None


def _cell(df: pd.DataFrame | None, row_names: tuple[str, ...], col: object) -> float:
    if df is None or df.empty or col not in df.columns:
        return float("nan")
    for r in row_names:
        if r in df.index:
            v = df.loc[r, col]
            if pd.isna(v):
                return float("nan")
            return float(v)
    return float("nan")


def _cell_asof_first(
    df: pd.DataFrame | None,
    row_name_groups: tuple[tuple[str, ...], ...],
    asof_col: object,
) -> float:
    """First finite ``_cell_asof`` across alternative row-name groups (e.g. EPS variants)."""
    for rows in row_name_groups:
        v = _cell_asof(df, rows, asof_col)
        if np.isfinite(v):
            return v
    return float("nan")


def _cell_asof(df: pd.DataFrame | None, row_names: tuple[str, ...], asof_col: object) -> float:
    """
    Latest non-NaN statement value for ``row_names`` from columns with period end
    ``<= asof_col`` (staggered fiscal calendars / union-of-columns).
    """
    if df is None or df.empty:
        return float("nan")
    deadline = pd.Timestamp(asof_col).normalize()
    eligible = sorted(
        (c for c in df.columns if pd.Timestamp(c).normalize() <= deadline),
        key=lambda c: pd.Timestamp(c),
    )
    if not eligible:
        return float("nan")
    for col in reversed(eligible):
        v = _cell(df, row_names, col)
        if np.isfinite(v):
            return v
    return float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return float("nan")
    return float(num / den)


def _statement_periods(
    inc: pd.DataFrame | None,
    bs: pd.DataFrame | None,
    cf: pd.DataFrame | None,
) -> list[pd.Timestamp]:
    cols: set[pd.Timestamp] = set()
    for df in (inc, bs, cf):
        if df is not None and not df.empty:
            for c in df.columns:
                cols.add(pd.Timestamp(c).normalize())
    return sorted(cols)


def _fetch_one_ticker(
    ticker: str,
    *,
    quarterly: bool,
    bar_spec: BarSpec,
    session: Any | None,
    requested_cols: tuple[str, ...],
) -> list[dict[str, Any]]:
    freq = "quarterly" if quarterly else "yearly"
    t = yf.Ticker(ticker, session=session)
    inc = t.get_income_stmt(freq=freq)
    bs = t.get_balance_sheet(freq=freq)
    cf = t.get_cash_flow(freq=freq)

    periods = _statement_periods(inc, bs, cf)
    rows_out: list[dict[str, Any]] = []
    # Carry last known quarter across sparse union dates (no lookahead into future filings).
    _last_snap: dict[str, float] = {}

    for raw_col in periods:
        col = normalize_bar_timestamp(pd.Timestamp(raw_col), bar_spec)
        rev = _cell_asof(inc, ("TotalRevenue", "OperatingRevenue"), raw_col)
        cogs = _cell_asof(inc, ("CostOfRevenue", "ReconciledCostOfRevenue"), raw_col)
        gp = _cell_asof(inc, ("GrossProfit",), raw_col)
        op_inc = _cell_asof(inc, ("OperatingIncome", "TotalOperatingIncomeAsReported"), raw_col)
        ebit = _cell_asof(inc, ("EBIT",), raw_col)
        ebitda = _cell_asof(inc, ("EBITDA", "NormalizedEBITDA"), raw_col)
        pretax = _cell_asof(inc, ("PretaxIncome",), raw_col)
        net = _cell_asof_first(
            inc,
            (
                (
                    "NetIncome",
                    "NetIncomeCommonStockholders",
                    "NetIncomeContinuousOperations",
                ),
                ("NetIncomeFromContinuingOperationNetMinorityInterest",),
                ("NormalizedIncome",),
            ),
            raw_col,
        )
        eps_dil = _cell_asof_first(
            inc,
            (("DilutedEPS",), ("BasicEPS",), ("NormalizedEPS",)),
            raw_col,
        )

        ta = _cell_asof(bs, ("TotalAssets",), raw_col)
        te = _cell_asof(bs, ("StockholdersEquity", "TotalEquityGrossMinorityInterest"), raw_col)
        td = _cell_asof(bs, ("TotalDebt",), raw_col)
        ca = _cell_asof(bs, ("CurrentAssets",), raw_col)
        cl = _cell_asof(bs, ("CurrentLiabilities",), raw_col)

        ocf = _cell_asof_first(
            cf,
            (
                ("OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"),
                ("CashFlowFromOperations",),
            ),
            raw_col,
        )
        fcf = _cell_asof_first(cf, (("FreeCashFlow",), ("FreeCashFlowFromContinuingOperations",)), raw_col)

        gross_margin = _safe_ratio(gp, rev) if np.isfinite(gp) else _safe_ratio(rev - cogs, rev)

        op_margin = _safe_ratio(op_inc, rev)
        if not np.isfinite(op_margin):
            op_margin = _safe_ratio(ebit, rev)
        if not np.isfinite(op_margin):
            op_margin = _safe_ratio(ebitda, rev)
        if not np.isfinite(op_margin):
            op_margin = _safe_ratio(pretax, rev)
        roa = _safe_ratio(net, ta)
        roe = _safe_ratio(net, te)
        de = _safe_ratio(td, te)
        cur = _safe_ratio(ca, cl)

        rec: dict[str, Any] = {
            "Revenue": rev,
            "Net_Income": net,
            "EPS_Diluted": eps_dil,
            "Operating_Cash_Flow": ocf,
            "Free_Cash_Flow": fcf,
            "Total_Assets": ta,
            "Total_Equity": te,
            "Total_Debt": td,
            "Current_Ratio": cur,
            "Gross_Margin": gross_margin,
            "Operating_Margin": op_margin,
            "Return_On_Assets": roa,
            "Return_On_Equity": roe,
            "Debt_To_Equity": de,
            "Price_To_Earnings": float("nan"),
            "Free_Cash_Flow_Yield": float("nan"),
        }
        for _k in list(rec.keys()):
            if _k in ("Price_To_Earnings", "Free_Cash_Flow_Yield"):
                continue
            if not np.isfinite(rec[_k]) and _k in _last_snap:
                rec[_k] = _last_snap[_k]

        _core = ("Revenue", "Net_Income", "Total_Assets", "Operating_Cash_Flow")
        if not any(np.isfinite(rec[k]) for k in _core):
            continue

        for _k, _v in rec.items():
            if _k in ("Price_To_Earnings", "Free_Cash_Flow_Yield"):
                continue
            if np.isfinite(_v):
                _last_snap[_k] = float(_v)

        row: dict[str, Any] = {"Ticker": ticker, "Date": col}
        for c in requested_cols:
            row[c] = rec[c]
        rows_out.append(row)

    return rows_out


class YFinanceFundamentalDataProvider:
    """Period-end fundamentals from yfinance statements, mapped to ``FUNDAMENTAL_FIELDS`` names."""

    def __init__(
        self,
        session: Any | None = None,
        *,
        max_workers: int = 6,
        thread_local_session: bool = True,
        enable_fetch_cache: bool = True,
    ) -> None:
        """
        Args:
            session: Used only when ``thread_local_session`` is False (single session
                across workers; use ``max_workers=1`` if the client is not thread-safe).
            max_workers: Parallel ticker fetches (keep small for Yahoo rate limits).
            thread_local_session: When True (default), each worker uses a dedicated
                ``curl_cffi.Session(verify=False)`` for safe parallel fetches.
            enable_fetch_cache: When True (default), identical ``fetch`` calls reuse the
                last dataframe (e.g. notebook preview then ``finTs`` attach).
        """
        self._session = session
        self._max_workers = max(1, int(max_workers))
        self._thread_local_session = bool(thread_local_session)
        self._enable_fetch_cache = bool(enable_fetch_cache)
        self._fetch_cache: dict[tuple[Any, ...], pd.DataFrame] = {}

    def fetch(
        self,
        ticker_list: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        fields: Optional[Sequence[str]] = None,
        quarterly: bool = True,
        bar_spec: Optional[BarSpec] = None,
    ) -> pd.DataFrame:
        specs = validate_fundamental_fields(fields)
        requested = tuple(spec.column for spec in specs)
        use_spec = bar_spec if bar_spec is not None else default_bar_spec()
        cache_key = (
            frozenset(str(t) for t in ticker_list),
            pd.Timestamp(start).normalize(),
            pd.Timestamp(end).normalize(),
            requested,
            bool(quarterly),
            str(use_spec.unit),
            int(use_spec.step),
        )
        if self._enable_fetch_cache and cache_key in self._fetch_cache:
            return self._fetch_cache[cache_key].copy()

        if not ticker_list:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=list(requested), dtype=float)

        t_end = pd.Timestamp(end).normalize()

        def job(sym: str) -> list[dict[str, Any]]:
            sess = _make_thread_session(None) if self._thread_local_session else _make_thread_session(
                self._session
            )
            return _fetch_one_ticker(
                sym,
                quarterly=quarterly,
                bar_spec=use_spec,
                session=sess,
                requested_cols=requested,
            )

        chunks: list[list[dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futs = {ex.submit(job, str(s)): str(s) for s in ticker_list}
            for fut in as_completed(futs):
                chunks.append(fut.result())

        flat = [row for part in chunks for row in part]
        if not flat:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=list(requested), dtype=float)

        raw_df = pd.DataFrame(flat)
        raw_df = raw_df[raw_df["Date"] <= t_end]
        if raw_df.empty:
            idx = pd.MultiIndex.from_arrays(
                [pd.Index([], dtype=object), pd.DatetimeIndex([], name="Date")],
                names=["Ticker", "Date"],
            )
            return pd.DataFrame(index=idx, columns=list(requested), dtype=float)

        out = raw_df.set_index(["Ticker", "Date"]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out = out.loc[:, list(requested)]
        out = out.astype(float)
        # Carry last known quarter forward along the date axis (per ticker).
        out = out.groupby(level=0, sort=False).ffill()
        if self._enable_fetch_cache:
            self._fetch_cache[cache_key] = out.copy()
        return out
