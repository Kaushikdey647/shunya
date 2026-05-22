"""yfinance-backed instrument dashboard payloads (overview, statements, holders, options)."""

# TODO(market-data-router): Dashboard yfinance calls should not bypass shared adapter / future fundamentals registry.

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from fastapi import HTTPException

from shunya.integration.yahoo_public import YahooPublicAdapter

from shunya.data.timescale.market_cache_lib import (
    DOC_FINANCIALS_BALANCE,
    DOC_FINANCIALS_CASHFLOW,
    DOC_FINANCIALS_INCOME,
    DOC_HOLDERS,
    DOC_OPTION_CHAIN,
    DOC_OPTION_EXPIRATIONS,
    DOC_OPTION_IV_SURFACE,
    DOC_OVERVIEW,
    DOC_VALUATION_MEASURES,
)

from api.schemas.models import (
    InstrumentCompanyProfile,
    InstrumentExecutive,
    InstrumentFeatureAvailability,
    InstrumentFinancialFrequencyLiteral,
    InstrumentFinancialLineRow,
    InstrumentFinancialStatementResponse,
    InstrumentFundSummary,
    InstrumentFundTopHolding,
    InstrumentHolderRow,
    InstrumentHoldersResponse,
    InstrumentKindLiteral,
    InstrumentIvHeatmapResponse,
    InstrumentOptionChainResponse,
    InstrumentOptionContractSummary,
    InstrumentOptionExpirationsResponse,
    InstrumentOptionLegRow,
    InstrumentOverviewResponse,
    InstrumentStatementLiteral,
    InstrumentValuationMeasuresPayload,
    InstrumentValuationMetrics,
)
from api.services.instrument_cache_store import (
    instrument_yfinance_document_get as _instrument_cache_get,
    instrument_yfinance_document_put as _instrument_cache_put,
)
from api.services.yfinance_tables import (
    beta_from_valuation_measures,
    dataframe_to_records,
    dict_to_jsonable,
    merge_valuation_metrics,
    valuation_measures_to_metrics,
)
_log = logging.getLogger(__name__)


_MAX_FIN_ROWS = 45
_MAX_FIN_COLS = 8
_MAX_HOLDERS = 50
_MAX_TOP_HOLDINGS = 15
_MAX_EXECUTIVES = 20

_EXPIRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MAX_IV_HEATMAP_EXPIRIES = 40

_YAHOO_KIND_MAP: dict[str, InstrumentKindLiteral] = {
    "EQUITY": "equity",
    "ETF": "etf",
    "MUTUALFUND": "mutualfund",
    "OPTION": "option",
    "INDEX": "index",
    "CURRENCY": "currency",
    "FUTURE": "future",
    "CRYPTOCURRENCY": "crypto",
}


def normalize_instrument_kind(yahoo_quote_type: str | None) -> InstrumentKindLiteral:
    if not yahoo_quote_type or not isinstance(yahoo_quote_type, str):
        return "unknown"
    key = yahoo_quote_type.strip().upper()
    return _YAHOO_KIND_MAP.get(key, "unknown")


def _feature_flags(kind: InstrumentKindLiteral) -> InstrumentFeatureAvailability:
    fin_h = kind in ("equity", "etf", "mutualfund")
    chain = kind in ("equity", "etf", "index")
    return InstrumentFeatureAvailability(financials=fin_h, holders=fin_h, options_chain=chain)


def _num_opt(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return float(val)
    return None


def _int_opt(val: Any) -> int | None:
    n = _num_opt(val)
    if n is None:
        return None
    return int(n)


def _str_opt(val: Any) -> str | None:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _period_label(col: Any) -> str:
    if hasattr(col, "isoformat"):
        try:
            return col.isoformat()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            pass
    return str(col)


def _df_to_financial_table(
    df: pd.DataFrame | None, *, max_rows: int, max_cols: int
) -> tuple[list[str], list[InstrumentFinancialLineRow], bool]:
    if df is None or df.empty:
        return [], [], False
    truncated = False
    ncols = min(max_cols, df.shape[1])
    sub = df.iloc[:, -ncols:]
    if df.shape[1] > max_cols:
        truncated = True
    periods = [_period_label(c) for c in sub.columns]
    rows_out: list[InstrumentFinancialLineRow] = []
    idx_list = list(sub.index)[:max_rows]
    if len(sub.index) > max_rows:
        truncated = True
    for idx in idx_list:
        label = str(idx).strip() if idx is not None else ""
        if not label:
            continue
        vals: list[float | None] = []
        for c in sub.columns:
            raw = sub.loc[idx, c]
            if raw is None or (isinstance(raw, float) and (math.isnan(raw) or math.isinf(raw))):
                vals.append(None)
            elif isinstance(raw, (int, float)):
                vals.append(float(raw))
            else:
                vals.append(None)
        rows_out.append(InstrumentFinancialLineRow(label=label, values=vals))
    return periods, rows_out, truncated


def _valuation_from_info(info: dict[str, Any]) -> InstrumentValuationMetrics:
    """Scalars from ``Ticker.get_info()`` (yfinance flattened quoteSummary / quote)."""
    trailing_eps = _num_opt(info.get("trailingEps"))
    if trailing_eps is None:
        trailing_eps = _num_opt(info.get("epsTrailingTwelveMonths"))
    forward_eps = _num_opt(info.get("forwardEps"))
    if forward_eps is None:
        forward_eps = _num_opt(info.get("epsForward"))
    price_to_sales = _num_opt(info.get("priceToSalesTrailing12Months"))
    if price_to_sales is None:
        mcap = _num_opt(info.get("marketCap"))
        revenue = _num_opt(info.get("totalRevenue"))
        if mcap is not None and revenue is not None and revenue > 0:
            price_to_sales = mcap / revenue
    return InstrumentValuationMetrics(
        trailing_pe=_num_opt(info.get("trailingPE")),
        forward_pe=_num_opt(info.get("forwardPE")),
        trailing_eps=trailing_eps,
        forward_eps=forward_eps,
        return_on_equity=_num_opt(info.get("returnOnEquity")),
        return_on_assets=_num_opt(info.get("returnOnAssets")),
        price_to_book=_num_opt(info.get("priceToBook")),
        price_to_sales=price_to_sales,
        debt_to_equity=_num_opt(info.get("debtToEquity")),
    )


def _executives_from_info(info: dict[str, Any]) -> list[InstrumentExecutive]:
    raw = info.get("companyOfficers")
    if not isinstance(raw, list):
        return []
    out: list[InstrumentExecutive] = []
    for item in raw[:_MAX_EXECUTIVES]:
        if not isinstance(item, dict):
            continue
        name = _str_opt(item.get("name"))
        title = _str_opt(item.get("title"))
        yb = _int_opt(item.get("yearBorn"))
        if name or title:
            out.append(InstrumentExecutive(name=name, title=title, year_born=yb))
    return out


def _company_from_info(info: dict[str, Any]) -> InstrumentCompanyProfile | None:
    summary = _str_opt(info.get("longBusinessSummary")) or _str_opt(info.get("description"))
    sector = _str_opt(info.get("sector"))
    industry = _str_opt(info.get("industry"))
    addr = _str_opt(info.get("address1"))
    city = _str_opt(info.get("city"))
    state = _str_opt(info.get("state"))
    z = _str_opt(info.get("zip"))
    country = _str_opt(info.get("country"))
    phone = _str_opt(info.get("phone"))
    website = _str_opt(info.get("website"))
    emp = _int_opt(info.get("fullTimeEmployees"))
    if not any([summary, sector, industry, addr, city, website, emp is not None]):
        return None
    return InstrumentCompanyProfile(
        long_business_summary=summary,
        sector=sector,
        industry=industry,
        address_line1=addr,
        city=city,
        state=state,
        zip_code=z,
        country=country,
        phone=phone,
        website=website,
        full_time_employees=emp,
    )


def _expense_ratio_from_info(info: dict[str, Any]) -> float | None:
    for key in ("netExpenseRatio", "annualReportExpenseRatio", "expenseRatio"):
        v = _num_opt(info.get(key))
        if v is not None:
            return v
    return None


def _yield_pct_from_info(info: dict[str, Any]) -> float | None:
    y = _num_opt(info.get("yield"))
    if y is not None and y <= 1.0:
        return y * 100.0
    dy = _num_opt(info.get("dividendYield"))
    if dy is not None:
        if dy > 1.0:
            return dy
        return dy * 100.0
    tay = _num_opt(info.get("trailingAnnualDividendYield"))
    if tay is not None:
        if tay > 1.0:
            return tay
        return tay * 100.0
    return None


def _fund_top_holdings(t: yf.Ticker) -> list[InstrumentFundTopHolding]:
    out: list[InstrumentFundTopHolding] = []
    try:
        fd = getattr(t, "funds_data", None)
        if fd is None:
            return out
        th = getattr(fd, "top_holdings", None)
        if th is None or getattr(th, "empty", True):
            return out
        df = th
        for sym, row in df.head(_MAX_TOP_HOLDINGS).iterrows():
            sym_s = str(sym).strip().upper()
            if not sym_s:
                continue
            name = None
            if "Name" in df.columns:
                name = _str_opt(row.get("Name"))
            hp = _num_opt(row.get("Holding Percent"))
            pct = hp * 100.0 if hp is not None and hp <= 1.0 else hp
            out.append(InstrumentFundTopHolding(symbol=sym_s, name=name, holding_percent=pct))
    except Exception as exc:  # noqa: BLE001
        _log.debug("top_holdings unavailable: %s", exc)
    return out


def _fund_summary(t: yf.Ticker, info: dict[str, Any], kind: InstrumentKindLiteral) -> InstrumentFundSummary | None:
    if kind not in ("etf", "mutualfund"):
        return None
    fam = _str_opt(info.get("fundFamily"))
    cat = _str_opt(info.get("category"))
    exp = _expense_ratio_from_info(info)
    yld = _yield_pct_from_info(info)
    holdings = _fund_top_holdings(t)
    if not any([fam, cat, exp is not None, yld is not None, holdings]):
        return InstrumentFundSummary(top_holdings=holdings)
    return InstrumentFundSummary(
        fund_family=fam,
        category=cat,
        expense_ratio=exp,
        yield_pct=yld,
        top_holdings=holdings,
    )


def _option_contract_from_info(info: dict[str, Any]) -> InstrumentOptionContractSummary:
    exp_raw = info.get("expireDate")
    expire_s: str | None = None
    if isinstance(exp_raw, (int, float)):
        try:
            expire_s = datetime.fromtimestamp(int(exp_raw), tz=timezone.utc).date().isoformat()
        except Exception:  # noqa: BLE001
            expire_s = None
    ct = _str_opt(info.get("optionsType")) or _str_opt(info.get("contractType"))
    return InstrumentOptionContractSummary(
        underlying_symbol=_str_opt(info.get("underlyingSymbol")),
        strike=_num_opt(info.get("strike")) or _num_opt(info.get("strikePrice")),
        expire_date=expire_s,
        contract_type=ct,
        last_price=_num_opt(info.get("regularMarketPrice")) or _num_opt(info.get("currentPrice")),
        bid=_num_opt(info.get("bid")),
        ask=_num_opt(info.get("ask")),
        volume=_int_opt(info.get("volume")),
        open_interest=_int_opt(info.get("openInterest")),
        implied_volatility=_num_opt(info.get("impliedVolatility")),
    )


def fetch_instrument_overview(symbol: str) -> InstrumentOverviewResponse:
    hit = _instrument_cache_get(
        InstrumentOverviewResponse, symbol=symbol, resource_type=DOC_OVERVIEW, resource_key=""
    )
    if hit is not None:
        return hit

    adapter = YahooPublicAdapter()
    try:
        t = adapter.ticker(symbol)
        info = t.get_info()
        vm_df: pd.DataFrame | None = None
        try:
            vm_df = t.get_valuation_measures()
        except Exception as vm_exc:  # noqa: BLE001
            _log.warning("yfinance get_valuation_measures failed for %s: %s", symbol, vm_exc)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance overview failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail="instrument provider unavailable") from exc

    if not isinstance(info, dict):
        info = {}

    yahoo_qt = _str_opt(info.get("quoteType"))
    kind = normalize_instrument_kind(yahoo_qt)
    features = _feature_flags(kind)

    mcap = _num_opt(info.get("marketCap"))
    beta = _num_opt(info.get("beta"))
    vm_frame = vm_df if isinstance(vm_df, pd.DataFrame) else None
    vm_metrics = valuation_measures_to_metrics(vm_frame)
    beta_vm = beta_from_valuation_measures(vm_frame)
    if beta is None and beta_vm is not None:
        beta = beta_vm

    cols, recs = dataframe_to_records(vm_frame)
    vm_payload = InstrumentValuationMeasuresPayload(
        symbol=symbol,
        available=bool(recs),
        columns=cols,
        records=recs,
    )
    _instrument_cache_put(symbol=symbol, resource_type=DOC_VALUATION_MEASURES, resource_key="", obj=vm_payload)

    company = _company_from_info(info) if kind == "equity" else None
    executives = _executives_from_info(info) if kind == "equity" else []
    fund = _fund_summary(t, info, kind)
    opt_contract = _option_contract_from_info(info) if kind == "option" else None

    info_val = _valuation_from_info(info)
    merged_val = merge_valuation_metrics(info_val, vm_metrics)

    out = InstrumentOverviewResponse(
        symbol=symbol,
        instrument_kind=kind,
        yahoo_quote_type=yahoo_qt,
        short_name=_str_opt(info.get("shortName")),
        long_name=_str_opt(info.get("longName")),
        exchange=_str_opt(info.get("exchange")),
        currency=_str_opt(info.get("currency")),
        market_cap=mcap,
        beta=beta,
        valuation=merged_val,
        company=company,
        fund=fund,
        option_contract=opt_contract,
        executives=executives,
        features=features,
    )
    _instrument_cache_put(symbol=symbol, resource_type=DOC_OVERVIEW, resource_key="", obj=out)
    return out


def _financials_resource_type(statement: InstrumentStatementLiteral) -> str:
    if statement == "income":
        return DOC_FINANCIALS_INCOME
    if statement == "balance":
        return DOC_FINANCIALS_BALANCE
    return DOC_FINANCIALS_CASHFLOW


def _statement_attr(
    statement: InstrumentStatementLiteral, frequency: InstrumentFinancialFrequencyLiteral
) -> str:
    q = frequency == "quarterly"
    if statement == "income":
        return "quarterly_income_stmt" if q else "income_stmt"
    if statement == "balance":
        return "quarterly_balance_sheet" if q else "balance_sheet"
    return "quarterly_cashflow" if q else "cashflow"


def fetch_instrument_financials(
    symbol: str,
    *,
    statement: InstrumentStatementLiteral,
    frequency: InstrumentFinancialFrequencyLiteral,
    periods: int,
) -> InstrumentFinancialStatementResponse:
    cap = max(1, min(periods, _MAX_FIN_COLS))
    rk = f"{frequency}|{cap}"
    rt = _financials_resource_type(statement)
    hit = _instrument_cache_get(
        InstrumentFinancialStatementResponse,
        symbol=symbol,
        resource_type=rt,
        resource_key=rk,
    )
    if hit is not None:
        return hit

    adapter = YahooPublicAdapter()
    try:
        t = adapter.ticker(symbol)
        attr = _statement_attr(statement, frequency)
        df = getattr(t, attr, None)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance financials failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail="instrument provider unavailable") from exc

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return InstrumentFinancialStatementResponse(
            symbol=symbol,
            statement=statement,
            frequency=frequency,
            periods=[],
            rows=[],
            available=False,
        )

    periods_labels, rows, truncated = _df_to_financial_table(df, max_rows=_MAX_FIN_ROWS, max_cols=cap)
    out = InstrumentFinancialStatementResponse(
        symbol=symbol,
        statement=statement,
        frequency=frequency,
        periods=periods_labels,
        rows=rows,
        truncated=truncated,
        available=True,
    )
    _instrument_cache_put(symbol=symbol, resource_type=rt, resource_key=rk, obj=out)
    return out


def _holders_df_to_rows(df: pd.DataFrame | None) -> tuple[list[InstrumentHolderRow], bool]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [], False
    rows: list[InstrumentHolderRow] = []
    for _, r in df.head(_MAX_HOLDERS).iterrows():
        holder = _str_opt(r.get("Holder")) or "—"
        dr = r.get("Date Reported")
        dr_s: str | None = None
        if dr is not None and not (isinstance(dr, float) and pd.isna(dr)):
            if hasattr(dr, "date") and callable(getattr(dr, "date")):
                try:
                    dr_s = dr.date().isoformat()  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001
                    dr_s = str(dr)
            elif isinstance(dr, str):
                dr_s = dr.strip() or None
            else:
                dr_s = str(dr)
        pct = _num_opt(r.get("pctHeld"))
        pct_out = pct * 100.0 if pct is not None and pct <= 1.0 else pct
        rows.append(
            InstrumentHolderRow(
                holder=holder,
                date_reported=dr_s,
                shares=_num_opt(r.get("Shares")),
                value=_num_opt(r.get("Value")),
                percent_held=pct_out,
                percent_change=_num_opt(r.get("pctChange")),
            )
        )
    return rows, len(df) > _MAX_HOLDERS


def fetch_instrument_holders(symbol: str) -> InstrumentHoldersResponse:
    hit = _instrument_cache_get(
        InstrumentHoldersResponse, symbol=symbol, resource_type=DOC_HOLDERS, resource_key=""
    )
    if hit is not None:
        return hit

    adapter = YahooPublicAdapter()
    try:
        t = adapter.ticker(symbol)
        inst = t.institutional_holders
        mf = t.mutualfund_holders
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance holders failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail="instrument provider unavailable") from exc

    i_rows, _ = _holders_df_to_rows(inst if isinstance(inst, pd.DataFrame) else None)
    m_rows, _ = _holders_df_to_rows(mf if isinstance(mf, pd.DataFrame) else None)
    out = InstrumentHoldersResponse(
        symbol=symbol,
        institutional=i_rows,
        mutual_funds=m_rows,
        available_institutional=bool(i_rows),
        available_mutual_funds=bool(m_rows),
    )
    _instrument_cache_put(symbol=symbol, resource_type=DOC_HOLDERS, resource_key="", obj=out)
    return out


def fetch_option_expirations(symbol: str) -> InstrumentOptionExpirationsResponse:
    hit = _instrument_cache_get(
        InstrumentOptionExpirationsResponse,
        symbol=symbol,
        resource_type=DOC_OPTION_EXPIRATIONS,
        resource_key="",
    )
    if hit is not None:
        return hit

    adapter = YahooPublicAdapter()
    try:
        t = adapter.ticker(symbol)
        opts = getattr(t, "options", None)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance option expirations failed for %s: %s", symbol, exc)
        raise HTTPException(status_code=502, detail="instrument provider unavailable") from exc

    if opts is None:
        out = InstrumentOptionExpirationsResponse(symbol=symbol, expirations=[], available=False)
        _instrument_cache_put(
            symbol=symbol, resource_type=DOC_OPTION_EXPIRATIONS, resource_key="", obj=out
        )
        return out
    if isinstance(opts, (tuple, list)):
        expirations = [str(x) for x in opts]
    else:
        expirations = []
    out = InstrumentOptionExpirationsResponse(symbol=symbol, expirations=expirations, available=bool(expirations))
    _instrument_cache_put(symbol=symbol, resource_type=DOC_OPTION_EXPIRATIONS, resource_key="", obj=out)
    return out


def _option_frame_to_rows(df: pd.DataFrame | None) -> list[InstrumentOptionLegRow]:
    if df is None or df.empty:
        return []
    rows: list[InstrumentOptionLegRow] = []
    for _, r in df.iterrows():
        strike = _num_opt(r.get("strike"))
        if strike is None:
            continue
        iv = _num_opt(r.get("impliedVolatility"))
        rows.append(
            InstrumentOptionLegRow(
                strike=strike,
                last=_num_opt(r.get("lastPrice")),
                bid=_num_opt(r.get("bid")),
                ask=_num_opt(r.get("ask")),
                volume=_int_opt(r.get("volume")),
                open_interest=_int_opt(r.get("openInterest")),
                implied_volatility=iv,
            )
        )
    return rows


def fetch_option_chain(symbol: str, expiry: str) -> InstrumentOptionChainResponse:
    if not _EXPIRY_RE.match(expiry.strip()):
        raise HTTPException(status_code=400, detail="invalid expiry format (use YYYY-MM-DD)")
    expiry_clean = expiry.strip()

    hit = _instrument_cache_get(
        InstrumentOptionChainResponse,
        symbol=symbol,
        resource_type=DOC_OPTION_CHAIN,
        resource_key=expiry_clean,
    )
    if hit is not None:
        return hit

    adapter = YahooPublicAdapter()
    try:
        t = adapter.ticker(symbol)
        oc = t.option_chain(expiry_clean)
    except Exception as exc:  # noqa: BLE001
        _log.warning("yfinance option chain failed for %s %s: %s", symbol, expiry_clean, exc)
        raise HTTPException(status_code=502, detail="instrument provider unavailable") from exc

    calls = _option_frame_to_rows(oc.calls if hasattr(oc, "calls") else None)
    puts = _option_frame_to_rows(oc.puts if hasattr(oc, "puts") else None)
    out = InstrumentOptionChainResponse(
        symbol=symbol,
        expiry=expiry_clean,
        calls=calls,
        puts=puts,
        available=bool(calls or puts),
    )
    _instrument_cache_put(
        symbol=symbol, resource_type=DOC_OPTION_CHAIN, resource_key=expiry_clean, obj=out
    )
    return out


def fetch_option_iv_heatmap(symbol: str, max_expirations: int) -> InstrumentIvHeatmapResponse:
    """Build strike×expiry IV matrices from per-expiry chains (reuses chain cache)."""
    cap = max(1, min(int(max_expirations), _MAX_IV_HEATMAP_EXPIRIES))
    cache_key = str(cap)
    hit = _instrument_cache_get(
        InstrumentIvHeatmapResponse,
        symbol=symbol,
        resource_type=DOC_OPTION_IV_SURFACE,
        resource_key=cache_key,
    )
    if hit is not None:
        return hit

    exp_resp = fetch_option_expirations(symbol)
    if not exp_resp.available or not exp_resp.expirations:
        out = InstrumentIvHeatmapResponse(symbol=symbol, available=False)
        _instrument_cache_put(
            symbol=symbol, resource_type=DOC_OPTION_IV_SURFACE, resource_key=cache_key, obj=out
        )
        return out

    sorted_exp = sorted(exp_resp.expirations)
    capped = sorted_exp[:cap]

    chain_cols: list[tuple[str, InstrumentOptionChainResponse]] = []
    for exp in capped:
        try:
            ch = fetch_option_chain(symbol, exp)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "chain fetch failed"
            _log.warning("option chain skipped for %s %s: %s", symbol, exp, detail)
            continue
        if ch.calls or ch.puts:
            chain_cols.append((exp, ch))

    if not chain_cols:
        out = InstrumentIvHeatmapResponse(symbol=symbol, available=False)
        _instrument_cache_put(
            symbol=symbol, resource_type=DOC_OPTION_IV_SURFACE, resource_key=cache_key, obj=out
        )
        return out

    strikes_set: set[float] = set()
    for _, ch in chain_cols:
        for r in ch.calls:
            strikes_set.add(r.strike)
        for r in ch.puts:
            strikes_set.add(r.strike)
    strikes_sorted = sorted(strikes_set)
    strike_index = {s: i for i, s in enumerate(strikes_sorted)}
    expirations = [exp for exp, _ in chain_cols]
    n_strikes = len(strikes_sorted)
    n_exp = len(expirations)

    iv_calls: list[list[float | None]] = [[None] * n_exp for _ in range(n_strikes)]
    iv_puts: list[list[float | None]] = [[None] * n_exp for _ in range(n_strikes)]

    for j, (_, ch) in enumerate(chain_cols):
        for r in ch.calls:
            i = strike_index.get(r.strike)
            if i is not None:
                iv_calls[i][j] = r.implied_volatility
        for r in ch.puts:
            i = strike_index.get(r.strike)
            if i is not None:
                iv_puts[i][j] = r.implied_volatility

    out = InstrumentIvHeatmapResponse(
        symbol=symbol,
        expirations=expirations,
        strikes=strikes_sorted,
        iv_calls=iv_calls,
        iv_puts=iv_puts,
        available=True,
    )
    _instrument_cache_put(
        symbol=symbol, resource_type=DOC_OPTION_IV_SURFACE, resource_key=cache_key, obj=out
    )
    return out
