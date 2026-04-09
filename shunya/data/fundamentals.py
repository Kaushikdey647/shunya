from __future__ import annotations

import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Literal, Optional, Protocol, Sequence, runtime_checkable

import pandas as pd

from .timeframes import BarSpec, default_bar_spec, normalize_bar_timestamp

FundamentalSource = Literal[
    "income_statement",
    "balance_sheet_statement",
    "cash_flow_statement",
    "profitability_ratios",
    "liquidity_ratios",
    "solvency_ratios",
    "valuation_ratios",
]


@dataclass(frozen=True)
class FundamentalFieldSpec:
    """Normalized fundamental column derived from a FinanceToolkit source row."""

    attr: str
    column: str
    source: FundamentalSource
    source_label: str


_FUNDAMENTAL_FIELD_SPECS: tuple[FundamentalFieldSpec, ...] = (
    FundamentalFieldSpec("REVENUE", "Revenue", "income_statement", "Revenue"),
    FundamentalFieldSpec("NET_INCOME", "Net_Income", "income_statement", "Net Income"),
    FundamentalFieldSpec("EPS_DILUTED", "EPS_Diluted", "income_statement", "EPS Diluted"),
    FundamentalFieldSpec(
        "OPERATING_CASH_FLOW",
        "Operating_Cash_Flow",
        "cash_flow_statement",
        "Operating Cash Flow",
    ),
    FundamentalFieldSpec(
        "FREE_CASH_FLOW",
        "Free_Cash_Flow",
        "cash_flow_statement",
        "Free Cash Flow",
    ),
    FundamentalFieldSpec("TOTAL_ASSETS", "Total_Assets", "balance_sheet_statement", "Total Assets"),
    FundamentalFieldSpec("TOTAL_EQUITY", "Total_Equity", "balance_sheet_statement", "Total Equity"),
    FundamentalFieldSpec("TOTAL_DEBT", "Total_Debt", "balance_sheet_statement", "Total Debt"),
    FundamentalFieldSpec("CURRENT_RATIO", "Current_Ratio", "liquidity_ratios", "Current Ratio"),
    FundamentalFieldSpec("GROSS_MARGIN", "Gross_Margin", "profitability_ratios", "Gross Margin"),
    FundamentalFieldSpec(
        "OPERATING_MARGIN",
        "Operating_Margin",
        "profitability_ratios",
        "Operating Margin",
    ),
    FundamentalFieldSpec(
        "RETURN_ON_ASSETS",
        "Return_On_Assets",
        "profitability_ratios",
        "Return on Assets",
    ),
    FundamentalFieldSpec(
        "RETURN_ON_EQUITY",
        "Return_On_Equity",
        "profitability_ratios",
        "Return on Equity",
    ),
    FundamentalFieldSpec(
        "DEBT_TO_EQUITY",
        "Debt_To_Equity",
        "solvency_ratios",
        "Debt-to-Equity Ratio",
    ),
    FundamentalFieldSpec(
        "FREE_CASH_FLOW_YIELD",
        "Free_Cash_Flow_Yield",
        "solvency_ratios",
        "Free Cash Flow Yield",
    ),
    FundamentalFieldSpec(
        "PRICE_TO_EARNINGS",
        "Price_To_Earnings",
        "valuation_ratios",
        "Price-to-Earnings",
    ),
)

FUND = SimpleNamespace(**{spec.attr: spec.column for spec in _FUNDAMENTAL_FIELD_SPECS})
FUNDAMENTAL_FIELDS: tuple[str, ...] = tuple(spec.column for spec in _FUNDAMENTAL_FIELD_SPECS)
FUNDAMENTAL_FIELD_MAP: Dict[str, FundamentalFieldSpec] = {
    spec.column: spec for spec in _FUNDAMENTAL_FIELD_SPECS
}


def default_fundamental_fields() -> tuple[str, ...]:
    """Curated first-pass field set exposed by default."""
    return FUNDAMENTAL_FIELDS


def validate_fundamental_fields(fields: Optional[Sequence[str]]) -> tuple[FundamentalFieldSpec, ...]:
    """Resolve requested field names into field specifications."""
    if fields is None:
        return _FUNDAMENTAL_FIELD_SPECS
    resolved: list[FundamentalFieldSpec] = []
    missing: list[str] = []
    for field in fields:
        spec = FUNDAMENTAL_FIELD_MAP.get(str(field))
        if spec is None:
            missing.append(str(field))
        else:
            resolved.append(spec)
    if missing:
        raise KeyError(f"Unknown fundamental fields: {missing}")
    return tuple(resolved)


@runtime_checkable
class FundamentalDataProvider(Protocol):
    """
    Fetch period-end fundamentals for one or more equities.

    Implementations return a dataframe indexed by ``('Ticker', 'Date')`` where ``Date`` is
    the normalized report period end and columns are normalized field names from
    :data:`FUNDAMENTAL_FIELDS`.
    """

    def fetch(
        self,
        ticker_list: Sequence[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        fields: Optional[Sequence[str]] = None,
        quarterly: bool = True,
        bar_spec: Optional[BarSpec] = None,
    ) -> pd.DataFrame: ...


def _empty_periodic_frame(columns: Sequence[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_arrays(
        [pd.Index([], dtype="object"), pd.DatetimeIndex([], name="Date")],
        names=["Ticker", "Date"],
    )
    return pd.DataFrame(index=idx, columns=list(columns), dtype=float)


def _parse_period_label(label: object) -> pd.Timestamp:
    if isinstance(label, pd.Timestamp):
        return label
    if isinstance(label, pd.Period):
        return pd.Timestamp(label.end_time).normalize()
    text = str(label).strip()
    q = re.fullmatch(r"(\d{4})Q([1-4])", text)
    if q is not None:
        year = int(q.group(1))
        quarter = int(q.group(2))
        return pd.Period(year=year, quarter=quarter, freq="Q").end_time.normalize()
    y = re.fullmatch(r"(\d{4})", text)
    if y is not None:
        return pd.Timestamp(year=int(y.group(1)), month=12, day=31)
    try:
        return pd.Timestamp(text)
    except ValueError as exc:
        raise ValueError(f"Unable to parse FinanceToolkit period label {label!r}") from exc


def _coerce_row_frame(raw: pd.DataFrame | pd.Series, *, item: str, tickers: Sequence[str]) -> pd.DataFrame:
    if isinstance(raw, pd.Series):
        ticker = str(tickers[0]) if tickers else ""
        return pd.DataFrame([raw], index=pd.Index([ticker], name="Ticker"))

    if not isinstance(raw.index, pd.MultiIndex):
        if item not in raw.index:
            raise KeyError(item)
        row = raw.loc[item]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        ticker = str(tickers[0]) if tickers else ""
        return pd.DataFrame([row], index=pd.Index([ticker], name="Ticker"))

    for level in (-1, 1, 0):
        if abs(level) >= raw.index.nlevels:
            continue
        labels = raw.index.get_level_values(level).astype(str)
        if item not in set(labels.tolist()):
            continue
        row = raw.xs(item, level=level)
        if isinstance(row, pd.Series):
            ticker = str(tickers[0]) if tickers else ""
            return pd.DataFrame([row], index=pd.Index([ticker], name="Ticker"))
        if isinstance(row.index, pd.MultiIndex):
            row.index = row.index.get_level_values(0)
        row.index = pd.Index([str(v) for v in row.index], name="Ticker")
        return row

    raise KeyError(item)


def _extract_field_panel(
    raw: pd.DataFrame | pd.Series | None,
    *,
    spec: FundamentalFieldSpec,
    tickers: Sequence[str],
    bar_spec: BarSpec,
) -> pd.DataFrame:
    if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
        return _empty_periodic_frame([spec.column])

    row_frame = _coerce_row_frame(raw, item=spec.source_label, tickers=tickers)
    row_frame = row_frame.loc[[t for t in row_frame.index if str(t) in {str(x) for x in tickers}]]
    if row_frame.empty:
        return _empty_periodic_frame([spec.column])

    records: list[tuple[str, pd.Timestamp, float]] = []
    for ticker, row in row_frame.iterrows():
        for period_label, value in row.items():
            ts = normalize_bar_timestamp(_parse_period_label(period_label), bar_spec)
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            records.append((str(ticker), ts, float(numeric) if pd.notna(numeric) else float("nan")))

    out = pd.DataFrame(records, columns=["Ticker", "Date", spec.column])
    out = out.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    out = out.set_index(["Ticker", "Date"]).sort_index()
    return out


def align_fundamental_panel_to_panel_index(
    periodic: pd.DataFrame,
    panel_index: pd.Index | pd.MultiIndex,
) -> pd.DataFrame:
    """
    Forward-fill period-end fundamentals onto an existing panel index.

    The returned frame uses the same index ordering as ``panel_index`` and keeps the source
    columns unchanged.
    """
    if isinstance(panel_index, pd.MultiIndex):
        if tuple(panel_index.names) != ("Ticker", "Date"):
            raise ValueError(
                f"Expected panel index names ('Ticker', 'Date'), got {tuple(panel_index.names)!r}"
            )
        columns = list(periodic.columns)
        if not columns:
            return pd.DataFrame(index=panel_index)
        frames: list[pd.DataFrame] = []
        tickers = panel_index.get_level_values("Ticker").unique().tolist()
        for ticker in tickers:
            target = pd.DatetimeIndex(
                panel_index[panel_index.get_level_values("Ticker") == ticker].get_level_values("Date"),
                name="Date",
            )
            if isinstance(periodic.index, pd.MultiIndex) and ticker in periodic.index.get_level_values("Ticker"):
                source = periodic.xs(ticker, level="Ticker").sort_index()
            else:
                source = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"), columns=columns, dtype=float)
            aligned = source.reindex(source.index.union(target)).sort_index().ffill().reindex(target)
            frames.append(aligned)
        out = pd.concat(frames, keys=tickers, names=["Ticker", "Date"])
        return out.reindex(panel_index)

    if isinstance(periodic.index, pd.MultiIndex):
        tickers = periodic.index.get_level_values("Ticker").unique().tolist()
        if len(tickers) > 1:
            raise ValueError("Single-index panel cannot align multiple ticker fundamentals")
        periodic = periodic.xs(tickers[0], level="Ticker")
    target = pd.DatetimeIndex(panel_index, name=getattr(panel_index, "name", "Date"))
    return periodic.reindex(periodic.index.union(target)).sort_index().ffill().reindex(target)


class FinanceToolkitFundamentalDataProvider:
    """Equities-only fundamental adapter backed by FinanceToolkit / FinanceModelingPrep."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        enforce_source: Optional[str] = "FinancialModelingPrep",
        use_cached_data: bool | str = False,
        progress_bar: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("FMP_API_KEY") or os.environ.get(
            "FINANCIAL_MODELING_PREP_API_KEY"
        )
        self._enforce_source = enforce_source
        self._use_cached_data = use_cached_data
        self._progress_bar = bool(progress_bar)

    @staticmethod
    def _toolkit_class():
        try:
            from financetoolkit import Toolkit
        except ModuleNotFoundError as exc:
            name = getattr(exc, "name", "")
            if name == "yaml":
                raise ImportError(
                    "FinanceToolkit import requires the optional 'PyYAML' dependency. "
                    "Install PyYAML before attaching fundamentals."
                ) from exc
            raise
        return Toolkit

    def _fetch_source(self, toolkit: object, source: FundamentalSource) -> pd.DataFrame | pd.Series | None:
        if source == "income_statement":
            return toolkit.get_income_statement(progress_bar=self._progress_bar)
        if source == "balance_sheet_statement":
            return toolkit.get_balance_sheet_statement(progress_bar=self._progress_bar)
        if source == "cash_flow_statement":
            return toolkit.get_cash_flow_statement(progress_bar=self._progress_bar)
        if source == "profitability_ratios":
            return toolkit.ratios.collect_profitability_ratios()
        if source == "liquidity_ratios":
            return toolkit.ratios.collect_liquidity_ratios()
        if source == "solvency_ratios":
            return toolkit.ratios.collect_solvency_ratios()
        if source == "valuation_ratios":
            return toolkit.ratios.collect_valuation_ratios()
        raise ValueError(f"Unsupported fundamental source {source!r}")

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
        if not ticker_list:
            return _empty_periodic_frame([spec.column for spec in specs])

        Toolkit = self._toolkit_class()
        toolkit = Toolkit(
            list(ticker_list),
            api_key=self._api_key or "",
            start_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end_date=pd.Timestamp(end).strftime("%Y-%m-%d"),
            quarterly=quarterly,
            use_cached_data=self._use_cached_data,
            enforce_source=self._enforce_source,
            progress_bar=self._progress_bar,
        )
        use_spec = bar_spec if bar_spec is not None else default_bar_spec()
        grouped: dict[FundamentalSource, list[FundamentalFieldSpec]] = {}
        for spec in specs:
            grouped.setdefault(spec.source, []).append(spec)

        parts: list[pd.DataFrame] = []
        for source, source_specs in grouped.items():
            raw = self._fetch_source(toolkit, source)
            for spec in source_specs:
                panel = _extract_field_panel(raw, spec=spec, tickers=ticker_list, bar_spec=use_spec)
                parts.append(panel)

        if not parts:
            return _empty_periodic_frame([spec.column for spec in specs])
        out = pd.concat(parts, axis=1).sort_index()
        out = out.loc[:, ~out.columns.duplicated()]
        return out


__all__ = [
    "FUND",
    "FUNDAMENTAL_FIELDS",
    "FUNDAMENTAL_FIELD_MAP",
    "FundamentalDataProvider",
    "FundamentalFieldSpec",
    "FinanceToolkitFundamentalDataProvider",
    "align_fundamental_panel_to_panel_index",
    "default_fundamental_fields",
    "validate_fundamental_fields",
]
