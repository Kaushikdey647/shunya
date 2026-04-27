from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from backtest_api.alpha_validation import validate_import_ref

BarUnitLiteral = Literal["SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS", "YEARS"]
FeatureModeLiteral = Literal["full", "ohlcv_only"]
TradingAxisModeLiteral = Literal["observed", "canonical"]
MarketDataProviderLiteral = Literal["auto", "timescale", "yfinance"]

DecayModeLiteral = Literal["ema", "linear"]
NanPolicyLiteral = Literal["strict", "zero_fill"]
TemporalModeLiteral = Literal["bar_step", "elapsed_trading_time"]
NeutralizationLiteral = Literal["none", "market", "sector", "industry"]
SectorCapModeLiteral = Literal["rescale", "raise"]
ConstraintsModeLiteral = Literal["rescale", "raise"]


class BarSpecModel(BaseModel):
    unit: BarUnitLiteral = "DAYS"
    step: int = Field(default=1, ge=1)


class FinTsRequest(BaseModel):
    start_date: str
    end_date: str
    ticker_list: list[str] = Field(default_factory=list)
    bar_spec: Optional[BarSpecModel] = None
    market_data_provider: MarketDataProviderLiteral = "auto"
    attach_yfinance_classifications: bool = True
    attach_fundamentals: bool = False
    feature_mode: FeatureModeLiteral = "full"
    require_history_bars: Optional[int] = Field(default=None, ge=1)
    trading_axis_mode: TradingAxisModeLiteral = "observed"
    strict_trading_grid: bool = False
    strict_provider_universe: bool = True
    strict_ohlcv: bool = True
    strict_empty: bool = True


class FinStratConfig(BaseModel):
    decay_mode: DecayModeLiteral = "ema"
    decay: float = Field(default=0.0, ge=0.0, lt=1.0)
    decay_window: int = Field(default=1, ge=1)
    signal_delay: int = Field(default=0, ge=0)
    intraday_session_isolated_lag: bool = False
    nan_policy: NanPolicyLiteral = "strict"
    temporal_mode: TemporalModeLiteral = "bar_step"
    neutralization: NeutralizationLiteral = "market"

    @field_validator("neutralization", mode="before")
    @classmethod
    def _neutralization_legacy_group(cls, v: object) -> object:
        """Stored alphas may still have neutralization='group' (old FinStrat default path)."""
        if v == "group":
            return "sector"
        return v
    truncation: float = Field(default=0.0, ge=0.0, lt=0.5)
    max_single_weight: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    panel_columns: Optional[list[str]] = None


class FinBtConfig(BaseModel):
    cash: float = Field(default=100_000.0, gt=0.0)
    commission: float = Field(default=0.0, ge=0.0)
    slippage_pct: float = Field(default=0.0, ge=0.0)
    group_column: Optional[str] = None
    sector_gross_cap_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    sector_cap_mode: SectorCapModeLiteral = "rescale"
    sector_group_column: str = "Sector"
    group_net_cap_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    turnover_budget_fraction: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    adv_participation_fraction: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    constraints_mode: ConstraintsModeLiteral = "rescale"
    validate_finite_targets: bool = True


_SOURCE_MAX = 524_288  # 512 KiB, aligned with backtest_api.inline_alpha


class AlphaCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None, max_length=2048)
    import_ref: Optional[str] = Field(default=None, max_length=256)
    source_code: Optional[str] = Field(default=None, max_length=_SOURCE_MAX)
    finstrat_config: FinStratConfig = Field(default_factory=FinStratConfig)

    @field_validator("import_ref", "source_code", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    @model_validator(mode="after")
    def _ref_or_source(self) -> Self:
        if not (self.import_ref or self.source_code):
            raise ValueError("Provide import_ref and/or non-empty source_code (at least one).")
        if self.import_ref is not None:
            try:
                validate_import_ref(self.import_ref)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        return self


class AlphaPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None, max_length=2048)
    import_ref: Optional[str] = Field(default=None, max_length=256)
    source_code: Optional[str] = Field(default=None, max_length=_SOURCE_MAX)
    finstrat_config: Optional[FinStratConfig] = None

    @field_validator("import_ref", "source_code", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v


class AlphaOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    import_ref: Optional[str]
    source_code: Optional[str]
    finstrat_config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class BacktestCreate(BaseModel):
    alpha_id: str
    index_code: Optional[str] = Field(default=None, max_length=64)
    fin_ts: FinTsRequest
    finstrat_override: Optional[FinStratConfig] = None
    finbt: FinBtConfig = Field(default_factory=FinBtConfig)
    benchmark_ticker: Optional[str] = Field(default=None, max_length=32)
    include_test_period_in_results: bool = Field(
        default=False,
        description="If false, stored metrics and series exclude the test window (2025-01-01 onward).",
    )
    omit_index_members_missing_ohlcv: bool = Field(
        default=False,
        description=(
            "When index_code is set: drop constituents with no OHLCV in the backtest window instead "
            "of failing; benchmark ticker must still have bars. Default false (strict full universe)."
        ),
    )

    @model_validator(mode="after")
    def _tickers_or_index(self) -> Self:
        if (self.index_code or "").strip():
            return self
        if not self.fin_ts.ticker_list:
            raise ValueError("fin_ts.ticker_list must be non-empty when index_code is not set")
        return self


class EquityIndexOut(BaseModel):
    code: str
    display_name: str
    member_count: int
    benchmark_ticker: str = Field(
        ...,
        description="Raw index symbol for benchmarks (e.g. ^GSPC, ^BFX), not an ETF.",
    )


class BacktestJobOut(BaseModel):
    id: str
    alpha_id: str
    alpha_name: Optional[str] = None
    index_code: Optional[str] = None
    include_test_period_in_results: bool = False
    status: Literal["queued", "running", "succeeded", "failed"]
    error_message: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class BacktestResultOut(BaseModel):
    job_id: str
    metrics: dict[str, Any]
    equity_curve: list[dict[str, Any]]
    turnover_history: list[dict[str, Any]]
    benchmark: Optional[dict[str, Any]] = None


class DataSummaryRequest(FinTsRequest):
    columns: Optional[list[str]] = Field(
        default=None,
        description="Subset of numeric columns for NaN counts; default all numeric columns.",
    )

    @model_validator(mode="after")
    def _data_summary_requires_tickers(self) -> Self:
        if not self.ticker_list:
            raise ValueError("ticker_list must be non-empty for data summary requests")
        return self


class TickerNanRow(BaseModel):
    ticker: str
    nan_counts: dict[str, int]


class TickerRiskRow(BaseModel):
    ticker: str
    return_pct: Optional[float]
    risk_ann_pct: Optional[float]
    sharpe: Optional[float]
    sortino: Optional[float]


class DataSummaryResponse(BaseModel):
    tickers: list[str]
    columns_used: list[str]
    nan_counts: list[TickerNanRow]
    per_ticker_metrics: list[TickerRiskRow]
    bar_unit: str
    bar_step: int
    periods_per_year: float


DashboardBucketGranularityLiteral = Literal["day", "week", "month"]
DashboardBucketParamLiteral = Literal["auto", "day", "week", "month"]


class DashboardBucketMeta(BaseModel):
    index: int
    start: str
    end: str


class TickerDashboardRow(BaseModel):
    ticker: str
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    raw_bar_count: int = 0
    completeness_pct: float = 0.0
    longest_run_buckets: int = 0
    coverage: list[int] = Field(default_factory=list)
    return_pct: Optional[float] = None
    risk_ann_pct: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None


class ClassificationLabelCount(BaseModel):
    label: str
    count: int


class DataDashboardResponse(BaseModel):
    interval: str
    source: str
    bucket_granularity: DashboardBucketGranularityLiteral
    bucket_auto_subsampled: bool = False
    reference_start: str
    reference_end: str
    bucket_count: int
    ticker_count: int
    truncated: bool = False
    aggregate_mean_completeness_pct: float
    aggregate_median_completeness_pct: float
    completeness_histogram: list[int]
    buckets: list[DashboardBucketMeta]
    tickers: list[TickerDashboardRow]
    per_ticker_metrics: list[TickerRiskRow]
    bar_unit: str
    bar_step: int
    periods_per_year: float
    max_buckets: int = 200
    sector_counts: list[ClassificationLabelCount] = Field(default_factory=list)
    industry_counts: list[ClassificationLabelCount] = Field(default_factory=list)
    sub_industry_counts: list[ClassificationLabelCount] = Field(default_factory=list)


class InstrumentSearchQuote(BaseModel):
    symbol: str
    shortname: Optional[str] = None
    longname: Optional[str] = None
    exchange: Optional[str] = None
    quote_type: Optional[str] = None


class InstrumentSearchNewsItem(BaseModel):
    title: str
    link: Optional[str] = None
    publisher: Optional[str] = None


class InstrumentNavLink(BaseModel):
    title: str
    url: str


class InstrumentSearchResponse(BaseModel):
    quotes: list[InstrumentSearchQuote]
    news: list[InstrumentSearchNewsItem]
    nav_links: list[InstrumentNavLink] = Field(default_factory=list)


class OhlcvBar(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


InstrumentOhlcvDataSourceLiteral = Literal["timescale", "yfinance"]
InstrumentOhlcvStorageStatusLiteral = Literal["none", "ok", "failed", "deferred"]


class InstrumentOhlcvResponse(BaseModel):
    symbol: str
    interval: str
    period: str
    bars: list[OhlcvBar]
    data_source: InstrumentOhlcvDataSourceLiteral = "yfinance"
    storage_status: InstrumentOhlcvStorageStatusLiteral = "none"
    storage_error: Optional[str] = None
    storage_job_id: Optional[int] = None
    storage_skipped: bool = False


class IngestionRunOut(BaseModel):
    id: int
    job: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    provider: Optional[str] = None
    params: Optional[Any] = None
    rows_upserted: Optional[int] = None
    status: str
    error: Optional[str] = None


HealthComponentStatusLiteral = Literal["ok", "error"]
OverallHealthStatusLiteral = Literal["ok", "degraded", "error"]


class HealthComponentModel(BaseModel):
    status: HealthComponentStatusLiteral
    latency_ms: float = Field(ge=0.0)


class HealthResponseModel(BaseModel):
    status: OverallHealthStatusLiteral
    backend: HealthComponentModel
    database: HealthComponentModel
    yfinance: HealthComponentModel
