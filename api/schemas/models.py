from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from api.alpha_validation import validate_import_ref
from shunya.schemas import (
    BarSpecModel,
    BarUnitLiteral,
    ConstraintsModeLiteral,
    DecayModeLiteral,
    FeatureModeLiteral,
    FinBtConfig,
    FinStratConfig,
    FinTsRequest,
    MarketDataProviderLiteral,
    NanPolicyLiteral,
    NeutralizationLiteral,
    SectorCapModeLiteral,
    TemporalModeLiteral,
    TradingAxisModeLiteral,
)

_SOURCE_MAX = 524_288  # 512 KiB, aligned with api.inline_alpha


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


class BacktestLogLineOut(BaseModel):
    ts: str
    message: str


class BacktestJobsDeleteBatchRequest(BaseModel):
    """Job ids to remove. Non-UUID strings are ignored. At most 200 unique valid UUIDs (see API)."""

    ids: list[str] = Field(..., min_length=1)


class BacktestJobsDeleteBatchOut(BaseModel):
    deleted: int


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


class InstrumentTickerNewsItem(BaseModel):
    """Ticker-scoped story from yfinance ``Ticker.news`` (structured where available)."""

    title: str
    link: Optional[str] = None
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    story_id: Optional[str] = None
    content_type: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    provider_url: Optional[str] = None
    provider_source_id: Optional[str] = None
    canonical_site: Optional[str] = None
    canonical_region: Optional[str] = None
    canonical_lang: Optional[str] = None
    is_hosted: Optional[bool] = None
    thumbnail_url: Optional[str] = None
    editors_pick: Optional[bool] = None
    is_premium_news: Optional[bool] = None
    is_premium_free_news: Optional[bool] = None


class InstrumentTickerNewsResponse(BaseModel):
    symbol: str
    news: list[InstrumentTickerNewsItem]


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


MoversKindLiteral = Literal["gainers", "losers", "active"]


class MarketSnapshotRow(BaseModel):
    symbol: str
    last: Optional[float] = None
    pct_change_1d: Optional[float] = None
    volume: Optional[float] = None
    sparkline_close: list[float] = Field(default_factory=list)


class MarketSnapshotRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=32)

    @model_validator(mode="after")
    def _non_empty_symbols(self) -> Self:
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        return self


class MarketSnapshotResponse(BaseModel):
    rows: list[MarketSnapshotRow]


class MarketMoverRow(BaseModel):
    ticker: str
    price: Optional[float] = None
    pct_change: Optional[float] = None
    volume: Optional[float] = None


class MarketMoversResponse(BaseModel):
    kind: MoversKindLiteral
    rows: list[MarketMoverRow]


class MarketHeadlineItem(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    published_at: Optional[str] = None


class MarketHeadlinesResponse(BaseModel):
    headlines: list[MarketHeadlineItem]
