from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

BarUnitLiteral = Literal["SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS", "YEARS"]
FeatureModeLiteral = Literal["full", "ohlcv_only"]
TradingAxisModeLiteral = Literal["observed", "canonical"]
MarketDataProviderLiteral = Literal["auto", "timescale", "yfinance"]

DecayModeLiteral = Literal["ema", "linear"]
NanPolicyLiteral = Literal["strict", "zero_fill"]
TemporalModeLiteral = Literal["bar_step", "elapsed_trading_time"]
NeutralizationLiteral = Literal["none", "market", "group"]
SectorCapModeLiteral = Literal["rescale", "raise"]
ConstraintsModeLiteral = Literal["rescale", "raise"]


class BarSpecModel(BaseModel):
    unit: BarUnitLiteral = "DAYS"
    step: int = Field(default=1, ge=1)


class FinTsRequest(BaseModel):
    start_date: str
    end_date: str
    ticker_list: list[str] = Field(min_length=1)
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


class AlphaCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None, max_length=2048)
    import_ref: str = Field(min_length=1, max_length=256)
    finstrat_config: FinStratConfig = Field(default_factory=FinStratConfig)


class AlphaPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    description: Optional[str] = Field(default=None, max_length=2048)
    import_ref: Optional[str] = Field(default=None, min_length=1, max_length=256)
    finstrat_config: Optional[FinStratConfig] = None


class AlphaOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    import_ref: str
    finstrat_config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class BacktestCreate(BaseModel):
    alpha_id: str
    fin_ts: FinTsRequest
    finstrat_override: Optional[FinStratConfig] = None
    finbt: FinBtConfig = Field(default_factory=FinBtConfig)
    benchmark_ticker: Optional[str] = Field(default=None, max_length=32)


class BacktestJobOut(BaseModel):
    id: str
    alpha_id: str
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
