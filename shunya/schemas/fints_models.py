"""Pydantic models shared by the Shunya library and the HTTP API (`api`) (FinTS / FinStrat / FinBT)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
    attach_fundamentals_annual: bool = False
    attach_fundamentals_daily: bool = False
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
