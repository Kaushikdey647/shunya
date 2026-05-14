import type { FinBtConfig, FinStratConfig, FinTsRequest } from './types'

/** Mirrors Pydantic defaults in api/schemas/models.py */
export const defaultFinStratConfig: FinStratConfig = {
  decay_mode: 'ema',
  decay: 0,
  decay_window: 1,
  signal_delay: 0,
  intraday_session_isolated_lag: false,
  nan_policy: 'strict',
  temporal_mode: 'bar_step',
  neutralization: 'market',
  truncation: 0,
  max_single_weight: undefined,
  panel_columns: undefined,
}

export const defaultFinBtConfig: FinBtConfig = {
  cash: 100_000,
  commission: 0,
  slippage_pct: 0,
  group_column: undefined,
  sector_gross_cap_fraction: undefined,
  sector_cap_mode: 'rescale',
  sector_group_column: 'Sector',
  group_net_cap_fraction: undefined,
  turnover_budget_fraction: undefined,
  adv_participation_fraction: undefined,
  constraints_mode: 'rescale',
  validate_finite_targets: true,
}

export function defaultFinTsRequest(
  partial: Pick<FinTsRequest, 'start_date' | 'end_date' | 'ticker_list'>,
): FinTsRequest {
  return {
    ...partial,
    bar_spec: undefined,
    market_data_provider: 'auto',
    attach_yfinance_classifications: true,
    attach_fundamentals: false,
    feature_mode: 'full',
    require_history_bars: undefined,
    trading_axis_mode: 'observed',
    strict_trading_grid: false,
    strict_provider_universe: true,
    strict_ohlcv: true,
    strict_empty: true,
  }
}
