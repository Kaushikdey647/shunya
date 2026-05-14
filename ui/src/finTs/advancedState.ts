import type {
  BarUnit,
  FeatureMode,
  FinTsRequest,
  MarketDataProvider,
  TradingAxisMode,
} from '../api/types'

export type FinTsAdvancedState = {
  useBarSpec: boolean
  barUnit: BarUnit
  barStep: number
  marketDataProvider: MarketDataProvider
  featureMode: FeatureMode
  tradingAxisMode: TradingAxisMode
  attachYf: boolean
  attachFund: boolean
  strictTradingGrid: boolean
  strictProviderUniverse: boolean
  strictOhlcv: boolean
  strictEmpty: boolean
  requireHistoryBars: string
}

export function initialFinTsAdvancedState(): FinTsAdvancedState {
  return {
    useBarSpec: false,
    barUnit: 'DAYS',
    barStep: 1,
    marketDataProvider: 'auto',
    featureMode: 'full',
    tradingAxisMode: 'observed',
    attachYf: true,
    attachFund: false,
    strictTradingGrid: false,
    strictProviderUniverse: true,
    strictOhlcv: true,
    strictEmpty: true,
    requireHistoryBars: '',
  }
}

export function applyFinTsAdvanced(
  fin_ts: FinTsRequest,
  a: FinTsAdvancedState,
): void {
  fin_ts.market_data_provider = a.marketDataProvider
  fin_ts.attach_yfinance_classifications = a.attachYf
  fin_ts.attach_fundamentals = a.attachFund
  fin_ts.feature_mode = a.featureMode
  fin_ts.trading_axis_mode = a.tradingAxisMode
  fin_ts.strict_trading_grid = a.strictTradingGrid
  fin_ts.strict_provider_universe = a.strictProviderUniverse
  fin_ts.strict_ohlcv = a.strictOhlcv
  fin_ts.strict_empty = a.strictEmpty
  const rh = a.requireHistoryBars.trim()
  if (rh !== '') {
    const n = Number(rh)
    if (Number.isFinite(n) && n >= 1) fin_ts.require_history_bars = n
  } else {
    fin_ts.require_history_bars = undefined
  }
  if (a.useBarSpec) {
    fin_ts.bar_spec = {
      unit: a.barUnit,
      step: Math.max(1, Math.floor(a.barStep)),
    }
  } else {
    fin_ts.bar_spec = undefined
  }
}
