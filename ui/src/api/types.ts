/** API DTOs mirroring api OpenAPI — regenerate from `/openapi.json` (e.g. openapi-typescript) after backend schema changes. */
export type BarUnit =
  | 'SECONDS'
  | 'MINUTES'
  | 'HOURS'
  | 'DAYS'
  | 'WEEKS'
  | 'MONTHS'
  | 'YEARS'

export type FeatureMode = 'full' | 'ohlcv_only'
export type TradingAxisMode = 'observed' | 'canonical'
export type MarketDataProvider = 'auto' | 'timescale' | 'yfinance'
export type DecayMode = 'ema' | 'linear'
export type NanPolicy = 'strict' | 'zero_fill'
export type TemporalMode = 'bar_step' | 'elapsed_trading_time'
export type Neutralization = 'none' | 'market' | 'sector' | 'industry'
export type SectorCapMode = 'rescale' | 'raise'
export type ConstraintsMode = 'rescale' | 'raise'

export interface BarSpecModel {
  unit?: BarUnit
  step?: number
}

export interface FinTsRequest {
  start_date: string
  end_date: string
  ticker_list: string[]
  bar_spec?: BarSpecModel | null
  market_data_provider?: MarketDataProvider
  attach_yfinance_classifications?: boolean
  attach_fundamentals?: boolean
  feature_mode?: FeatureMode
  require_history_bars?: number | null
  trading_axis_mode?: TradingAxisMode
  strict_trading_grid?: boolean
  strict_provider_universe?: boolean
  strict_ohlcv?: boolean
  strict_empty?: boolean
}

export interface FinStratConfig {
  decay_mode?: DecayMode
  decay?: number
  decay_window?: number
  signal_delay?: number
  intraday_session_isolated_lag?: boolean
  nan_policy?: NanPolicy
  temporal_mode?: TemporalMode
  neutralization?: Neutralization
  truncation?: number
  max_single_weight?: number | null
  panel_columns?: string[] | null
}

export interface FinBtConfig {
  cash?: number
  commission?: number
  slippage_pct?: number
  group_column?: string | null
  sector_gross_cap_fraction?: number | null
  sector_cap_mode?: SectorCapMode
  sector_group_column?: string
  group_net_cap_fraction?: number | null
  turnover_budget_fraction?: number | null
  adv_participation_fraction?: number | null
  constraints_mode?: ConstraintsMode
  validate_finite_targets?: boolean
}

export interface AlphaCreate {
  name: string
  description?: string | null
  /** When set, takes precedence at backtest time. Either this or `import_ref` (or both). */
  source_code?: string | null
  import_ref?: string | null
  finstrat_config: FinStratConfig
  /** Optional saved universe used as default for portfolio union / backtests. */
  default_universe_id?: string | null
}

export interface AlphaPatch {
  name?: string | null
  description?: string | null
  import_ref?: string | null
  source_code?: string | null
  finstrat_config?: FinStratConfig | null
  default_universe_id?: string | null
}

export interface AlphaOut {
  id: string
  name: string
  description: string | null
  import_ref: string | null
  source_code: string | null
  finstrat_config: Record<string, unknown>
  default_universe_id?: string | null
  created_at: string
  updated_at: string
}

export interface AlphaLintDiagnostic {
  severity: string
  message: string
  startLineNumber: number
  startColumn: number
  endLineNumber: number
  endColumn: number
}

export interface AlphaLintBodyResponse {
  diagnostics: AlphaLintDiagnostic[]
}

export interface AlphaAssistMarker {
  severity: string
  message: string
  startLineNumber: number
  startColumn: number
  endLineNumber: number
  endColumn: number
}

export interface AlphaAssistIssue {
  id: string
  severity: string
  message: string
  startLineNumber: number
  startColumn: number
  endLineNumber: number
  endColumn: number
  corrected_body?: string | null
}

export interface AlphaAssistBodyResponse {
  issues: AlphaAssistIssue[]
  /** Coordinate mirror for legacy consumers. */
  markers: AlphaAssistMarker[]
}

export interface AlphaAssistBacktestReviewRequest {
  source_body: string
  alpha_name?: string | null
  alpha_description?: string | null
  metrics: Record<string, unknown>
  result_summary?: Record<string, unknown> | null
}

export interface AlphaAssistBacktestReviewResponse {
  summary_points: string[]
  risk_points: string[]
  summary_markdown: string
  suggested_body?: string | null
}

export interface EquityIndexOut {
  code: string
  display_name: string
  member_count: number
  /** Yahoo-style raw index symbol (e.g. ^GSPC). */
  benchmark_ticker: string
}

export interface UniverseOut {
  id: string
  name: string
  description: string | null
  member_count: number
  created_at: string
  updated_at: string
}

export interface UniverseCreate {
  name: string
  description?: string | null
}

export interface UniversePatch {
  name?: string | null
  description?: string | null
}

export interface UniverseMemberOut {
  ticker: string
  long_name?: string | null
  sector_disp?: string | null
  industry_disp?: string | null
}

export interface UniverseMembersMutationOut {
  changed: number
  member_count: number
}

export interface UniverseBreakdownSlice {
  label: string
  count: number
  fraction: number
}

export interface UniverseSummaryOut {
  member_count: number
  classified_for_breakdown_count: number
  sector_breakdown: UniverseBreakdownSlice[]
  industry_breakdown: UniverseBreakdownSlice[]
  fundamentals_coverage_count: number
  median_market_cap?: number | null
  mean_trailing_pe?: number | null
  median_beta?: number | null
}

export interface UniverseTickerWeight {
  ticker: string
  weight: number
}

export interface UniverseConcentrationOut {
  hhi: number
  cr5: number
  cr10: number
  weight_mode: 'mcap' | 'equal'
  mcap_weights_partial: boolean
  top_holdings: UniverseTickerWeight[]
}

export interface UniverseXsVolPoint {
  date: string
  xs_vol: number
}

export interface UniversePcaScorePoint {
  date: string
  score: number
}

export interface UniverseTickerLoading {
  ticker: string
  loading: number
}

export interface UniversePcaLoadingsScatterPoint {
  ticker: string
  pc1_loading: number
  pc2_loading: number
}

export interface UniverseReturnAnalyticsOut {
  universe_id: string
  period: string
  interval: string
  source: string
  start_date: string
  end_date_exclusive: string
  tickers: string[]
  n_observations: number
  alignment: string
  correlation_simple: number[][]
  correlation_log: number[][]
  cross_sectional_vol: UniverseXsVolPoint[]
  pca_explained_variance_ratio: number[]
  pca_pc1_loadings: UniverseTickerLoading[]
  pca_pc2_loadings: UniverseTickerLoading[]
  pca_pc1_scores: UniversePcaScorePoint[]
  pca_loadings_scatter: UniversePcaLoadingsScatterPoint[]
  concentration: UniverseConcentrationOut
}

export interface UniverseTickerListOut {
  tickers: string[]
}

export interface BacktestCreate {
  alpha_id: string
  /** When set, server resolves constituents from Timescale and sets raw index benchmark ticker. */
  index_code?: string | null
  /** Saved custom universe (mutually exclusive with index_code). Requires benchmark_ticker. */
  universe_id?: string | null
  fin_ts: FinTsRequest
  finstrat_override?: FinStratConfig | null
  finbt?: FinBtConfig
  benchmark_ticker?: string | null
  /** When true, result metrics and series include the test window from 2025-01-01 onward. */
  include_test_period_in_results?: boolean
  /**
   * When true with index_code: drop constituents with no OHLCV in the window; benchmark must still have bars.
   */
  omit_index_members_missing_ohlcv?: boolean
  /** When true with universe_id: drop members missing OHLCV in the window. */
  omit_universe_members_missing_ohlcv?: boolean
}

export type BacktestJobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface BacktestJobOut {
  id: string
  alpha_id: string
  alpha_name?: string | null
  index_code?: string | null
  universe_id?: string | null
  include_test_period_in_results?: boolean
  status: BacktestJobStatus
  error_message: string | null
  /** Stable failure code when the job failed (see API ErrorCode). */
  error_code?: string | null
  result_summary: Record<string, unknown> | null
  created_at: string
  started_at: string | null
  finished_at: string | null
}

export interface BacktestLogLine {
  ts: string
  message: string
}

export interface BacktestTargetHistoryRow {
  date: string
  targets: Record<string, unknown>
}

export interface BacktestGroupExposureRow {
  date: string
  gross_by_group: Record<string, unknown>
  net_by_group: Record<string, unknown>
}

/** Benchmark summary + optional `benchmark_equity_curve` (aligned buy-and-hold series). */
export type BacktestBenchmarkPayload = Record<string, unknown> & {
  error?: string
  ticker?: string
  correlation?: number | null
  n_overlap?: number
  benchmark_total_return_pct?: number | null
  benchmark_equity_curve?: Record<string, unknown>[]
}

export interface BacktestTradeEvent {
  ts: string
  ticker: string
  side: string
  size: number | null
  price: number | null
  value: number | null
}

export interface BacktestResultPayload {
  job_id: string
  metrics: Record<string, unknown>
  equity_curve: Record<string, unknown>[]
  turnover_history: Record<string, unknown>[]
  turnover_pct_history?: Record<string, unknown>[]
  benchmark?: BacktestBenchmarkPayload | null
  returns_analysis?: unknown
  drawdown_analysis?: unknown
  sharpe_analysis?: unknown
  target_history?: BacktestTargetHistoryRow[] | unknown[]
  group_exposure_history?: BacktestGroupExposureRow[]
  exposure_history?: Record<string, unknown>[]
  trade_events?: BacktestTradeEvent[] | Record<string, unknown>[]
  return_quantiles?: Record<string, unknown>
  tearsheet_summary?: Record<string, unknown>
  ff_single_factor?: Record<string, unknown>
}

export interface DataSummaryRequest extends FinTsRequest {
  columns?: string[] | null
}

export interface TickerNanRow {
  ticker: string
  nan_counts: Record<string, number>
}

export interface TickerRiskRow {
  ticker: string
  return_pct: number | null
  log_return_pct?: number | null
  risk_ann_pct: number | null
  sharpe: number | null
  sortino: number | null
}

export interface DataSummaryResponse {
  tickers: string[]
  columns_used: string[]
  nan_counts: TickerNanRow[]
  per_ticker_metrics: TickerRiskRow[]
  bar_unit: string
  bar_step: number
  periods_per_year: number
}

export type DashboardBucketGranularity = 'day' | 'week' | 'month'

export type DashboardBucketParam = 'auto' | DashboardBucketGranularity

export interface DashboardBucketMeta {
  index: number
  start: string
  end: string
}

export interface TickerDashboardRow extends TickerRiskRow {
  first_ts: string | null
  last_ts: string | null
  raw_bar_count: number
  completeness_pct: number
  longest_run_buckets: number
  coverage: number[]
}

export interface ClassificationLabelCount {
  label: string
  count: number
}

export interface DataDashboardResponse {
  interval: string
  source: string
  bucket_granularity: DashboardBucketGranularity
  bucket_auto_subsampled: boolean
  reference_start: string
  reference_end: string
  bucket_count: number
  ticker_count: number
  truncated: boolean
  aggregate_mean_completeness_pct: number
  aggregate_median_completeness_pct: number
  completeness_histogram: number[]
  buckets: DashboardBucketMeta[]
  tickers: TickerDashboardRow[]
  per_ticker_metrics: TickerRiskRow[]
  bar_unit: string
  bar_step: number
  periods_per_year: number
  max_buckets: number
  sector_counts: ClassificationLabelCount[]
  industry_counts: ClassificationLabelCount[]
}

export type HealthComponentStatus = 'ok' | 'error' | 'skipped'

export type OverallHealthStatus = 'ok' | 'degraded' | 'error'

export interface HealthCheckItem {
  status: HealthComponentStatus
  latency_ms: number
}

export interface HealthResponse {
  status: OverallHealthStatus
  backend: HealthCheckItem
  database: HealthCheckItem
  yfinance: HealthCheckItem
  alpaca: HealthCheckItem
}

function isCoreHealthCheckItem(v: unknown): v is HealthCheckItem {
  if (v === null || typeof v !== 'object') return false
  const h = v as Record<string, unknown>
  return (
    (h.status === 'ok' || h.status === 'error') &&
    typeof h.latency_ms === 'number'
  )
}

function isAlpacaHealthField(v: unknown): v is HealthCheckItem {
  if (v === null || typeof v !== 'object') return false
  const h = v as Record<string, unknown>
  return (
    (h.status === 'ok' || h.status === 'error' || h.status === 'skipped') &&
    typeof h.latency_ms === 'number'
  )
}

/** True when the JSON body matches {@link HealthResponse} (avoids crashes on HTML/plaintext/wrong API). */
export function isHealthResponse(x: unknown): x is HealthResponse {
  if (x === null || typeof x !== 'object') return false
  const o = x as Record<string, unknown>
  const s = o.status
  if (s !== 'ok' && s !== 'degraded' && s !== 'error') return false
  return (
    isCoreHealthCheckItem(o.backend) &&
    isCoreHealthCheckItem(o.database) &&
    isCoreHealthCheckItem(o.yfinance) &&
    isAlpacaHealthField(o.alpaca)
  )
}

export type TunableValueSource = 'database' | 'environment'

export interface AppSettingsEnvironment {
  database_configured: boolean
  alpaca_enabled: boolean
  ollama_host_configured: boolean
  trade_desk_write_configured: boolean
}

/** Effective operator tunables (env merged with optional DB overlay). */
export interface AppRuntimeTunables {
  worker_poll_interval_seconds: number
  max_target_history_points: number
  max_group_exposure_history_points: number
  max_exposure_history_points: number
  max_trade_events: number
  index_ohlcv_backfill_batch_size: number
  market_data_cache_ttl_days: number
  ollama_timeout_seconds: number
  ollama_model: string
}

export type AppTunableSources = { [K in keyof AppRuntimeTunables]: TunableValueSource }

export interface AppSettingsResponse {
  environment: AppSettingsEnvironment
  runtime: AppRuntimeTunables
  sources: AppTunableSources
}

export type AppRuntimeTunableKey = keyof AppRuntimeTunables

export const APP_RUNTIME_TUNABLE_KEYS = [
  'worker_poll_interval_seconds',
  'max_target_history_points',
  'max_group_exposure_history_points',
  'max_exposure_history_points',
  'max_trade_events',
  'index_ohlcv_backfill_batch_size',
  'market_data_cache_ttl_days',
  'ollama_timeout_seconds',
  'ollama_model',
] as const satisfies readonly AppRuntimeTunableKey[]

/** Alpaca ``/v2/account`` subset (equities; crypto/options fields stripped server-side). */
export interface AlpacaEquityAccountOut {
  id?: string | null
  account_number?: string | null
  status?: string | null
  currency?: string | null
  buying_power?: string | null
  regt_buying_power?: string | null
  daytrading_buying_power?: string | null
  effective_buying_power?: string | null
  non_marginable_buying_power?: string | null
  bod_dtbp?: string | null
  cash?: string | null
  accrued_fees?: string | null
  pending_transfer_out?: string | null
  pending_transfer_in?: string | null
  portfolio_value?: string | null
  pattern_day_trader?: boolean | null
  trading_blocked?: boolean | null
  transfers_blocked?: boolean | null
  account_blocked?: boolean | null
  created_at?: string | null
  trade_suspended_by_user?: boolean | null
  multiplier?: string | null
  shorting_enabled?: boolean | null
  equity?: string | null
  last_equity?: string | null
  long_market_value?: string | null
  short_market_value?: string | null
  position_market_value?: string | null
  initial_margin?: string | null
  maintenance_margin?: string | null
  last_maintenance_margin?: string | null
  sma?: string | null
  daytrade_count?: number | null
  balance_asof?: string | null
  intraday_adjustments?: string | null
  pending_reg_taf_fees?: string | null
}

export interface AlpacaAccountActivityOut {
  id: string
  activity_type: string
  transaction_time?: string | null
  date?: string | null
  symbol?: string | null
  qty?: number | null
  price?: number | null
  side?: string | null
  net_amount?: number | null
  description?: string | null
  order_id?: string | null
  trade_activity_type?: string | null
}

export interface AlpacaAccountActivitiesResponse {
  activities: AlpacaAccountActivityOut[]
  next_page_token?: string | null
}

export interface AlpacaPortfolioHistoryOut {
  timestamp: number[]
  equity: number[]
  profit_loss: number[]
  profit_loss_pct: (number | null)[]
  base_value?: number | null
  timeframe: string
  cashflow: Record<string, number[]>
}

export interface AlpacaAccountConfigurationsOut {
  dtbp_check: string
  fractional_trading: boolean
  max_margin_multiplier: string
  no_shorting: boolean
  pdt_check: string
  suspend_trade: boolean
  trade_confirm_email: string
  ptp_no_exception_entry: boolean
  max_options_trading_level?: number | null
}

export type AlpacaAccountConfigurationsPatch = Partial<
  Pick<
    AlpacaAccountConfigurationsOut,
    | 'dtbp_check'
    | 'fractional_trading'
    | 'max_margin_multiplier'
    | 'no_shorting'
    | 'pdt_check'
    | 'suspend_trade'
    | 'trade_confirm_email'
    | 'ptp_no_exception_entry'
    | 'max_options_trading_level'
  >
>

export type MoversKind = 'gainers' | 'losers' | 'active'

export interface OhlcvProvenance {
  read_path: 'timescale' | 'live_fetch'
  upstream_source_id: string
  route_rule_id?: string | null
  attempted_sources?: string[] | null
  partial_coverage?: boolean | null
}

export interface MarketSnapshotRow {
  symbol: string
  last: number | null
  pct_change_1d: number | null
  volume: number | null
  sparkline_close: number[]
  provenance?: OhlcvProvenance | null
}

export interface MarketSnapshotResponse {
  rows: MarketSnapshotRow[]
}

export interface MarketMoverRow {
  ticker: string
  price: number | null
  pct_change: number | null
  volume: number | null
}

export interface MarketMoversResponse {
  kind: MoversKind
  rows: MarketMoverRow[]
}

export interface MarketHeadlineItem {
  title: string
  publisher?: string | null
  link?: string | null
  published_at?: string | null
}

export interface MarketHeadlinesResponse {
  headlines: MarketHeadlineItem[]
}

export interface InstrumentSearchQuote {
  symbol: string
  shortname?: string | null
  longname?: string | null
  exchange?: string | null
  quote_type?: string | null
}

export interface InstrumentSearchNewsItem {
  title: string
  link?: string | null
  publisher?: string | null
}

export interface InstrumentNavLink {
  title: string
  url: string
}

export interface InstrumentSearchResponse {
  quotes: InstrumentSearchQuote[]
  news: InstrumentSearchNewsItem[]
  nav_links: InstrumentNavLink[]
}

export interface InstrumentTickerNewsItem {
  title: string
  link?: string | null
  publisher?: string | null
  published_at?: string | null
  story_id?: string | null
  content_type?: string | null
  summary?: string | null
  description?: string | null
  provider_url?: string | null
  provider_source_id?: string | null
  canonical_site?: string | null
  canonical_region?: string | null
  canonical_lang?: string | null
  is_hosted?: boolean | null
  thumbnail_url?: string | null
  editors_pick?: boolean | null
  is_premium_news?: boolean | null
  is_premium_free_news?: boolean | null
}

export interface InstrumentTickerNewsResponse {
  symbol: string
  news: InstrumentTickerNewsItem[]
}

export interface OhlcvBar {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume?: number | null
}

export type InstrumentOhlcvStorageStatus = 'none' | 'ok' | 'failed' | 'deferred'

export interface InstrumentOhlcvResponse {
  symbol: string
  interval: string
  period: string
  bars: OhlcvBar[]
  provenance?: OhlcvProvenance | null
  storage_status?: InstrumentOhlcvStorageStatus
  storage_error?: string | null
  storage_job_id?: number | null
  storage_skipped?: boolean
}

export type InstrumentKind =
  | 'equity'
  | 'etf'
  | 'mutualfund'
  | 'option'
  | 'index'
  | 'currency'
  | 'future'
  | 'crypto'
  | 'structured'
  | 'unknown'

export type InstrumentStatement = 'income' | 'balance' | 'cashflow'

export type InstrumentFinancialFrequency = 'quarterly' | 'annual'

export interface InstrumentFeatureAvailability {
  financials: boolean
  holders: boolean
  options_chain: boolean
}

export interface InstrumentValuationMetrics {
  trailing_pe?: number | null
  forward_pe?: number | null
  trailing_eps?: number | null
  forward_eps?: number | null
  return_on_equity?: number | null
  return_on_assets?: number | null
  price_to_book?: number | null
  price_to_sales?: number | null
  debt_to_equity?: number | null
}

export interface InstrumentExecutive {
  name?: string | null
  title?: string | null
  year_born?: number | null
}

export interface InstrumentCompanyProfile {
  long_business_summary?: string | null
  sector?: string | null
  industry?: string | null
  address_line1?: string | null
  city?: string | null
  state?: string | null
  zip_code?: string | null
  country?: string | null
  phone?: string | null
  website?: string | null
  full_time_employees?: number | null
}

export interface InstrumentFundTopHolding {
  symbol: string
  name?: string | null
  holding_percent?: number | null
}

export interface InstrumentFundSummary {
  fund_family?: string | null
  category?: string | null
  expense_ratio?: number | null
  yield_pct?: number | null
  top_holdings?: InstrumentFundTopHolding[]
}

export interface InstrumentOptionContractSummary {
  underlying_symbol?: string | null
  strike?: number | null
  expire_date?: string | null
  contract_type?: string | null
  last_price?: number | null
  bid?: number | null
  ask?: number | null
  volume?: number | null
  open_interest?: number | null
  implied_volatility?: number | null
}

export interface InstrumentOverviewResponse {
  symbol: string
  instrument_kind: InstrumentKind
  yahoo_quote_type?: string | null
  short_name?: string | null
  long_name?: string | null
  exchange?: string | null
  currency?: string | null
  market_cap?: number | null
  beta?: number | null
  valuation: InstrumentValuationMetrics
  company?: InstrumentCompanyProfile | null
  fund?: InstrumentFundSummary | null
  option_contract?: InstrumentOptionContractSummary | null
  executives: InstrumentExecutive[]
  features: InstrumentFeatureAvailability
}

export interface InstrumentFinancialLineRow {
  label: string
  values: (number | null)[]
}

export interface InstrumentFinancialStatementResponse {
  symbol: string
  statement: InstrumentStatement
  frequency: InstrumentFinancialFrequency
  periods: string[]
  rows: InstrumentFinancialLineRow[]
  truncated: boolean
  available: boolean
}

export interface InstrumentHolderRow {
  holder: string
  date_reported?: string | null
  shares?: number | null
  value?: number | null
  percent_held?: number | null
  percent_change?: number | null
}

export interface InstrumentHoldersResponse {
  symbol: string
  institutional: InstrumentHolderRow[]
  mutual_funds: InstrumentHolderRow[]
  available_institutional: boolean
  available_mutual_funds: boolean
}

export interface InstrumentOptionExpirationsResponse {
  symbol: string
  expirations: string[]
  available: boolean
}

export interface InstrumentOptionLegRow {
  strike: number
  last?: number | null
  bid?: number | null
  ask?: number | null
  volume?: number | null
  open_interest?: number | null
  implied_volatility?: number | null
}

export interface InstrumentOptionChainResponse {
  symbol: string
  expiry: string
  calls: InstrumentOptionLegRow[]
  puts: InstrumentOptionLegRow[]
  available: boolean
}

export interface InstrumentIvHeatmapResponse {
  symbol: string
  expirations: string[]
  strikes: number[]
  iv_calls: (number | null)[][]
  iv_puts: (number | null)[][]
  available: boolean
}

/** Cached ``get_valuation_measures()``-style table (columns + records). */
export interface InstrumentValuationMeasuresPayload {
  symbol: string
  available: boolean
  columns: string[]
  records: Record<string, unknown>[]
}

export interface InstrumentYfinanceTablePayload {
  columns: string[]
  records: Record<string, unknown>[]
}

export interface InstrumentYfinanceTableResponse {
  symbol: string
  available: boolean
  data: InstrumentYfinanceTablePayload
}

export interface InstrumentAnalystPriceTargetsResponse {
  symbol: string
  available: boolean
  current?: number | null
  low?: number | null
  high?: number | null
  mean?: number | null
  median?: number | null
}

export interface InstrumentJsonBlobResponse {
  symbol: string
  available: boolean
  data: Record<string, unknown>
}
