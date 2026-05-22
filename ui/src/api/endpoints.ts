import { ApiError, apiFetch } from './client'
import type {
  AlpacaAccountActivitiesResponse,
  AlpacaAccountConfigurationsOut,
  AlpacaAccountConfigurationsPatch,
  AlpacaEquityAccountOut,
  AlpacaPortfolioHistoryOut,
  AppRuntimeTunables,
  AppSettingsResponse,
  AlphaAssistBacktestReviewRequest,
  AlphaAssistBacktestReviewResponse,
  AlphaAssistBodyResponse,
  AlphaCreate,
  AlphaLintBodyResponse,
  AlphaOut,
  AlphaPatch,
  BacktestCreate,
  BacktestJobOut,
  BacktestLogLine,
  BacktestResultPayload,
  DashboardBucketParam,
  DataDashboardResponse,
  DataSummaryRequest,
  DataSummaryResponse,
  HealthResponse,
  InstrumentFinancialFrequency,
  InstrumentFinancialStatementResponse,
  InstrumentHoldersResponse,
  InstrumentOhlcvResponse,
  InstrumentAnalystPriceTargetsResponse,
  InstrumentIvHeatmapResponse,
  InstrumentJsonBlobResponse,
  InstrumentOptionChainResponse,
  InstrumentOptionExpirationsResponse,
  InstrumentOverviewResponse,
  InstrumentSearchResponse,
  InstrumentStatement,
  InstrumentTickerNewsResponse,
  InstrumentValuationMeasuresPayload,
  InstrumentYfinanceTableResponse,
  EquityIndexOut,
  UniverseCreate,
  UniverseMemberOut,
  UniverseMembersMutationOut,
  UniverseOut,
  UniversePatch,
  UniverseReturnAnalyticsOut,
  UniverseSummaryOut,
  UniverseTickerListOut,
  MarketHeadlinesResponse,
  MarketMoversResponse,
  MarketSnapshotResponse,
  MoversKind,
} from './types'

export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health', { method: 'GET' })
}

export async function getAppSettings(): Promise<AppSettingsResponse> {
  return apiFetch<AppSettingsResponse>('/settings/app', { method: 'GET' })
}

export async function patchAppSettings(
  body: Partial<AppRuntimeTunables>,
  tradeDeskToken: string,
): Promise<AppSettingsResponse> {
  return apiFetch<AppSettingsResponse>('/settings/app', {
    method: 'PATCH',
    body: JSON.stringify(body),
    headers: { 'X-Shunya-Trade-Desk-Token': tradeDeskToken },
  })
}

function tradeDeskHeaders(tradeDeskToken: string): HeadersInit {
  return { 'X-Shunya-Trade-Desk-Token': tradeDeskToken }
}

export async function getTradeAccountEquity(
  tradeDeskToken: string,
): Promise<AlpacaEquityAccountOut> {
  return apiFetch<AlpacaEquityAccountOut>('/trade/account/equity', {
    method: 'GET',
    headers: tradeDeskHeaders(tradeDeskToken),
  })
}

export async function getTradeAccountActivities(
  tradeDeskToken: string,
  params: {
    page_size?: number
    page_token?: string | null
    activity_types?: string | null
    after?: string | null
    until?: string | null
  } = {},
): Promise<AlpacaAccountActivitiesResponse> {
  const sp = new URLSearchParams()
  if (params.page_size != null) sp.set('page_size', String(params.page_size))
  if (params.page_token) sp.set('page_token', params.page_token)
  if (params.activity_types) sp.set('activity_types', params.activity_types)
  if (params.after) sp.set('after', params.after)
  if (params.until) sp.set('until', params.until)
  const q = sp.toString()
  return apiFetch<AlpacaAccountActivitiesResponse>(
    `/trade/account/activities${q ? `?${q}` : ''}`,
    {
      method: 'GET',
      headers: tradeDeskHeaders(tradeDeskToken),
    },
  )
}

export async function getTradePortfolioHistory(
  tradeDeskToken: string,
  params: {
    period: string
    timeframe?: string | null
    date_end?: string | null
    extended_hours?: boolean | null
    intraday_reporting?: string | null
    pnl_reset?: string | null
  },
): Promise<AlpacaPortfolioHistoryOut> {
  const sp = new URLSearchParams({ period: params.period })
  if (params.timeframe) sp.set('timeframe', params.timeframe)
  if (params.date_end) sp.set('date_end', params.date_end)
  if (params.extended_hours != null) sp.set('extended_hours', String(params.extended_hours))
  if (params.intraday_reporting) sp.set('intraday_reporting', params.intraday_reporting)
  if (params.pnl_reset) sp.set('pnl_reset', params.pnl_reset)
  return apiFetch<AlpacaPortfolioHistoryOut>(
    `/trade/account/portfolio-history?${sp.toString()}`,
    {
      method: 'GET',
      headers: tradeDeskHeaders(tradeDeskToken),
    },
  )
}

export async function getTradeAccountConfigurations(
  tradeDeskToken: string,
): Promise<AlpacaAccountConfigurationsOut> {
  return apiFetch<AlpacaAccountConfigurationsOut>('/trade/account/configurations', {
    method: 'GET',
    headers: tradeDeskHeaders(tradeDeskToken),
  })
}

export async function patchTradeAccountConfigurations(
  tradeDeskToken: string,
  body: AlpacaAccountConfigurationsPatch,
): Promise<AlpacaAccountConfigurationsOut> {
  return apiFetch<AlpacaAccountConfigurationsOut>('/trade/account/configurations', {
    method: 'PATCH',
    body: JSON.stringify(body),
    headers: tradeDeskHeaders(tradeDeskToken),
  })
}

export async function postMarketSnapshot(body: {
  symbols: string[]
}): Promise<MarketSnapshotResponse> {
  return apiFetch<MarketSnapshotResponse>('/market/snapshot', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function getMarketMovers(params: {
  kind: MoversKind
  limit?: number
}): Promise<MarketMoversResponse> {
  const sp = new URLSearchParams({ kind: params.kind })
  if (params.limit != null) sp.set('limit', String(params.limit))
  return apiFetch<MarketMoversResponse>(`/market/movers?${sp.toString()}`, {
    method: 'GET',
  })
}

export async function getMarketHeadlines(params?: {
  limit?: number
}): Promise<MarketHeadlinesResponse> {
  const sp = new URLSearchParams()
  if (params?.limit != null) sp.set('limit', String(params.limit))
  const q = sp.toString()
  return apiFetch<MarketHeadlinesResponse>(
    `/market/headlines${q ? `?${q}` : ''}`,
    { method: 'GET' },
  )
}

export async function listEquityIndices(): Promise<EquityIndexOut[]> {
  return apiFetch<EquityIndexOut[]>('/indices', { method: 'GET' })
}

export async function listUniverses(params?: {
  limit?: number
  offset?: number
}): Promise<UniverseOut[]> {
  const sp = new URLSearchParams()
  if (params?.limit != null) sp.set('limit', String(params.limit))
  if (params?.offset != null) sp.set('offset', String(params.offset))
  const q = sp.toString()
  return apiFetch<UniverseOut[]>(`/universes${q ? `?${q}` : ''}`, { method: 'GET' })
}

export async function createUniverse(body: UniverseCreate): Promise<UniverseOut> {
  return apiFetch<UniverseOut>('/universes', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function getUniverse(universeId: string): Promise<UniverseOut> {
  return apiFetch<UniverseOut>(`/universes/${encodeURIComponent(universeId)}`, { method: 'GET' })
}

export async function patchUniverse(
  universeId: string,
  body: UniversePatch,
): Promise<UniverseOut> {
  return apiFetch<UniverseOut>(`/universes/${encodeURIComponent(universeId)}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  })
}

export async function deleteUniverse(universeId: string): Promise<void> {
  await apiFetch<void>(`/universes/${encodeURIComponent(universeId)}`, { method: 'DELETE' })
}

export async function listUniverseMembers(
  universeId: string,
  params?: { limit?: number; offset?: number },
): Promise<UniverseMemberOut[]> {
  const sp = new URLSearchParams()
  if (params?.limit != null) sp.set('limit', String(params.limit))
  if (params?.offset != null) sp.set('offset', String(params.offset))
  const q = sp.toString()
  return apiFetch<UniverseMemberOut[]>(
    `/universes/${encodeURIComponent(universeId)}/members${q ? `?${q}` : ''}`,
    { method: 'GET' },
  )
}

export async function getUniverseTickers(universeId: string): Promise<UniverseTickerListOut> {
  return apiFetch<UniverseTickerListOut>(
    `/universes/${encodeURIComponent(universeId)}/tickers`,
    { method: 'GET' },
  )
}

export async function getUniverseSummary(universeId: string): Promise<UniverseSummaryOut> {
  return apiFetch<UniverseSummaryOut>(
    `/universes/${encodeURIComponent(universeId)}/summary`,
    { method: 'GET' },
  )
}

export async function getUniverseReturnAnalytics(
  universeId: string,
  params?: {
    period?: string
    interval?: string
    source?: string
    max_members?: number
    n_pca_components?: number
  },
): Promise<UniverseReturnAnalyticsOut> {
  const sp = new URLSearchParams()
  if (params?.period != null) sp.set('period', params.period)
  if (params?.interval != null) sp.set('interval', params.interval)
  if (params?.source != null) sp.set('source', params.source)
  if (params?.max_members != null) sp.set('max_members', String(params.max_members))
  if (params?.n_pca_components != null) sp.set('n_pca_components', String(params.n_pca_components))
  const q = sp.toString()
  return apiFetch<UniverseReturnAnalyticsOut>(
    `/universes/${encodeURIComponent(universeId)}/return-analytics${q ? `?${q}` : ''}`,
    { method: 'GET' },
  )
}

export async function addUniverseMembers(
  universeId: string,
  body: { tickers: string[] },
): Promise<UniverseMembersMutationOut> {
  return apiFetch<UniverseMembersMutationOut>(
    `/universes/${encodeURIComponent(universeId)}/members`,
    {
      method: 'POST',
      body: JSON.stringify({ tickers: body.tickers }),
    },
  )
}

export async function removeUniverseMembers(
  universeId: string,
  tickers: string[],
): Promise<UniverseMembersMutationOut> {
  const sp = new URLSearchParams()
  for (const t of tickers) {
    if (t.trim()) sp.append('tickers', t.trim().toUpperCase())
  }
  const q = sp.toString()
  return apiFetch<UniverseMembersMutationOut>(
    `/universes/${encodeURIComponent(universeId)}/members${q ? `?${q}` : ''}`,
    { method: 'DELETE' },
  )
}

export async function replaceUniverseMembers(
  universeId: string,
  body: { tickers: string[] },
): Promise<UniverseMembersMutationOut> {
  return apiFetch<UniverseMembersMutationOut>(
    `/universes/${encodeURIComponent(universeId)}/members`,
    {
      method: 'PUT',
      body: JSON.stringify({ tickers: body.tickers }),
    },
  )
}

export async function listAlphas(params: {
  limit?: number
  offset?: number
}): Promise<AlphaOut[]> {
  const sp = new URLSearchParams()
  if (params.limit != null) sp.set('limit', String(params.limit))
  if (params.offset != null) sp.set('offset', String(params.offset))
  const q = sp.toString()
  return apiFetch<AlphaOut[]>(`/alphas${q ? `?${q}` : ''}`, { method: 'GET' })
}

export async function getAlpha(alphaId: string): Promise<AlphaOut> {
  return apiFetch<AlphaOut>(`/alphas/${encodeURIComponent(alphaId)}`, {
    method: 'GET',
  })
}

export async function createAlpha(body: AlphaCreate): Promise<AlphaOut> {
  return apiFetch<AlphaOut>('/alphas', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function patchAlpha(
  alphaId: string,
  body: AlphaPatch,
): Promise<AlphaOut> {
  return apiFetch<AlphaOut>(`/alphas/${encodeURIComponent(alphaId)}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  })
}

export async function deleteAlpha(alphaId: string): Promise<void> {
  await apiFetch<void>(`/alphas/${encodeURIComponent(alphaId)}`, {
    method: 'DELETE',
  })
}

export async function postAlphaLintBody(body: {
  source_body: string
}): Promise<AlphaLintBodyResponse> {
  return apiFetch<AlphaLintBodyResponse>('/alphas/lint-body', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function postAlphaAssistBody(body: {
  source_body: string
  alpha_name?: string | null
  alpha_description?: string | null
}): Promise<AlphaAssistBodyResponse> {
  return apiFetch<AlphaAssistBodyResponse>('/alphas/assist-body', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function postAlphaAssistBacktestReview(
  body: AlphaAssistBacktestReviewRequest,
): Promise<AlphaAssistBacktestReviewResponse> {
  return apiFetch<AlphaAssistBacktestReviewResponse>('/alphas/assist-backtest-review', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function listBacktests(params: {
  limit?: number
  offset?: number
  alpha_id?: string | null
  status?: string | null
}): Promise<BacktestJobOut[]> {
  const sp = new URLSearchParams()
  if (params.limit != null) sp.set('limit', String(params.limit))
  if (params.offset != null) sp.set('offset', String(params.offset))
  if (params.alpha_id) sp.set('alpha_id', params.alpha_id)
  if (params.status) sp.set('status', params.status)
  const q = sp.toString()
  return apiFetch<BacktestJobOut[]>(`/backtests${q ? `?${q}` : ''}`, {
    method: 'GET',
  })
}

export async function deleteBacktest(jobId: string): Promise<void> {
  await apiFetch<void>(`/backtests/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
  })
}

export async function deleteBacktestsBatch(
  ids: string[],
): Promise<{ deleted: number }> {
  return apiFetch<{ deleted: number }>('/backtests/delete-batch', {
    method: 'POST',
    body: JSON.stringify({ ids }),
  })
}

export async function getBacktest(jobId: string): Promise<BacktestJobOut> {
  return apiFetch<BacktestJobOut>(`/backtests/${encodeURIComponent(jobId)}`, {
    method: 'GET',
  })
}

export async function getBacktestLogs(jobId: string): Promise<BacktestLogLine[]> {
  try {
    return await apiFetch<BacktestLogLine[]>(
      `/backtests/${encodeURIComponent(jobId)}/logs`,
      { method: 'GET' },
    )
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return []
    throw e
  }
}

export async function getBacktestResult(
  jobId: string,
): Promise<BacktestResultPayload> {
  return apiFetch<BacktestResultPayload>(
    `/backtests/${encodeURIComponent(jobId)}/result`,
    { method: 'GET' },
  )
}

export async function enqueueBacktest(
  body: BacktestCreate,
): Promise<BacktestJobOut> {
  return apiFetch<BacktestJobOut>('/backtests', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function postDataSummary(
  body: DataSummaryRequest,
): Promise<DataSummaryResponse> {
  return apiFetch<DataSummaryResponse>('/data', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function getDataDashboard(params: {
  interval?: string
  source?: string
  bucket?: DashboardBucketParam
  max_buckets?: number
}): Promise<DataDashboardResponse> {
  const sp = new URLSearchParams()
  if (params.interval != null) sp.set('interval', params.interval)
  if (params.source != null) sp.set('source', params.source)
  if (params.bucket != null) sp.set('bucket', params.bucket)
  if (params.max_buckets != null) sp.set('max_buckets', String(params.max_buckets))
  const q = sp.toString()
  return apiFetch<DataDashboardResponse>(`/data/dashboard${q ? `?${q}` : ''}`, {
    method: 'GET',
  })
}

export async function searchInstruments(q: string): Promise<InstrumentSearchResponse> {
  const qs = new URLSearchParams({ q })
  return apiFetch<InstrumentSearchResponse>(`/instruments/search?${qs.toString()}`, {
    method: 'GET',
  })
}

/** Path to instrument detail; optional `quoteType` from search for optimistic UI. */
export function instrumentDetailPath(symbol: string, quoteType?: string | null): string {
  const enc = encodeURIComponent(symbol)
  const t = quoteType?.trim()
  if (t) return `/instruments/${enc}?qt=${encodeURIComponent(t)}`
  return `/instruments/${enc}`
}

export async function getInstrumentOverview(
  symbol: string,
): Promise<InstrumentOverviewResponse> {
  return apiFetch<InstrumentOverviewResponse>(
    `/instruments/${encodeURIComponent(symbol)}/overview`,
    { method: 'GET' },
  )
}

export async function getInstrumentFinancials(
  symbol: string,
  params: {
    statement: InstrumentStatement
    frequency?: InstrumentFinancialFrequency
    periods?: number
  },
): Promise<InstrumentFinancialStatementResponse> {
  const sp = new URLSearchParams({ statement: params.statement })
  if (params.frequency != null) sp.set('frequency', params.frequency)
  if (params.periods != null) sp.set('periods', String(params.periods))
  return apiFetch<InstrumentFinancialStatementResponse>(
    `/instruments/${encodeURIComponent(symbol)}/financials?${sp.toString()}`,
    { method: 'GET' },
  )
}

export async function getInstrumentHolders(
  symbol: string,
): Promise<InstrumentHoldersResponse> {
  return apiFetch<InstrumentHoldersResponse>(
    `/instruments/${encodeURIComponent(symbol)}/holders`,
    { method: 'GET' },
  )
}

export async function getInstrumentOptionExpirations(
  symbol: string,
): Promise<InstrumentOptionExpirationsResponse> {
  return apiFetch<InstrumentOptionExpirationsResponse>(
    `/instruments/${encodeURIComponent(symbol)}/options/expirations`,
    { method: 'GET' },
  )
}

export async function getInstrumentOptionChain(
  symbol: string,
  expiry: string,
): Promise<InstrumentOptionChainResponse> {
  const sp = new URLSearchParams({ expiry })
  return apiFetch<InstrumentOptionChainResponse>(
    `/instruments/${encodeURIComponent(symbol)}/options/chain?${sp.toString()}`,
    { method: 'GET' },
  )
}

export async function getInstrumentOptionIvHeatmap(
  symbol: string,
  params?: { maxExpirations?: number },
): Promise<InstrumentIvHeatmapResponse> {
  const sp = new URLSearchParams()
  if (params?.maxExpirations != null) sp.set('max_expirations', String(params.maxExpirations))
  const q = sp.toString()
  return apiFetch<InstrumentIvHeatmapResponse>(
    `/instruments/${encodeURIComponent(symbol)}/options/iv-heatmap${q ? `?${q}` : ''}`,
    { method: 'GET' },
  )
}

export async function getInstrumentValuationMeasures(
  symbol: string,
): Promise<InstrumentValuationMeasuresPayload> {
  return apiFetch<InstrumentValuationMeasuresPayload>(
    `/instruments/${encodeURIComponent(symbol)}/valuation-measures`,
    { method: 'GET' },
  )
}

export async function getInstrumentAnalystPriceTargets(
  symbol: string,
): Promise<InstrumentAnalystPriceTargetsResponse> {
  return apiFetch<InstrumentAnalystPriceTargetsResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/price-targets`,
    { method: 'GET' },
  )
}

export async function getInstrumentEarningsEstimate(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/earnings-estimate`,
    { method: 'GET' },
  )
}

export async function getInstrumentRevenueEstimate(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/revenue-estimate`,
    { method: 'GET' },
  )
}

export async function getInstrumentEarningsHistory(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/earnings-history`,
    { method: 'GET' },
  )
}

export async function getInstrumentEpsTrend(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/eps-trend`,
    { method: 'GET' },
  )
}

export async function getInstrumentEpsRevisions(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/eps-revisions`,
    { method: 'GET' },
  )
}

export async function getInstrumentGrowthEstimates(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/growth-estimates`,
    { method: 'GET' },
  )
}

export async function getInstrumentRecommendations(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/recommendations`,
    { method: 'GET' },
  )
}

export async function getInstrumentRecommendationsSummary(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/recommendations-summary`,
    { method: 'GET' },
  )
}

export async function getInstrumentUpgradesDowngrades(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/analyst/upgrades-downgrades`,
    { method: 'GET' },
  )
}

export async function getInstrumentSustainability(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/sustainability`,
    { method: 'GET' },
  )
}

export async function getInstrumentInsiderPurchases(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/insider/purchases`,
    { method: 'GET' },
  )
}

export async function getInstrumentInsiderTransactions(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/insider/transactions`,
    { method: 'GET' },
  )
}

export async function getInstrumentInsiderRosterHolders(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/insider/roster-holders`,
    { method: 'GET' },
  )
}

export async function getInstrumentMajorHoldersYf(
  symbol: string,
): Promise<InstrumentYfinanceTableResponse> {
  return apiFetch<InstrumentYfinanceTableResponse>(
    `/instruments/${encodeURIComponent(symbol)}/insider/major-holders`,
    { method: 'GET' },
  )
}

export async function getInstrumentCalendar(
  symbol: string,
): Promise<InstrumentJsonBlobResponse> {
  return apiFetch<InstrumentJsonBlobResponse>(
    `/instruments/${encodeURIComponent(symbol)}/calendar`,
    { method: 'GET' },
  )
}

export async function getInstrumentSecFilings(
  symbol: string,
): Promise<InstrumentJsonBlobResponse> {
  return apiFetch<InstrumentJsonBlobResponse>(
    `/instruments/${encodeURIComponent(symbol)}/sec-filings`,
    { method: 'GET' },
  )
}

export async function getInstrumentNews(
  symbol: string,
  params?: { limit?: number },
): Promise<InstrumentTickerNewsResponse> {
  const sp = new URLSearchParams()
  if (params?.limit != null) sp.set('limit', String(params.limit))
  const q = sp.toString()
  return apiFetch<InstrumentTickerNewsResponse>(
    `/instruments/${encodeURIComponent(symbol)}/news${q ? `?${q}` : ''}`,
    { method: 'GET' },
  )
}

export async function getInstrumentOhlcv(
  symbol: string,
  params: { interval: string; period: string; route?: string },
): Promise<InstrumentOhlcvResponse> {
  const qs = new URLSearchParams({
    interval: params.interval,
    period: params.period,
  })
  if (params.route) {
    qs.set('route', params.route)
  }
  return apiFetch<InstrumentOhlcvResponse>(
    `/instruments/${encodeURIComponent(symbol)}/ohlcv?${qs.toString()}`,
    { method: 'GET' },
  )
}
