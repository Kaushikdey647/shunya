/**
 * User-facing titles for API error codes (mirror shunya.errors.ErrorCode).
 * Unknown codes still show the server message via ApiError.
 */
export const ERROR_CATALOG: Record<string, string> = {
  FIN_TS_TIMESCALE_DEPENDENCY: 'Timescale data provider is not installed',
  FIN_TS_TIMESCALE_DSN_REQUIRED: 'Database URL required for Timescale',
  FIN_TS_TIMESCALE_UNAVAILABLE: 'Timescale market data unavailable',
  FIN_TS_FUNDAMENTALS_DSN_REQUIRED: 'Database URL required for fundamentals',
  ALPHA_NOT_FOUND: 'Alpha not found',
  ALPHA_NAME_CONFLICT: 'That alpha name is already taken',
  BACKTEST_JOB_NOT_FOUND: 'Backtest job not found',
  BACKTEST_RESULT_MISSING: 'Backtest result missing',
  BACKTEST_RESULT_NOT_READY: 'Backtest not finished yet',
  DELETE_BATCH_TOO_LARGE: 'Too many jobs in one delete request',
  INVALID_STATUS_FILTER: 'Invalid status filter',
  BACKTEST_JOB_EXECUTION_ERROR: 'Backtest run failed',
  BACKTEST_JOB_SERVER_RESTART: 'Job interrupted by server restart',
  BACKTEST_INDEX_UNKNOWN: 'Unknown index code',
  BACKTEST_INDEX_NOT_FOUND: 'Index not found',
  BACKTEST_INDEX_NO_MEMBERS: 'Index has no members',
  BACKTEST_INDEX_OHLCV: 'Index or benchmark OHLCV issue',
  DATA_INVALID_INTERVAL: 'Invalid dashboard interval',
  DATA_INVALID_SOURCE: 'Invalid data source',
  DATA_DASHBOARD_FAILED: 'Dashboard computation failed',
  VALIDATION_ERROR: 'Invalid request',
  INTERNAL_ERROR: 'Unexpected error',
}

export function titleForErrorCode(code: string | undefined): string | undefined {
  if (!code) return undefined
  return ERROR_CATALOG[code]
}
