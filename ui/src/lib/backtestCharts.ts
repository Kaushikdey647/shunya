/** Normalize API equity / turnover row shapes for Recharts. */

export type EquityChartRow = {
  t: number
  equity: number
  peak: number
  drawdownPct: number
}

/** Strategy + benchmark for performance chart (indexed to 100 at each series’ first valid bar). */
export type PerformanceOverlayRow = {
  t: number
  equity: number
  benchEquity: number | null
  equityIdx: number
  benchIdx: number | null
}

export type TurnoverChartRow = {
  t: number
  turnoverUsd: number
  turnoverPct: number | null
}

export type TargetStackRow = Record<string, number | null> & { t: number }

export type HistogramBin = {
  /** Bin center for numeric X axis (readable ticks). */
  mid: number
  label: string
  count: number
  x0: number
  x1: number
}

export type RollingSharpeRow = { t: number; sharpe: number | null }

export type ExposureChartRow = {
  t: number
  grossLeverage: number
  longExposure: number
  shortExposure: number
}

export type TurnoverPctRow = { t: number; turnoverPct: number | null }

export type BacktestTargetRow = { date: string; targets: Record<string, unknown> }

function num(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string') {
    const t = v.trim()
    if (t === '' || t.toLowerCase() === 'nan' || t === 'NaT') return null
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}

/** Slightly widen epoch-ms domain so Recharts numeric X scale does not clip series at edges. */
export function padEpochMsDomain(domain: [number, number], padRatio = 0.002): [number, number] {
  const [a, b] = domain
  if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return domain
  const span = b - a
  const pad = Math.max(span * padRatio, 1)
  return [a - pad, b + pad]
}

function rowTimeMs(row: Record<string, unknown>): number | null {
  const raw = row.Date ?? row.date
  if (typeof raw !== 'string') return null
  const ms = Date.parse(raw)
  return Number.isFinite(ms) ? ms : null
}

export function adaptEquityCurve(rows: Record<string, unknown>[]): EquityChartRow[] {
  const out: EquityChartRow[] = []
  for (const row of rows) {
    const t = rowTimeMs(row)
    const equity = num(row.Equity ?? row.equity)
    if (t == null || equity == null) continue
    const peak = num(row.Peak ?? row.peak)
    const peakN = peak ?? equity
    let dd = num(row.DrawdownPct ?? row.drawdownPct ?? row.drawdown_pct)
    if (dd == null && peakN != null && peakN > 0 && equity != null) {
      dd = (equity / peakN - 1) * 100
    }
    let drawdownPct = dd ?? 0
    if (!Number.isFinite(drawdownPct)) drawdownPct = 0
    // Corrupt / mis-typed payloads (e.g. wrong column merged) can produce absurd magnitudes and
    // break Recharts Y scales; re-derive from peak/equity when clearly impossible for a % DD.
    if (Math.abs(drawdownPct) > 10_000 && peakN > 0 && equity != null) {
      drawdownPct = (equity / peakN - 1) * 100
    }
    if (!Number.isFinite(drawdownPct) || Math.abs(drawdownPct) > 50_000) {
      drawdownPct = peakN > 0 && equity != null ? (equity / peakN - 1) * 100 : 0
    }
    drawdownPct = Math.max(-1000, Math.min(1000, drawdownPct))
    out.push({
      t,
      equity,
      peak: peakN,
      drawdownPct,
    })
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

/**
 * Merge strategy equity with optional API `benchmark.benchmark_equity_curve` for overlay charting.
 * Returns null if benchmark missing, errored, or has no usable curve.
 */
export function buildPerformanceOverlay(
  strategyEquityRows: Record<string, unknown>[],
  benchmark: Record<string, unknown> | null | undefined,
): PerformanceOverlayRow[] | null {
  if (!benchmark || typeof benchmark !== 'object') return null
  const err = benchmark.error
  if (err != null && String(err).length > 0) return null
  const raw = benchmark.benchmark_equity_curve
  if (!Array.isArray(raw) || raw.length === 0) return null
  const strat = adaptEquityCurve(strategyEquityRows)
  const bench = adaptEquityCurve(raw as Record<string, unknown>[])
  if (strat.length === 0) return null
  const benchByT = new Map<number, number>()
  for (const r of bench) benchByT.set(r.t, r.equity)
  let firstBenchBase: number | null = null
  for (const r of strat) {
    const be = benchByT.get(r.t)
    if (be != null && be > 0) {
      firstBenchBase = be
      break
    }
  }
  if (firstBenchBase == null) return null
  const firstEq = strat[0]!.equity
  if (firstEq === 0) return null
  const out: PerformanceOverlayRow[] = []
  let anyBench = false
  for (const r of strat) {
    const be = benchByT.get(r.t) ?? null
    if (be != null) anyBench = true
    const equityIdx = (100 * r.equity) / firstEq
    const benchIdx =
      be != null && be > 0 ? (100 * be) / firstBenchBase : null
    out.push({ t: r.t, equity: r.equity, benchEquity: be, equityIdx, benchIdx })
  }
  return anyBench ? out : null
}

/** Y domain for indexed performance (strategy + benchmark on one axis). */
export function indexedPerformanceYDomain(
  rows: PerformanceOverlayRow[],
): [number, number] | null {
  if (rows.length === 0) return null
  let mn = Infinity
  let mx = -Infinity
  for (const r of rows) {
    if (Number.isFinite(r.equityIdx)) {
      mn = Math.min(mn, r.equityIdx)
      mx = Math.max(mx, r.equityIdx)
    }
    if (r.benchIdx != null && Number.isFinite(r.benchIdx)) {
      mn = Math.min(mn, r.benchIdx)
      mx = Math.max(mx, r.benchIdx)
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx) || mn === Infinity || mx === -Infinity) {
    return null
  }
  if (!(mn < mx)) return [mn - 1, mx + 1]
  const pad = Math.max((mx - mn) * 0.05, 0.25)
  return [mn - pad, mx + pad]
}

/** Y domain for raw equity line (no benchmark overlay). */
export function equityLineYDomain(rows: EquityChartRow[]): [number, number] | null {
  if (rows.length === 0) return null
  let mn = Infinity
  let mx = -Infinity
  for (const r of rows) {
    if (Number.isFinite(r.equity)) {
      mn = Math.min(mn, r.equity)
      mx = Math.max(mx, r.equity)
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return null
  if (!(mn < mx)) {
    const pad = Math.max(Math.abs(mn) * 1e-6, 1)
    return [mn - pad, mx + pad]
  }
  const pad = Math.max((mx - mn) * 0.04, 1)
  return [mn - pad, mx + pad]
}

/** Y domain for underwater drawdown % (≤ 0 at the top of the natural range for long-only curves). */
export function drawdownPercentYDomain(rows: EquityChartRow[]): [number, number] | null {
  if (rows.length === 0) return null
  let lo = 0
  let hi = 0
  for (const p of rows) {
    const v = p.drawdownPct
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
  }
  const padLo = Math.max(0.5, Math.abs(lo) * 0.02)
  const padHi = Math.max(0.25, hi * 0.02)
  return [lo - padLo, hi + padHi]
}

export function assertPlottableEquitySeries(
  rows: EquityChartRow[],
): { ok: true } | { ok: false; reason: string } {
  if (rows.length === 0) return { ok: false, reason: 'no_points' }
  for (const r of rows) {
    if (!Number.isFinite(r.t) || !Number.isFinite(r.equity)) {
      return { ok: false, reason: 'non_finite_coordinate' }
    }
  }
  return { ok: true }
}

export function assertPlottablePerformanceOverlay(
  rows: PerformanceOverlayRow[],
): { ok: true } | { ok: false; reason: string } {
  if (rows.length === 0) return { ok: false, reason: 'no_points' }
  for (const r of rows) {
    if (!Number.isFinite(r.t) || !Number.isFinite(r.equityIdx)) {
      return { ok: false, reason: 'non_finite_overlay' }
    }
  }
  return { ok: true }
}

export function assertPlottableDrawdownSeries(
  rows: EquityChartRow[],
): { ok: true } | { ok: false; reason: string } {
  if (rows.length === 0) return { ok: false, reason: 'no_points' }
  for (const r of rows) {
    if (!Number.isFinite(r.t) || !Number.isFinite(r.drawdownPct)) {
      return { ok: false, reason: 'non_finite_drawdown' }
    }
  }
  return { ok: true }
}

export function adaptExposureHistory(rows: Record<string, unknown>[]): ExposureChartRow[] {
  const out: ExposureChartRow[] = []
  for (const row of rows) {
    const t = rowTimeMs(row)
    const gl = num(row.GrossLeverage ?? row.grossLeverage ?? row.gross_leverage)
    const le = num(row.LongExposure ?? row.longExposure ?? row.long_exposure)
    const se = num(row.ShortExposure ?? row.shortExposure ?? row.short_exposure)
    if (t == null || gl == null || le == null || se == null) continue
    out.push({
      t,
      grossLeverage: gl,
      longExposure: le,
      shortExposure: se,
    })
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

export function adaptTurnoverPctHistory(rows: Record<string, unknown>[]): TurnoverPctRow[] {
  const out: TurnoverPctRow[] = []
  for (const row of rows) {
    const t = rowTimeMs(row)
    const rawPct = num(row.TurnoverPct ?? row.turnoverPct ?? row.turnover_pct)
    if (t == null) continue
    out.push({ t, turnoverPct: rawPct != null ? rawPct * 100 : null })
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

export function adaptTurnoverHistory(
  turnoverRows: Record<string, unknown>[],
  equityRows: Record<string, unknown>[],
): TurnoverChartRow[] {
  const eq = adaptEquityCurve(equityRows)
  const raw: { t: number; usd: number }[] = []
  for (const row of turnoverRows) {
    const t = rowTimeMs(row)
    const usd = num(row.TurnoverUSD ?? row.turnoverUSD)
    if (t == null || usd == null) continue
    raw.push({ t, usd: usd })
  }
  raw.sort((a, b) => a.t - b.t)

  let eqPtr = 0
  let lastEquity: number | null = null
  const out: TurnoverChartRow[] = []
  for (const to of raw) {
    while (eqPtr < eq.length && eq[eqPtr].t <= to.t) {
      lastEquity = eq[eqPtr].equity
      eqPtr++
    }
    const pct =
      lastEquity != null && lastEquity !== 0 ? (to.usd / lastEquity) * 100 : null
    out.push({ t: to.t, turnoverUsd: to.usd, turnoverPct: pct })
  }
  return out
}

export function formatMetricNumber(v: unknown, digits = 2): string {
  const n = num(v)
  if (n == null) return '—'
  return n.toFixed(digits)
}

/** Compact axis tick for notionals / equity-like magnitudes. */
export function formatChartAxisCompact(v: number): string {
  if (!Number.isFinite(v)) return ''
  const a = Math.abs(v)
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (a >= 1e3) return `${(v / 1e3).toFixed(1)}k`
  if (a >= 1) return v.toFixed(v % 1 === 0 ? 0 : 2)
  return v.toPrecision(2)
}

/** Approximate periods per year from FinBT metrics (matches server logic). */
export function periodsPerYearFromMetrics(metrics: Record<string, unknown>): number {
  const unit = String(metrics.bar_unit ?? 'DAYS').toUpperCase()
  const step = Math.max(1, Number(metrics.bar_step) || 1)
  switch (unit) {
    case 'SECONDS':
      return (252 * 6.5 * 60 * 60) / step
    case 'MINUTES':
      return (252 * 6.5 * 60) / step
    case 'HOURS':
      return (252 * 6.5) / step
    case 'DAYS':
      return 252 / step
    case 'WEEKS':
      return 52 / step
    case 'MONTHS':
      return 12 / step
    default:
      return 252 / step
  }
}

export function equityBarReturns(equityPts: EquityChartRow[]): number[] {
  const out: number[] = []
  for (let i = 1; i < equityPts.length; i++) {
    const a = equityPts[i - 1]!.equity
    const b = equityPts[i]!.equity
    if (a > 0 && Number.isFinite(b)) out.push(b / a - 1)
  }
  return out
}

export function returnHistogramBins(
  returns: number[],
  binCount = 24,
): HistogramBin[] {
  if (returns.length === 0) return []
  let lo = Math.min(...returns)
  let hi = Math.max(...returns)
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.05 + 1e-8
    lo -= pad
    hi += pad
  }
  const w = (hi - lo) / binCount
  const counts = new Array(binCount).fill(0) as number[]
  for (const r of returns) {
    let i = Math.floor((r - lo) / w)
    if (i < 0) i = 0
    if (i >= binCount) i = binCount - 1
    counts[i]++
  }
  return counts.map((c, i) => {
    const x0 = lo + i * w
    const x1 = lo + (i + 1) * w
    const mid = (x0 + x1) / 2
    return {
      mid,
      label: `${x0.toPrecision(3)}…${x1.toPrecision(3)}`,
      count: c,
      x0,
      x1,
    }
  })
}

export function rollingSharpeFromEquity(
  equityPts: EquityChartRow[],
  window: number,
  periodsPerYear: number,
): RollingSharpeRow[] {
  const rets = equityBarReturns(equityPts)
  if (rets.length < window + 1) return []
  const out: RollingSharpeRow[] = []
  for (let i = window - 1; i < rets.length; i++) {
    const slice = rets.slice(i - window + 1, i + 1)
    const mean = slice.reduce((s, x) => s + x, 0) / slice.length
    let varSum = 0
    for (const x of slice) varSum += (x - mean) ** 2
    const sd = Math.sqrt(varSum / Math.max(1, slice.length - 1))
    const t = equityPts[i + 1]!.t
    const sharpe =
      sd > 1e-12 && Number.isFinite(mean) ? (Math.sqrt(periodsPerYear) * mean) / sd : null
    out.push({ t, sharpe })
  }
  return out
}

function parseTargetRow(row: unknown): { t: number; w: Map<string, number> } | null {
  if (!row || typeof row !== 'object') return null
  const r = row as Record<string, unknown>
  const rawD = r.date ?? r.Date
  if (typeof rawD !== 'string') return null
  const t = Date.parse(rawD)
  if (!Number.isFinite(t)) return null
  const tg = r.targets
  if (!tg || typeof tg !== 'object') return null
  const w = new Map<string, number>()
  for (const [k, v] of Object.entries(tg as Record<string, unknown>)) {
    const n = num(v)
    if (n != null) w.set(k, n)
  }
  return { t, w }
}

/** Top-K names by mean absolute weight; rest rolled into "Other". */
export function adaptTargetHistoryStacked(
  rows: unknown[],
  topK = 10,
): { keys: string[]; series: TargetStackRow[] } {
  const parsed: { t: number; w: Map<string, number> }[] = []
  for (const row of rows) {
    const p = parseTargetRow(row)
    if (p) parsed.push(p)
  }
  parsed.sort((a, b) => a.t - b.t)
  if (parsed.length === 0) return { keys: [], series: [] }

  const meanAbs = new Map<string, number>()
  for (const { w } of parsed) {
    for (const [sym, v] of w) {
      meanAbs.set(sym, (meanAbs.get(sym) ?? 0) + Math.abs(v))
    }
  }
  for (const s of meanAbs.keys()) {
    meanAbs.set(s, (meanAbs.get(s) ?? 0) / parsed.length)
  }
  const ranked = [...meanAbs.entries()].sort((a, b) => b[1] - a[1])
  const top = ranked.slice(0, topK).map(([s]) => s)
  const topSet = new Set(top)

  const series: TargetStackRow[] = []
  for (const { t, w } of parsed) {
    const row: TargetStackRow = { t }
    let other = 0
    for (const [sym, v] of w) {
      if (topSet.has(sym)) row[sym] = v
      else other += v
    }
    row.Other = other
    series.push(row)
  }

  return { keys: [...top, 'Other'], series }
}

export function targetHistoryConcentration(rows: unknown[]): { t: number; hhi: number; maxAbs: number }[] {
  const out: { t: number; hhi: number; maxAbs: number }[] = []
  for (const row of rows) {
    const p = parseTargetRow(row)
    if (!p) continue
    let hhi = 0
    let maxAbs = 0
    for (const v of p.w.values()) {
      hhi += v * v
      maxAbs = Math.max(maxAbs, Math.abs(v))
    }
    out.push({ t: p.t, hhi, maxAbs })
  }
  out.sort((a, b) => a.t - b.t)
  return out
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return x != null && typeof x === 'object' && !Array.isArray(x)
}

/** Defensive read of backtrader DrawDown analyzer JSON. */
export function summarizeDrawdownAnalysis(dd: unknown): {
  maxLen?: number | null
  maxDrawdownFrac?: number | null
  maxMoneyDown?: number | null
} {
  if (!isRecord(dd)) return {}
  const max = dd.max
  if (!isRecord(max)) return {}
  return {
    maxLen: num(max.len) != null ? Math.round(num(max.len)!) : null,
    maxDrawdownFrac: num(max.drawdown),
    maxMoneyDown: num(max.moneydown),
  }
}

export function summarizeReturnsAnalysis(ret: unknown): {
  rtot?: number | null
  ravg?: number | null
} {
  if (!isRecord(ret)) return {}
  return {
    rtot: num(ret.rtot),
    ravg: num(ret.ravg),
  }
}

export function summarizeSharpeAnalysis(sh: unknown): { sharperatio?: number | null } {
  if (!isRecord(sh)) return {}
  return { sharperatio: num(sh.sharperatio) }
}
