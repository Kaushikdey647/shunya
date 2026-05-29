/**
 * Monthly portfolio returns from equity curve rows (for heatmap).
 * Uses calendar month buckets in UTC from ISO date strings.
 */

export type MonthlyReturnCell = {
  year: number
  month: number /** 1–12 */
  /** Percent return for that month (not annualized) */
  monthReturnPct: number
}

export type MonthlyHeatmapGrid = {
  years: number[]
  months: readonly number[]
  /** Map key `${year}-${month}` → monthReturnPct */
  values: Map<string, number>
  /** Max absolute return for color scaling */
  normAbs: number
}

function parseRowTime(row: Record<string, unknown>): number | null {
  const raw = row.Date ?? row.date
  if (typeof raw !== 'string') return null
  const ms = Date.parse(raw)
  return Number.isFinite(ms) ? ms : null
}

function num(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string') {
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  }
  return null
}

/**
 * End-of-month equity → simple monthly return chain.
 */
export function monthlyReturnsFromEquityCurve(rows: Record<string, unknown>[]): MonthlyHeatmapGrid {
  type Pt = { t: number; equity: number }
  const pts: Pt[] = []
  for (const row of rows) {
    const t = parseRowTime(row)
    const equity = num(row.Equity ?? row.equity)
    if (t == null || equity == null || equity <= 0) continue
    pts.push({ t, equity })
  }
  pts.sort((a, b) => a.t - b.t)

  const monthLast = new Map<string, { t: number; equity: number }>()
  for (const p of pts) {
    const d = new Date(p.t)
    const y = d.getUTCFullYear()
    const m = d.getUTCMonth() + 1
    const key = `${y}-${m}`
    const prev = monthLast.get(key)
    if (!prev || p.t >= prev.t) {
      monthLast.set(key, { t: p.t, equity: p.equity })
    }
  }

  const keysSorted = [...monthLast.keys()].sort((a, b) => {
    const [ya, ma] = a.split('-').map(Number)
    const [yb, mb] = b.split('-').map(Number)
    if (ya !== yb) return ya! - yb!
    return ma! - mb!
  })

  const values = new Map<string, number>()
  let prevEquity: number | null = null
  for (const key of keysSorted) {
    const e = monthLast.get(key)!.equity
    if (prevEquity != null && prevEquity > 0) {
      const r = ((e / prevEquity - 1) * 100)
      values.set(key, r)
    }
    prevEquity = e
  }

  const yearsSet = new Set<number>()
  for (const key of values.keys()) {
    yearsSet.add(Number(key.split('-')[0]))
  }
  const years = [...yearsSet].sort((a, b) => a - b)
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const

  let minPct = 0
  let maxPct = 0
  for (const v of values.values()) {
    minPct = Math.min(minPct, v)
    maxPct = Math.max(maxPct, v)
  }
  const normAbs = Math.max(Math.abs(minPct), Math.abs(maxPct), 1e-6)

  return { years, months, values, normAbs }
}

export type HeatmapPalette = {
  success: string
  error: string
  surface: string
  empty: string
}

export function heatmapCellColor(
  pct: number | undefined,
  normAbs: number,
  palette?: HeatmapPalette,
): string {
  const surface = palette?.surface ?? 'var(--surface-panel)'
  const empty = palette?.empty ?? 'var(--surface-hover)'
  const success = palette?.success ?? 'var(--success)'
  const error = palette?.error ?? 'var(--error)'
  if (pct === undefined || Number.isNaN(pct)) return empty
  const t = normAbs > 0 ? Math.max(-1, Math.min(1, pct / normAbs)) : 0
  if (t >= 0) {
    const pctMix = Math.round((22 + Math.abs(t) * 58) * 100) / 100
    return `color-mix(in srgb, ${success} ${pctMix}%, ${surface})`
  }
  const pctMix = Math.round((22 + Math.abs(t) * 58) * 100) / 100
  return `color-mix(in srgb, ${error} ${pctMix}%, ${surface})`
}
