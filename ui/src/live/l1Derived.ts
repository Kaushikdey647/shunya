import type { AlpacaL1WsQuote, AlpacaL1WsTrade } from '../api/types'

/** Runtime guard: WebSocket JSON may omit or null-coerce numbers; charts require finite prices/sizes. */
export function isUsableAlpacaL1Quote(msg: AlpacaL1WsQuote): boolean {
  return (
    msg.type === 'quote' &&
    typeof msg.time === 'string' &&
    msg.time.length > 0 &&
    Number.isFinite(msg.bid_price) &&
    Number.isFinite(msg.ask_price) &&
    Number.isFinite(msg.bid_size) &&
    Number.isFinite(msg.ask_size)
  )
}

export function isUsableAlpacaL1Trade(msg: AlpacaL1WsTrade): boolean {
  return (
    msg.type === 'trade' &&
    typeof msg.time === 'string' &&
    msg.time.length > 0 &&
    Number.isFinite(msg.price) &&
    Number.isFinite(msg.size)
  )
}

/**
 * **OFI / imbalance (this tab only)** — not a canonical exchange order-flow imbalance.
 *
 * For each **BBO quote** we take the log-space size skew:
 * `log(1 + bid_size) − log(1 + ask_size)`.
 * Positive values mean relatively more size on the bid; negative means more on the ask.
 * The histogram buckets the **last** `window` samples into `numBins` equal-width bins
 * between `minV` and `maxV` (values outside are counted in the edge bins).
 */
export function quoteLogSizeImbalance(q: AlpacaL1WsQuote): number {
  const b = Math.max(0, q.bid_size)
  const a = Math.max(0, q.ask_size)
  return Math.log(1 + b) - Math.log(1 + a)
}

export type OfiHistogramBin = { label: string; count: number }

export function buildOfiHistogramFromQuotes(
  quotes: AlpacaL1WsQuote[],
  opts?: { window?: number; numBins?: number; range?: { min: number; max: number } },
): OfiHistogramBin[] {
  const window = opts?.window ?? 120
  const numBins = opts?.numBins ?? 10
  const slice = quotes.slice(-window).filter(isUsableAlpacaL1Quote)
  const values = slice.map(quoteLogSizeImbalance)
  if (values.length === 0) {
    return Array.from({ length: numBins }, (_, i) => ({
      label: String(i + 1),
      count: 0,
    }))
  }
  let minV = opts?.range?.min
  let maxV = opts?.range?.max
  if (minV === undefined || maxV === undefined) {
    minV = Math.min(...values)
    maxV = Math.max(...values)
    if (minV === maxV) {
      minV -= 0.5
      maxV += 0.5
    }
  }
  const span = maxV - minV || 1
  const counts = new Array<number>(numBins).fill(0)
  for (const v of values) {
    let idx = Math.floor(((v - minV) / span) * numBins)
    if (idx < 0) idx = 0
    if (idx >= numBins) idx = numBins - 1
    counts[idx] += 1
  }
  const labels: string[] = []
  for (let i = 0; i < numBins; i += 1) {
    const lo = minV + (span * i) / numBins
    const hi = minV + (span * (i + 1)) / numBins
    labels.push(`${lo.toFixed(2)}…${hi.toFixed(2)}`)
  }
  return counts.map((count, i) => ({ label: labels[i] ?? String(i), count }))
}

export function isoToUnixSec(iso: string): number {
  const ms = Date.parse(iso)
  return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0
}

export type LwcLinePoint = { time: number; value: number }

/**
 * Mid/spread line data for lightweight-charts.
 *
 * `isoToUnixSec` truncates to **whole seconds**; Alpaca can emit many BBO updates per second.
 * Line series `setData` requires **strictly increasing, unique** `time` values — duplicates
 * throw internally (`Value is null`). We keep the **last** quote per UTC second (stepped
 * mid/spread is still correct between seconds).
 */
export function buildL1MidSpreadLinePoints(quotes: AlpacaL1WsQuote[]): {
  mid: LwcLinePoint[]
  spread: LwcLinePoint[]
} {
  const bySec = new Map<number, { mid: number; spread: number }>()
  for (const q of quotes) {
    if (!isUsableAlpacaL1Quote(q)) continue
    const sec = isoToUnixSec(q.time)
    if (!Number.isFinite(sec) || sec <= 0) continue
    const mid = (q.bid_price + q.ask_price) / 2
    const spread = Math.max(0, q.ask_price - q.bid_price)
    bySec.set(sec, { mid, spread })
  }
  const sorted = Array.from(bySec.entries()).sort((a, b) => a[0] - b[0])
  return {
    mid: sorted.map(([t, v]) => ({ time: t, value: v.mid })),
    spread: sorted.map(([t, v]) => ({ time: t, value: v.spread })),
  }
}
