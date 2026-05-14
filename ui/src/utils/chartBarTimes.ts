import type { Time } from 'lightweight-charts'
import type { OhlcvBar } from '../api/types'

export function barTimesUtcSeconds(bars: OhlcvBar[]): number[] {
  return bars
    .map((b) => Math.floor(new Date(b.time).getTime() / 1000))
    .sort((a, b) => a - b)
}

/** Marker times must match a bar time; use latest bar at or before the headline time. */
export function snapNewsToBarTime(newsSec: number, barTimes: number[]): number {
  const first = barTimes[0]!
  const last = barTimes[barTimes.length - 1]!
  if (newsSec <= first) return first
  if (newsSec >= last) return last
  let lo = 0
  let hi = barTimes.length - 1
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (barTimes[mid]! <= newsSec) lo = mid
    else hi = mid - 1
  }
  return barTimes[lo]!
}

export function timeToUnixSeconds(t: Time | undefined): number | null {
  if (t === undefined || t === null) return null
  if (typeof t === 'number') return t
  if (typeof t === 'string') {
    const ms = Date.parse(t)
    return Number.isFinite(ms) ? Math.floor(ms / 1000) : null
  }
  const o = t as { year: number; month: number; day: number }
  if (typeof o === 'object' && o != null && 'year' in o && 'month' in o && 'day' in o) {
    return Math.floor(Date.UTC(o.year, o.month - 1, o.day) / 1000)
  }
  return null
}

/** Snap arbitrary unix (from crosshair) to nearest bar open time that exists in series. */
export function snapCrosshairToBarTime(targetSec: number | null, barTimes: number[]): number | null {
  if (targetSec == null || barTimes.length === 0) return null
  return snapNewsToBarTime(targetSec, barTimes)
}
