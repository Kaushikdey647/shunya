import { describe, expect, it } from 'vitest'
import {
  adaptEquityCurve,
  assertPlottableEquitySeries,
  assertPlottablePerformanceOverlay,
  buildPerformanceOverlay,
  drawdownPercentYDomain,
  indexedPerformanceYDomain,
  padEpochMsDomain,
} from './backtestCharts'

describe('padEpochMsDomain', () => {
  it('widens epoch-ms range and keeps finiteness', () => {
    const d: [number, number] = [1_000_000, 2_000_000]
    const [a, b] = padEpochMsDomain(d, 0.01)
    expect(a).toBeLessThan(d[0])
    expect(b).toBeGreaterThan(d[1])
    expect(Number.isFinite(a) && Number.isFinite(b)).toBe(true)
  })

  it('returns original when span invalid', () => {
    expect(padEpochMsDomain([NaN, 1] as unknown as [number, number])).toEqual([NaN, 1])
  })
})

describe('adaptEquityCurve', () => {
  it('skips legacy NaT string equity values', () => {
    const rows = [
      { Date: '2020-01-01', Equity: 100, Peak: 100 },
      { Date: '2020-01-02', Equity: 'NaT', Peak: 100 },
      { Date: '2020-01-03', Equity: 90, Peak: 100 },
    ]
    const pts = adaptEquityCurve(rows as Record<string, unknown>[])
    expect(pts).toHaveLength(2)
    expect(pts[1]!.equity).toBe(90)
  })

  it('derives drawdown from peak when DrawdownPct missing', () => {
    const rows = [
      { Date: '2020-01-01', Equity: 100, Peak: 100 },
      { Date: '2020-01-02', Equity: 90, Peak: 100 },
    ]
    const pts = adaptEquityCurve(rows as Record<string, unknown>[])
    expect(pts).toHaveLength(2)
    expect(pts[1]!.drawdownPct).toBeCloseTo(-10, 5)
  })

  it('re-derives drawdown when DrawdownPct is absurdly large', () => {
    const rows = [
      { Date: '2020-01-01', Equity: 100_000, Peak: 100_000, DrawdownPct: 0 },
      { Date: '2020-01-02', Equity: 90_000, Peak: 100_000, DrawdownPct: 35_110_587 },
    ]
    const pts = adaptEquityCurve(rows as Record<string, unknown>[])
    expect(pts[1]!.drawdownPct).toBeCloseTo(-10, 5)
  })
})

describe('assertPlottableEquitySeries', () => {
  it('rejects empty', () => {
    expect(assertPlottableEquitySeries([]).ok).toBe(false)
  })

  it('accepts finite series', () => {
    expect(
      assertPlottableEquitySeries([
        { t: 0, equity: 1, peak: 1, drawdownPct: 0 },
        { t: 1, equity: 2, peak: 2, drawdownPct: 0 },
      ]).ok,
    ).toBe(true)
  })
})

describe('buildPerformanceOverlay + assertPlottablePerformanceOverlay', () => {
  it('returns null without benchmark curve', () => {
    expect(buildPerformanceOverlay([], null)).toBeNull()
    expect(buildPerformanceOverlay([], {})).toBeNull()
  })

  it('returns null when benchmark has error string', () => {
    expect(
      buildPerformanceOverlay(
        [{ Date: '2020-01-01', Equity: 100 }],
        { error: 'fail', benchmark_equity_curve: [{ Date: '2020-01-01', Equity: 100 }] },
      ),
    ).toBeNull()
  })

  it('builds overlay when curves align and overlay is plottable', () => {
    const strat = [
      { Date: '2020-01-01', Equity: 100 },
      { Date: '2020-01-02', Equity: 110 },
    ]
    const bench = [
      { Date: '2020-01-01', Equity: 200 },
      { Date: '2020-01-02', Equity: 220 },
    ]
    const overlay = buildPerformanceOverlay(strat as Record<string, unknown>[], {
      benchmark_equity_curve: bench,
    })
    expect(overlay).not.toBeNull()
    expect(overlay!.length).toBe(2)
    const g = assertPlottablePerformanceOverlay(overlay!)
    expect(g.ok).toBe(true)
  })
})

describe('indexedPerformanceYDomain', () => {
  it('spans both strategy and benchmark indexed series', () => {
    const d = indexedPerformanceYDomain([
      { t: 0, equity: 1, benchEquity: 1, equityIdx: 100, benchIdx: 100 },
      { t: 1, equity: 1, benchEquity: 1, equityIdx: 110, benchIdx: 95 },
    ])
    expect(d).not.toBeNull()
    expect(d![0]).toBeLessThan(95)
    expect(d![1]).toBeGreaterThan(110)
  })
})

describe('drawdownPercentYDomain', () => {
  it('pads negative underwater range', () => {
    const d = drawdownPercentYDomain([
      { t: 0, equity: 1, peak: 1, drawdownPct: 0 },
      { t: 1, equity: 1, peak: 2, drawdownPct: -10 },
    ])
    expect(d).not.toBeNull()
    expect(d![0]).toBeLessThanOrEqual(-10)
    expect(d![1]).toBeGreaterThanOrEqual(0)
  })
})
