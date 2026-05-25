import { describe, expect, it } from 'vitest'
import type { AlpacaL1WsQuote } from '../api/types'
import { buildL1MidSpreadLinePoints, isUsableAlpacaL1Quote } from './l1Derived'

function q(
  time: string,
  bid: number,
  ask: number,
  bidSize = 1,
  askSize = 1,
): AlpacaL1WsQuote {
  return {
    type: 'quote',
    symbol: 'AAPL',
    time,
    bid_price: bid,
    ask_price: ask,
    bid_size: bidSize,
    ask_size: askSize,
  }
}

describe('buildL1MidSpreadLinePoints', () => {
  it('dedupes multiple quotes in the same UTC second (last wins)', () => {
    const t = '2026-05-22T19:48:17.100Z'
    const t2 = '2026-05-22T19:48:17.900Z'
    const t3 = '2026-05-22T19:48:18.050Z'
    const quotes = [q(t, 100, 101), q(t2, 102, 103), q(t3, 104, 105)]
    const { mid, spread, bid, ask } = buildL1MidSpreadLinePoints(quotes)
    expect(mid).toHaveLength(2)
    expect(spread).toHaveLength(2)
    expect(bid).toHaveLength(2)
    expect(ask).toHaveLength(2)
    expect(mid[0]!.time).toBe(mid[1]!.time - 1)
    expect(mid[1]!.value).toBe((104 + 105) / 2)
    expect(spread[1]!.value).toBe(1)
    expect(bid[1]!.value).toBe(104)
    expect(ask[1]!.value).toBe(105)
  })

  it('sorts by time when quotes arrive out of order', () => {
    const quotes = [
      q('2026-05-22T19:48:20.000Z', 10, 11),
      q('2026-05-22T19:48:18.000Z', 12, 13),
    ]
    const { mid } = buildL1MidSpreadLinePoints(quotes)
    expect(mid.map((p) => p.time)).toEqual([expect.any(Number), expect.any(Number)])
    expect(mid[0]!.time < mid[1]!.time).toBe(true)
  })
})

describe('isUsableAlpacaL1Quote', () => {
  it('rejects nullish prices from malformed JSON', () => {
    const bad = { type: 'quote', symbol: 'X', time: '2026-01-01T00:00:00.000Z' } as AlpacaL1WsQuote
    expect(isUsableAlpacaL1Quote(bad)).toBe(false)
  })
})
