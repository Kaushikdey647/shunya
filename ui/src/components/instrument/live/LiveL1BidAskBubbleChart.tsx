import { Stack, Text, Tooltip } from '@mantine/core'
import { useMemo } from 'react'
import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { isUsableAlpacaL1Quote, quoteLogSizeImbalance } from '../../../live/l1Derived'
import { useLiveL1 } from '../../../live/l1Store'

type Row = { t: number; mid: number; z: number; bid: number; ask: number }

/** Bubble size from log-space bid/ask size skew; green when bid size dominates, red when ask dominates. */
export function LiveL1BidAskBubbleChart() {
  const { state } = useLiveL1()
  const quotes = state.quotes

  const data = useMemo<Row[]>(() => {
    return quotes.slice(-180).filter(isUsableAlpacaL1Quote).map((q) => {
      const mid = (q.bid_price + q.ask_price) / 2
      const imb = quoteLogSizeImbalance(q)
      return {
        t: Date.parse(q.time),
        mid,
        z: Math.abs(imb) * 80 + 20,
        bid: q.bid_size,
        ask: q.ask_size,
      }
    })
  }, [quotes])

  const bidHeavy = useMemo(() => data.filter((r) => r.bid >= r.ask), [data])
  const askHeavy = useMemo(() => data.filter((r) => r.ask > r.bid), [data])

  return (
    <Stack gap="xs" style={{ minHeight: 280 }}>
      <Tooltip
        label="Each point is one BBO quote: Y = mid, area ∝ |log(1+bid_sz)−log(1+ask_sz)|. Teal = bid size ≥ ask size; red = ask size larger."
        multiline
        w={300}
      >
        <Text size="sm" style={{ cursor: 'help' }}>
          Bid/ask size imbalance (bubble)
        </Text>
      </Tooltip>
      <div style={{ width: '100%', height: 280 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              type="number"
              dataKey="t"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(v) => {
                const d = new Date(v)
                return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`
              }}
              name="Time"
            />
            <YAxis type="number" dataKey="mid" name="Mid" domain={['auto', 'auto']} />
            <ZAxis type="number" dataKey="z" range={[40, 400]} name="Skew" />
            <RTooltip
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(value: number, name: string, props: { payload?: Row }) => {
                if (name === 'mid') return [value.toFixed(4), 'Mid']
                if (props.payload) {
                  return [
                    `mid ${props.payload.mid.toFixed(4)} bid ${props.payload.bid} ask ${props.payload.ask}`,
                    'Quote',
                  ]
                }
                return [String(value), name]
              }}
            />
            <Scatter name="Bid-heavy" data={bidHeavy} fill="var(--mantine-color-teal-6)" />
            <Scatter name="Ask-heavy" data={askHeavy} fill="var(--mantine-color-red-6)" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </Stack>
  )
}
