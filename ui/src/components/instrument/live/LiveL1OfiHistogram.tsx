import { Stack, Text, Tooltip } from '@mantine/core'
import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { buildOfiHistogramFromQuotes } from '../../../live/l1Derived'
import { useLiveL1 } from '../../../live/l1Store'

/** Histogram of recent quote imbalance samples (definition in `l1Derived.ts`). */
export function LiveL1OfiHistogram() {
  const { state } = useLiveL1()
  const quotes = state.quotes

  const bins = useMemo(() => buildOfiHistogramFromQuotes(quotes), [quotes])

  return (
    <Stack gap="xs" style={{ minHeight: 260 }}>
      <Tooltip
        label="Per-quote log size skew: log(1+bid_sz)−log(1+ask_sz); last 120 quotes, 10 equal-width bins."
        multiline
        w={300}
      >
        <Text size="sm" style={{ cursor: 'help' }}>
          Quote imbalance histogram
        </Text>
      </Tooltip>
      <div style={{ width: '100%', height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 8, right: 8, bottom: 64, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="label" interval={0} angle={-35} textAnchor="end" height={70} tick={{ fontSize: 9 }} />
            <YAxis allowDecimals={false} />
            <RTooltip />
            <Bar dataKey="count" fill="var(--mantine-color-blue-6)" name="Quotes" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Stack>
  )
}
