import { Alert, Text, Title } from '@mantine/core'
import { useMemo } from 'react'
import { Area, AreaChart, CartesianGrid, Tooltip, XAxis, YAxis } from 'recharts'
import { RechartsPanel } from '../charts/RechartsPanel'
import { tickDate } from '../../lib/chartTimeTicks'
import {
  assertPlottableDrawdownSeries,
  drawdownPercentYDomain,
  type EquityChartRow,
} from '../../lib/backtestCharts'
import type { BacktestChartTheme } from './useBacktestChartTheme'

type Props = {
  theme: BacktestChartTheme
  points: EquityChartRow[]
  xDomain: [number, number]
}

export function BacktestDrawdownChart({ theme, points, xDomain }: Props) {
  const guard = assertPlottableDrawdownSeries(points)

  const yDomain = useMemo(() => drawdownPercentYDomain(points), [points])

  const axis = theme.chartAxisStyle

  const chartMargin = { top: 12, right: 16, bottom: 44, left: 52 } as const

  return (
    <>
      <Title order={4} size="h5" mb="sm">
        Drawdown
      </Title>
      <Text size="xs" c="dimmed" mb="sm">
        Underwater % from running peak equity (same bar spec as the backtest).
      </Text>
      {!guard.ok ? (
        <Alert color="yellow" title="Chart data" mb="sm">
          Cannot plot drawdown ({guard.reason}).
        </Alert>
      ) : (
        <RechartsPanel heightPx={240} dataLength={points.length}>
          <AreaChart data={points} margin={chartMargin}>
            <CartesianGrid stroke={theme.gridStroke} strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="t"
              domain={xDomain}
              tickFormatter={tickDate}
              tick={axis}
            />
            <YAxis
              tick={axis}
              tickFormatter={(v) => {
                const n = typeof v === 'number' ? v : Number(v)
                if (!Number.isFinite(n) || Math.abs(n) > 1_000_000) return ''
                return `${n.toFixed(1)}%`
              }}
              domain={yDomain == null ? ['auto', 'auto'] : yDomain}
              allowDecimals
            />
            <Tooltip
              contentStyle={theme.tooltipStyle}
              labelFormatter={(ms) => (typeof ms === 'number' ? new Date(ms).toLocaleString() : '')}
            />
            <Area
              type="stepAfter"
              dataKey="drawdownPct"
              name="Drawdown %"
              stroke={theme.ddStroke}
              fill={theme.ddFill}
              fillOpacity={0.55}
              isAnimationActive={false}
            />
          </AreaChart>
        </RechartsPanel>
      )}
    </>
  )
}
