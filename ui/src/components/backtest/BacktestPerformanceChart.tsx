import { Alert, Text, Title } from '@mantine/core'
import { useMemo } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { RechartsPanel } from '../charts/RechartsPanel'
import { tickDate } from '../../lib/chartTimeTicks'
import {
  assertPlottableEquitySeries,
  assertPlottablePerformanceOverlay,
  type EquityChartRow,
  equityLineYDomain,
  type PerformanceOverlayRow,
} from '../../lib/backtestCharts'
import type { BacktestChartTheme } from './useBacktestChartTheme'
import { BacktestPerformanceLcPane } from './BacktestPerformanceLcPane'

const PERF_CHART_HEIGHT_PX = 520

type Props = {
  theme: BacktestChartTheme
  equityPts: EquityChartRow[]
  perfOverlay: PerformanceOverlayRow[] | null
  xDomain: [number, number]
  benchTicker: string | null
  tradeEvents?: Record<string, unknown>[]
}

export function BacktestPerformanceChart({
  theme,
  equityPts,
  perfOverlay,
  xDomain,
  benchTicker,
  tradeEvents = [],
}: Props) {
  const { accent, chartAxisStyle, gridStroke, tooltipStyle } = theme

  const guard = perfOverlay
    ? assertPlottablePerformanceOverlay(perfOverlay)
    : assertPlottableEquitySeries(equityPts)

  const equityYDomain = useMemo(() => equityLineYDomain(equityPts), [equityPts])

  const chartMargin = { top: 12, right: 20, bottom: 56, left: 56 } as const

  return (
    <>
      <Title order={4} size="h5" mb="sm">
        Performance
      </Title>
      {perfOverlay ? (
        <Text size="xs" c="dimmed" mb="sm">
          Indexed to 100 at the first bar where benchmark data is available (buy-and-hold benchmark
          in USD); crosshair shows indexed levels.
          {benchTicker ? ` Benchmark: ${benchTicker}.` : ''}
        </Text>
      ) : (
        <Text size="xs" c="dimmed" mb="sm">
          Strategy equity curve.
        </Text>
      )}
      {!guard.ok ? (
        <Alert color="yellow" title="Chart data" mb="sm">
          Cannot plot performance ({guard.ok === false ? guard.reason : ''}).
        </Alert>
      ) : perfOverlay ? (
        <BacktestPerformanceLcPane
          overlay={perfOverlay}
          heightPx={PERF_CHART_HEIGHT_PX}
          theme={theme}
          benchTicker={benchTicker}
          tradeEvents={tradeEvents}
        />
      ) : (
        <RechartsPanel heightPx={PERF_CHART_HEIGHT_PX} dataLength={equityPts.length}>
          <LineChart data={equityPts} margin={chartMargin}>
            <CartesianGrid stroke={gridStroke} strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="t"
              domain={xDomain}
              tickFormatter={tickDate}
              tick={chartAxisStyle}
            />
            <YAxis
              tick={chartAxisStyle}
              domain={equityYDomain ?? ['auto', 'auto']}
              tickFormatter={(v) => {
                const n = typeof v === 'number' ? v : Number(v)
                if (!Number.isFinite(n)) return ''
                if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`
                if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`
                return String(n)
              }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(ms) => (typeof ms === 'number' ? new Date(ms).toLocaleString() : '')}
              formatter={(value: unknown, name: string) => {
                const v = value as number
                return [
                  name === 'Equity' || name === 'equity'
                    ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
                    : v,
                  name === 'Equity' || name === 'equity' ? 'Equity' : name,
                ]
              }}
            />
            <Line
              type="monotone"
              dataKey="equity"
              name="Equity"
              stroke={accent}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Legend verticalAlign="bottom" height={36} />
          </LineChart>
        </RechartsPanel>
      )}
    </>
  )
}
