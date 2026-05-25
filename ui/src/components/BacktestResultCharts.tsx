import {
  Accordion,
  Box,
  Group,
  Paper,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import type { CSSProperties, ReactNode } from 'react'
import { useMemo } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { BacktestResultPayload } from '../api/types'
import {
  adaptEquityCurve,
  adaptExposureHistory,
  adaptTargetHistoryStacked,
  adaptTurnoverHistory,
  adaptTurnoverPctHistory,
  buildPerformanceOverlay,
  equityBarReturns,
  formatChartAxisCompact,
  formatMetricNumber,
  padEpochMsDomain,
  periodsPerYearFromMetrics,
  returnHistogramBins,
  rollingSharpeFromEquity,
  summarizeDrawdownAnalysis,
  summarizeReturnsAnalysis,
  summarizeSharpeAnalysis,
  targetHistoryConcentration,
} from '../lib/backtestCharts'
import { tickDate } from '../lib/chartTimeTicks'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import MonthlyReturnsHeatmap from './MonthlyReturnsHeatmap'
import { BacktestDrawdownChart } from './backtest/BacktestDrawdownChart'
import { BacktestPerformanceChart } from './backtest/BacktestPerformanceChart'
import { RechartsPanel } from './charts/RechartsPanel'
import { useBacktestChartTheme } from './backtest/useBacktestChartTheme'

/** Numeric columns: monospace + tabular alignment for scanning. */
const TD_NUM: CSSProperties = { fontVariantNumeric: 'tabular-nums', textAlign: 'right' }

/**
 * Backtest charts orchestrator.
 *
 * Layout zones: keep hero time-series (performance, drawdown) in a full-width PrimarySeries
 * stack — never inside a multi-column SimpleGrid cell. Secondary charts (heatmap, histograms,
 * Sharpe, targets, concentration, turnover) live in SecondaryGrid; each cell uses `miw={0}` so
 * grid `min-width: auto` does not collapse Recharts to zero width.
 */
function BenchmarkPanel({ benchmark }: { benchmark: Record<string, unknown> }) {
  const err = benchmark.error
  if (err != null && String(err).length > 0) {
    return (
      <Stack gap="xs">
        <Title order={5} size="sm">
          Benchmark
        </Title>
        <Text size="sm" c="dimmed">
          {String(err)}
        </Text>
      </Stack>
    )
  }

  const ticker = typeof benchmark.ticker === 'string' ? benchmark.ticker : null
  const cor = benchmark.correlation
  const nOverlap = benchmark.n_overlap
  const benchTr = benchmark.benchmark_total_return_pct

  return (
    <Stack gap="xs">
      <Title order={5} size="sm">
        Benchmark
      </Title>
      <Group gap="xl" wrap="wrap">
        {ticker && (
          <Stack gap={2}>
            <Text size="xs" c="dimmed">
              Ticker
            </Text>
            <Text ff="monospace" size="sm">
              {ticker}
            </Text>
          </Stack>
        )}
        <Stack gap={2}>
          <Text size="xs" c="dimmed">
            Correlation (vs strategy returns)
          </Text>
          <Text ff="monospace" size="sm">
            {formatMetricNumber(cor, 4)}
          </Text>
        </Stack>
        <Stack gap={2}>
          <Text size="xs" c="dimmed">
            Overlap bars
          </Text>
          <Text ff="monospace" size="sm">
            {typeof nOverlap === 'number' && Number.isFinite(nOverlap)
              ? String(nOverlap)
              : '—'}
          </Text>
        </Stack>
        <Stack gap={2}>
          <Text size="xs" c="dimmed">
            Benchmark total return %
          </Text>
          <Text ff="monospace" size="sm">
            {typeof benchTr === 'number' && Number.isFinite(benchTr)
              ? `${benchTr.toFixed(2)}%`
              : '—'}
          </Text>
        </Stack>
      </Group>
    </Stack>
  )
}

function pctLabel(v: unknown, digits: number): string {
  const s = formatMetricNumber(v, digits)
  return s === '—' ? '—' : `${s}%`
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : ''
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function GroupExposureLatestTable({
  rows,
  tableProps,
}: {
  rows: { date: string; gross_by_group: Record<string, unknown>; net_by_group: Record<string, unknown> }[]
  tableProps: ReturnType<typeof useMantineTableDensity>
}) {
  if (rows.length === 0) return null
  const last = rows[rows.length - 1]!
  const gross = last.gross_by_group
  const sorted = Object.entries(gross).sort((a, b) => Number(b[1]) - Number(a[1]))
  if (sorted.length === 0) return null
  return (
    <Paper withBorder p="md" radius="md">
      <Title order={4} size="h5" mb="sm">
        Group / sector exposure (latest bar)
      </Title>
      <Text size="xs" c="dimmed" mb="sm">
        As of {last.date}
      </Text>
      <Table.ScrollContainer minWidth={320}>
        <Table {...tableProps} striped>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Group</Table.Th>
              <Table.Th ta="right">Gross</Table.Th>
              <Table.Th ta="right">Net</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {sorted.map(([g]) => (
              <Table.Tr key={g}>
                <Table.Td ff="monospace">{g}</Table.Td>
                <Table.Td ff="monospace" style={TD_NUM}>
                  {formatMetricNumber(gross[g], 4)}
                </Table.Td>
                <Table.Td ff="monospace" style={TD_NUM}>
                  {formatMetricNumber(last.net_by_group[g], 4)}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Paper>
  )
}

function AnalyzerSummaryCard({
  metrics,
  drawdownAnalysis,
  returnsAnalysis,
  sharpeAnalysis,
}: {
  metrics: Record<string, unknown>
  drawdownAnalysis: unknown
  returnsAnalysis: unknown
  sharpeAnalysis: unknown
}) {
  const dd = summarizeDrawdownAnalysis(drawdownAnalysis)
  const ra = summarizeReturnsAnalysis(returnsAnalysis)
  const sa = summarizeSharpeAnalysis(sharpeAnalysis)
  return (
    <Paper withBorder p="md" radius="md">
      <Title order={4} size="h5" mb="sm">
        Backtrader analyzers
      </Title>
      <Text size="xs" c="dimmed" mb="md">
        Trimmed portfolio metrics (cards above) may differ from raw analyzer outputs on the same
        window.
      </Text>
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
        <Stack gap={4}>
          <Text size="xs" c="dimmed" tt="uppercase">
            DrawDown (analyzer)
          </Text>
          <Text size="sm">
            Max DD (frac):{' '}
            <Text span ff="monospace">
              {dd.maxDrawdownFrac != null ? formatMetricNumber(dd.maxDrawdownFrac * 100, 2) + '%' : '—'}
            </Text>
          </Text>
          <Text size="sm">
            Max length (bars):{' '}
            <Text span ff="monospace">
              {dd.maxLen != null ? String(dd.maxLen) : '—'}
            </Text>
          </Text>
          <Text size="sm">
            Max money down:{' '}
            <Text span ff="monospace">
              {dd.maxMoneyDown != null ? formatMetricNumber(dd.maxMoneyDown, 0) : '—'}
            </Text>
          </Text>
        </Stack>
        <Stack gap={4}>
          <Text size="xs" c="dimmed" tt="uppercase">
            Returns / Sharpe (analyzer)
          </Text>
          <Text size="sm">
            rtot:{' '}
            <Text span ff="monospace">
              {ra.rtot != null ? pctLabel(ra.rtot * 100, 2) : '—'}
            </Text>
          </Text>
          <Text size="sm">
            ravg (per bar):{' '}
            <Text span ff="monospace">
              {ra.ravg != null ? formatMetricNumber(ra.ravg * 100, 4) + '%' : '—'}
            </Text>
          </Text>
          <Text size="sm">
            sharperatio:{' '}
            <Text span ff="monospace">
              {formatMetricNumber(sa.sharperatio, 4)}
            </Text>
          </Text>
          <Text size="sm">
            vs trimmed Sharpe:{' '}
            <Text span ff="monospace">
              {formatMetricNumber(metrics.sharpe_ratio, 4)}
            </Text>
          </Text>
        </Stack>
      </SimpleGrid>
    </Paper>
  )
}

const STACK_PALETTE = [
  '#f59e0b',
  '#3b82f6',
  '#10b981',
  '#8b5cf6',
  '#ec4899',
  '#06b6d4',
  '#84cc16',
  '#f97316',
  '#6366f1',
  '#14b8a6',
]

export default function BacktestResultCharts({
  data,
  metricsStrip = false,
  balancedChartColumns = false,
  show = 'all',
}: {
  data: BacktestResultPayload
  /** Single horizontal row of metric tiles (e.g. Alpha Studio). */
  metricsStrip?: boolean
  /** Place chart widgets in two balanced columns. */
  balancedChartColumns?: boolean
  /** ``metrics`` / ``charts`` for split layout (e.g. AI row between). */
  show?: 'all' | 'metrics' | 'charts'
}) {
  const cht = useBacktestChartTheme()
  const tableProps = useMantineTableDensity()

  const equityPts = useMemo(() => adaptEquityCurve(data.equity_curve), [data.equity_curve])
  const perfOverlay = useMemo(
    () => buildPerformanceOverlay(data.equity_curve, data.benchmark ?? undefined),
    [data.equity_curve, data.benchmark],
  )
  const turnoverPts = useMemo(
    () => adaptTurnoverHistory(data.turnover_history, data.equity_curve),
    [data.turnover_history, data.equity_curve],
  )

  const targetHist = useMemo(() => {
    const th = data.target_history
    return Array.isArray(th) ? th : []
  }, [data.target_history])

  const targetStack = useMemo(
    () => adaptTargetHistoryStacked(targetHist, 10),
    [targetHist],
  )

  const concPts = useMemo(() => targetHistoryConcentration(targetHist), [targetHist])

  const periodsPerYear = useMemo(
    () => periodsPerYearFromMetrics(data.metrics),
    [data.metrics],
  )

  const barReturns = useMemo(() => equityBarReturns(equityPts), [equityPts])
  const histBins = useMemo(() => returnHistogramBins(barReturns, 20), [barReturns])

  const rollWindow = useMemo(() => {
    const n = barReturns.length
    if (n < 12) return 5
    return Math.min(63, Math.max(8, Math.floor(n / 6)))
  }, [barReturns.length])

  const rollSharpe = useMemo(
    () => rollingSharpeFromEquity(equityPts, rollWindow, periodsPerYear),
    [equityPts, rollWindow, periodsPerYear],
  )

  const rollSharpeTimeDomain = useMemo((): [number, number] | null => {
    if (rollSharpe.length === 0) return null
    const ts = rollSharpe.map((r) => r.t).filter((t) => Number.isFinite(t))
    if (ts.length === 0) return null
    return [Math.min(...ts), Math.max(...ts)]
  }, [rollSharpe])

  const concTimeDomain = useMemo((): [number, number] | null => {
    if (concPts.length === 0) return null
    const ts = concPts.map((r) => r.t).filter((t) => Number.isFinite(t))
    if (ts.length === 0) return null
    return [Math.min(...ts), Math.max(...ts)]
  }, [concPts])

  const targetTimeDomain = useMemo((): [number, number] | null => {
    const s = targetStack.series
    if (s.length === 0) return null
    const ts = s.map((r) => r.t).filter((t) => Number.isFinite(t))
    if (ts.length === 0) return null
    return [Math.min(...ts), Math.max(...ts)]
  }, [targetStack.series])

  const histXDomain = useMemo((): [number, number] | null => {
    if (histBins.length === 0) return null
    return [histBins[0]!.x0, histBins[histBins.length - 1]!.x1]
  }, [histBins])

  const targetWeightAxisMode = useMemo(() => {
    let maxAbs = 0
    for (const row of targetStack.series) {
      for (const k of targetStack.keys) {
        const v = row[k]
        if (typeof v === 'number' && Number.isFinite(v)) maxAbs = Math.max(maxAbs, Math.abs(v))
      }
    }
    if (maxAbs <= 1.5) return 'fraction' as const
    if (maxAbs <= 100) return 'percentWhole' as const
    return 'notional' as const
  }, [targetStack.series, targetStack.keys])

  const xDomain = useMemo((): [number, number] | null => {
    if (equityPts.length === 0) return null
    const t0 = equityPts[0]!.t
    const t1 = equityPts[equityPts.length - 1]!.t
    if (!Number.isFinite(t0) || !Number.isFinite(t1)) return null
    return padEpochMsDomain([t0, t1])
  }, [equityPts])

  const exposurePts = useMemo(() => {
    const rows = data.exposure_history
    return Array.isArray(rows) ? adaptExposureHistory(rows) : []
  }, [data.exposure_history])

  const turnoverPctPts = useMemo(() => {
    const rows = data.turnover_pct_history
    return Array.isArray(rows) ? adaptTurnoverPctHistory(rows) : []
  }, [data.turnover_pct_history])

  const tradeEventsList = useMemo(() => {
    const te = data.trade_events
    return Array.isArray(te) ? (te as Record<string, unknown>[]) : []
  }, [data.trade_events])

  const sectorPieSlices = useMemo(() => {
    const gh = data.group_exposure_history
    if (!Array.isArray(gh) || gh.length === 0) return [] as { name: string; value: number }[]
    const last = gh[gh.length - 1] as { gross_by_group?: Record<string, unknown> }
    const g = last?.gross_by_group
    if (!g || typeof g !== 'object') return []
    let total = 0
    for (const v of Object.values(g)) {
      const n = typeof v === 'number' ? v : Number(v)
      if (Number.isFinite(n) && n > 0) total += n
    }
    if (total <= 0) return []
    return Object.entries(g)
      .map(([name, v]) => {
        const val = typeof v === 'number' ? v : Number(v)
        return { name, value: Number.isFinite(val) && val > 0 ? (val / total) * 100 : 0 }
      })
      .filter((s) => s.value > 0)
      .sort((a, b) => b.value - a.value)
  }, [data.group_exposure_history])

  const exposureTimeDomain = useMemo((): [number, number] | null => {
    if (exposurePts.length === 0) return null
    const ts = exposurePts.map((r) => r.t).filter((t) => Number.isFinite(t))
    if (ts.length === 0) return null
    return [Math.min(...ts), Math.max(...ts)]
  }, [exposurePts])

  const turnoverPctTimeDomain = useMemo((): [number, number] | null => {
    if (turnoverPctPts.length === 0) return null
    const ts = turnoverPctPts.map((r) => r.t).filter((t) => Number.isFinite(t))
    if (ts.length === 0) return null
    return [Math.min(...ts), Math.max(...ts)]
  }, [turnoverPctPts])

  const m = data.metrics
  const hasBenchmark = data.benchmark != null && typeof data.benchmark === 'object'
  const benchTicker =
    hasBenchmark && typeof (data.benchmark as { ticker?: string }).ticker === 'string'
      ? (data.benchmark as { ticker: string }).ticker
      : null
  const showBenchTurnover = hasBenchmark || turnoverPts.length > 0
  const hasAnalyzerPayload =
    data.drawdown_analysis != null ||
    data.returns_analysis != null ||
    data.sharpe_analysis != null

  const metricTile = (key: string, label: string, value: ReactNode) => (
    <Paper key={key} withBorder p="sm" radius="md" miw={metricsStrip ? 120 : undefined}>
      <Text size="xs" c="dimmed">
        {label}
      </Text>
      <Text fw={700}>{value}</Text>
    </Paper>
  )

  const metricNodes: ReactNode[] = [
    metricTile('cagr', 'CAGR', pctLabel(m.cagr_pct, 2)),
    metricTile('sharpe', 'Sharpe', formatMetricNumber(m.sharpe_ratio, 3)),
    metricTile('mdd', 'Max drawdown', pctLabel(m.max_drawdown_pct, 2)),
    metricTile('win', 'Win rate', pctLabel(m.win_rate_pct, 1)),
    metricTile('tot', 'Total return', pctLabel(m.total_return_pct, 2)),
    metricTile('at', 'Avg turnover % / rebalance', pctLabel(m.avg_turnover_pct, 2)),
    metricTile('reb', 'Rebalances', formatMetricNumber(m.rebalance_count, 0)),
    metricTile('top', 'Top name gross %', pctLabel(m.top_name_gross_share_pct, 1)),
  ]
  if (m.max_group_gross_share_pct != null) {
    metricNodes.push(
      metricTile('mgg', 'Max group gross %', pctLabel(m.max_group_gross_share_pct, 1)),
    )
  }
  if (m.max_group_net_share_pct != null) {
    metricNodes.push(metricTile('mgn', 'Max group net %', pctLabel(m.max_group_net_share_pct, 1)))
  }

  const chartBreak = balancedChartColumns
    ? ({ marginBottom: 'var(--mantine-spacing-lg)', breakInside: 'avoid' as const } satisfies CSSProperties)
    : undefined

  const showMetrics = show !== 'charts'
  const showCharts = show !== 'metrics'

  return (
    <>
      {showMetrics && (
        <Stack gap="lg">
          {metricsStrip ? (
            <ScrollArea scrollbars="x" offsetScrollbars="x">
              <Group wrap="nowrap" align="stretch" gap="sm" pb={4}>
                {metricNodes}
              </Group>
            </ScrollArea>
          ) : (
            <SimpleGrid cols={{ base: 2, sm: 3, md: 4 }} spacing="sm">
              {metricNodes}
            </SimpleGrid>
          )}
        </Stack>
      )}

      {showCharts && (
        <Stack gap="lg">
          {equityPts.length > 0 && xDomain && (
            <Stack gap="lg">
              <Paper withBorder p="md" radius="md" miw={0} style={chartBreak}>
                <BacktestPerformanceChart
                  theme={cht}
                  equityPts={equityPts}
                  perfOverlay={perfOverlay}
                  xDomain={xDomain}
                  benchTicker={benchTicker}
                  tradeEvents={tradeEventsList}
                />
              </Paper>

              <Paper withBorder p="md" radius="md" miw={0} style={chartBreak}>
                <BacktestDrawdownChart theme={cht} points={equityPts} xDomain={xDomain} />
              </Paper>

              {exposurePts.length > 0 && exposureTimeDomain && (
                <Paper withBorder p="md" radius="md" miw={0} style={chartBreak}>
                  <Title order={4} size="h5" mb="sm">
                    Gross leverage and exposures
                  </Title>
                  <Text size="xs" c="dimmed" mb="sm">
                    Target notionals vs equity: gross leverage (sum of absolute weights), long and
                    short sides as fractions of equity.
                  </Text>
                  <RechartsPanel heightPx={240} dataLength={exposurePts.length}>
                    <LineChart data={exposurePts} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                      <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="t"
                        domain={exposureTimeDomain}
                        tickFormatter={tickDate}
                        tick={cht.chartAxisStyle}
                      />
                      <YAxis
                        tick={cht.chartAxisStyle}
                        tickFormatter={(v) => `${Number(v).toFixed(2)}×`}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={cht.tooltipStyle}
                        labelFormatter={(ms) => (typeof ms === 'number' ? new Date(ms).toLocaleString() : '')}
                      />
                      <Legend />
                      <Line
                        type="stepAfter"
                        dataKey="grossLeverage"
                        name="Gross leverage"
                        stroke={cht.accent}
                        dot={false}
                        isAnimationActive={false}
                      />
                      <Line
                        type="stepAfter"
                        dataKey="longExposure"
                        name="Long / equity"
                        stroke={cht.concBlue}
                        dot={false}
                        isAnimationActive={false}
                      />
                      <Line
                        type="stepAfter"
                        dataKey="shortExposure"
                        name="Short / equity"
                        stroke={cht.concCyan}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </RechartsPanel>
                </Paper>
              )}
            </Stack>
          )}

          <SimpleGrid cols={{ base: 1, sm: balancedChartColumns ? 2 : 1 }} spacing="lg">
            {equityPts.length > 0 && (
              <Box miw={0} style={chartBreak}>
                <MonthlyReturnsHeatmap equityCurve={data.equity_curve} />
              </Box>
            )}

            {turnoverPctPts.length > 0 && turnoverPctTimeDomain && (
              <Box miw={0} style={chartBreak}>
                <Paper withBorder p="md" radius="md">
                  <Title order={4} size="h5" mb="sm">
                    Turnover (% of equity per bar)
                  </Title>
                  <Text size="xs" c="dimmed" mb="sm">
                    One-way turnover as sum of absolute target changes divided by equity at that bar.
                    Bar unit: {String(m.bar_unit ?? 'DAYS')}
                    {Number(m.bar_step) > 1 ? ` ×${String(m.bar_step)}` : ''}.
                  </Text>
                  <RechartsPanel heightPx={220} dataLength={turnoverPctPts.length}>
                    <LineChart data={turnoverPctPts} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                      <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="t"
                        domain={turnoverPctTimeDomain}
                        tickFormatter={tickDate}
                        tick={cht.chartAxisStyle}
                      />
                      <YAxis
                        tick={cht.chartAxisStyle}
                        tickFormatter={(v) => `${Number(v).toFixed(1)}%`}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={cht.tooltipStyle}
                        labelFormatter={(ms) => (typeof ms === 'number' ? new Date(ms).toLocaleString() : '')}
                      />
                      <Line
                        type="stepAfter"
                        dataKey="turnoverPct"
                        name="Turnover % equity"
                        stroke={cht.turnoverLine}
                        strokeWidth={1.5}
                        dot={false}
                        connectNulls
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </RechartsPanel>
                </Paper>
              </Box>
            )}

            {sectorPieSlices.length > 0 && (
              <Box miw={0} style={chartBreak}>
                <Paper withBorder p="md" radius="md">
                  <Title order={4} size="h5" mb="sm">
                    Sector gross exposure (latest bar)
                  </Title>
                  <Text size="xs" c="dimmed" mb="sm">
                    Share of total gross target notionals by group (e.g. sector).
                  </Text>
                  <RechartsPanel heightPx={280} dataLength={sectorPieSlices.length}>
                    <PieChart>
                      <Pie
                        data={sectorPieSlices}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        label={({ name, percent }) => `${name}: ${((percent ?? 0) * 100).toFixed(0)}%`}
                      >
                        {sectorPieSlices.map((_, i) => (
                          <Cell key={`cell-${i}`} fill={STACK_PALETTE[i % STACK_PALETTE.length]!} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={cht.tooltipStyle} />
                    </PieChart>
                  </RechartsPanel>
                </Paper>
              </Box>
            )}

            {histBins.some((b) => b.count > 0) && (
              <Box miw={0} style={chartBreak}>
                <Paper withBorder p="md" radius="md">
                  <Title order={4} size="h5" mb="sm">
                    Bar return distribution
                  </Title>
                  <Text size="xs" c="dimmed" mb="sm">
                    Histogram of equity step returns (same cadence as bar spec).
                  </Text>
                  <RechartsPanel heightPx={220} dataLength={histBins.length}>
                    <BarChart data={histBins} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                      <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="mid"
                        domain={histXDomain ?? ['auto', 'auto']}
                        tick={cht.chartAxisStyle}
                        tickFormatter={(v) => {
                          const x = Number(v)
                          if (!Number.isFinite(x)) return ''
                          if (x === 0) return '0'
                          const a = Math.abs(x)
                          return a < 0.01 ? x.toExponential(1) : x.toPrecision(3)
                        }}
                        label={{
                          value: 'Bar return',
                          position: 'insideBottom',
                          offset: -2,
                          fill: cht.muted,
                          fontSize: 11,
                        }}
                      />
                      <YAxis tick={cht.chartAxisStyle} allowDecimals={false} />
                      <Tooltip
                        contentStyle={cht.tooltipStyle}
                        labelFormatter={(label) => {
                          const v = Number(label)
                          if (!Number.isFinite(v) || histBins.length === 0) return String(label)
                          const nearest = histBins.reduce(
                            (best, b) => (Math.abs(b.mid - v) < Math.abs(best.mid - v) ? b : best),
                            histBins[0]!,
                          )
                          return `${nearest.x0.toPrecision(4)} … ${nearest.x1.toPrecision(4)}`
                        }}
                      />
                      <Bar
                        dataKey="count"
                        name="Count"
                        fill={cht.turnoverBar}
                        stroke={cht.turnoverBarStroke}
                        isAnimationActive={false}
                        maxBarSize={48}
                      />
                    </BarChart>
                  </RechartsPanel>
                </Paper>
              </Box>
            )}

      {rollSharpe.length > 0 && (
        <Box miw={0} style={chartBreak}>
          <Paper withBorder p="md" radius="md">
            <Title order={4} size="h5" mb="sm">
              Rolling Sharpe ({rollWindow} bars)
            </Title>
            <Text size="xs" c="dimmed" mb="sm">
              Annualized from rolling mean / std of bar returns; window sized to sample length.
            </Text>
            <RechartsPanel heightPx={220} dataLength={rollSharpe.length}>
              <LineChart data={rollSharpe} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="t"
                  domain={rollSharpeTimeDomain ?? ['dataMin', 'dataMax']}
                  tickFormatter={tickDate}
                  tick={cht.chartAxisStyle}
                />
                <YAxis tick={cht.chartAxisStyle} domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={cht.tooltipStyle}
                  labelFormatter={(ms) => (typeof ms === 'number' ? new Date(ms).toLocaleString() : '')}
                />
                <Line
                  type="stepAfter"
                  dataKey="sharpe"
                  name="Sharpe"
                  stroke={cht.accent}
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </RechartsPanel>
          </Paper>
        </Box>
      )}

      {targetStack.keys.length > 0 && targetStack.series.length > 0 && (
        <Box miw={0} style={chartBreak}>
          <Paper withBorder p="md" radius="md">
            <Title order={4} size="h5" mb="sm">
              Target weights (top 10 + Other)
            </Title>
            <Text size="xs" c="dimmed" mb="sm">
              Stacked notionals from strategy output per rebalance (Yahoo / FinStrat convention).
            </Text>
            <RechartsPanel heightPx={320} dataLength={targetStack.series.length}>
              <AreaChart data={targetStack.series} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="t"
                  domain={targetTimeDomain ?? ['dataMin', 'dataMax']}
                  tickFormatter={tickDate}
                  tick={cht.chartAxisStyle}
                />
                <YAxis
                  tick={cht.chartAxisStyle}
                  tickFormatter={(v) => {
                    const n = Number(v)
                    if (!Number.isFinite(n)) return ''
                    if (targetWeightAxisMode === 'fraction') return `${(n * 100).toFixed(0)}%`
                    if (targetWeightAxisMode === 'percentWhole') return `${n.toFixed(1)}%`
                    return formatChartAxisCompact(n)
                  }}
                  domain={['auto', 'auto']}
                  label={
                    targetWeightAxisMode === 'notional'
                      ? {
                          value: 'USD',
                          angle: -90,
                          position: 'insideLeft',
                          fill: cht.muted,
                          fontSize: 11,
                        }
                      : undefined
                  }
                />
                <Tooltip
                  contentStyle={cht.tooltipStyle}
                  labelFormatter={(ms) => (typeof ms === 'number' ? new Date(ms).toLocaleString() : String(ms))}
                />
                <Legend />
                {targetStack.keys.map((k, i) => (
                  <Area
                    key={k}
                    type="stepAfter"
                    dataKey={k}
                    name={k}
                    stackId="w"
                    stroke={STACK_PALETTE[i % STACK_PALETTE.length]}
                    fill={STACK_PALETTE[i % STACK_PALETTE.length]}
                    fillOpacity={0.65}
                    isAnimationActive={false}
                  />
                ))}
              </AreaChart>
            </RechartsPanel>
          </Paper>
        </Box>
      )}

      {concPts.length > 0 && (
        <Box miw={0} style={chartBreak}>
          <Paper withBorder p="md" radius="md">
            <Title order={4} size="h5" mb="sm">
              Concentration (HHI and max |weight|)
            </Title>
            <RechartsPanel heightPx={200} dataLength={concPts.length}>
              <LineChart data={concPts} margin={{ top: 8, right: 12, bottom: 8, left: 8 }}>
                <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="t"
                  domain={concTimeDomain ?? ['dataMin', 'dataMax']}
                  tickFormatter={tickDate}
                  tick={cht.chartAxisStyle}
                />
                <YAxis
                  tick={cht.chartAxisStyle}
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => formatChartAxisCompact(Number(v))}
                />
                <Tooltip
                  contentStyle={cht.tooltipStyle}
                  labelFormatter={(ms) => (typeof ms === 'number' ? tickDate(ms) : String(ms))}
                />
                <Legend />
                <Line
                  type="stepAfter"
                  dataKey="hhi"
                  name="HHI"
                  stroke={cht.concBlue}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="maxAbs"
                  name="Max |w|"
                  stroke={cht.concCyan}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </RechartsPanel>
          </Paper>
        </Box>
      )}

      {data.group_exposure_history && data.group_exposure_history.length > 0 && (
        <Box miw={0} style={chartBreak}>
          <GroupExposureLatestTable rows={data.group_exposure_history} tableProps={tableProps} />
        </Box>
      )}

      {showBenchTurnover && (
        <Box miw={0} style={chartBreak}>
          <Paper withBorder p="md" radius="md">
            <Title order={4} size="h5" mb="md">
              Benchmark and turnover
            </Title>
            <Stack gap="lg">
              {hasBenchmark && <BenchmarkPanel benchmark={data.benchmark as Record<string, unknown>} />}
              {turnoverPts.length > 0 && (
                <Stack gap="xs">
                  <Title order={5} size="sm">
                    Turnover (USD per rebalance; % of equity)
                  </Title>
                  <RechartsPanel heightPx={280} dataLength={turnoverPts.length}>
                    <ComposedChart data={turnoverPts} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
                      <CartesianGrid stroke={cht.gridStroke} strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="t"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={tickDate}
                        tick={cht.chartAxisStyle}
                      />
                      <YAxis
                        yAxisId="usd"
                        orientation="left"
                        tick={cht.chartAxisStyle}
                        tickFormatter={(v) =>
                          typeof v === 'number' && Math.abs(v) >= 1e6
                            ? `${(v / 1e6).toFixed(1)}M`
                            : typeof v === 'number' && Math.abs(v) >= 1e3
                              ? `${(v / 1e3).toFixed(0)}k`
                              : String(v)
                        }
                        label={{
                          value: 'USD',
                          angle: -90,
                          position: 'insideLeft',
                          fill: cht.muted,
                          fontSize: 11,
                        }}
                      />
                      <YAxis
                        yAxisId="pct"
                        orientation="right"
                        tick={cht.chartAxisStyle}
                        tickFormatter={(v) => `${v}%`}
                        label={{
                          value: '% equity',
                          angle: 90,
                          position: 'insideRight',
                          fill: cht.muted,
                          fontSize: 11,
                        }}
                      />
                      <Tooltip
                        contentStyle={cht.tooltipStyle}
                        labelFormatter={(ms) =>
                          typeof ms === 'number' ? new Date(ms).toLocaleString() : ''
                        }
                      />
                      <Legend />
                      <Bar
                        yAxisId="usd"
                        dataKey="turnoverUsd"
                        name="Turnover USD"
                        fill={cht.turnoverBar}
                        stroke={cht.turnoverBarStroke}
                        radius={[2, 2, 0, 0]}
                        isAnimationActive={false}
                      />
                      <Line
                        yAxisId="pct"
                        type="stepAfter"
                        dataKey="turnoverPct"
                        name="Turnover % equity"
                        stroke={cht.turnoverLine}
                        strokeWidth={1.5}
                        dot={false}
                        connectNulls
                        isAnimationActive={false}
                      />
                    </ComposedChart>
                  </RechartsPanel>
                </Stack>
              )}
            </Stack>
          </Paper>
        </Box>
      )}

      {(data.tearsheet_summary != null ||
        data.return_quantiles != null ||
        data.ff_single_factor != null) && (
        <Box miw={0} style={chartBreak}>
          <Paper withBorder p="md" radius="md">
            <Title order={4} size="h5" mb="sm">
              Tearsheet (summary)
            </Title>
            <Text size="xs" c="dimmed" mb="md">
              Pyfolio-style risk and factor summaries computed server-side (structured payload, no
              matplotlib).
            </Text>
            <Accordion variant="contained" multiple>
              {data.tearsheet_summary != null && typeof data.tearsheet_summary === 'object' && (
                <Accordion.Item value="risk">
                  <Accordion.Control>Risk and moments</Accordion.Control>
                  <Accordion.Panel>
                    <Table {...tableProps} striped>
                      <Table.Tbody>
                        {Object.entries(data.tearsheet_summary as Record<string, unknown>).map(
                          ([k, v]) => (
                            <Table.Tr key={k}>
                              <Table.Th w="45%">{k}</Table.Th>
                              <Table.Td ff="monospace" style={TD_NUM}>
                                {Array.isArray(v) ? JSON.stringify(v) : formatCell(v)}
                              </Table.Td>
                            </Table.Tr>
                          ),
                        )}
                      </Table.Tbody>
                    </Table>
                  </Accordion.Panel>
                </Accordion.Item>
              )}
              {data.return_quantiles != null && typeof data.return_quantiles === 'object' && (
                <Accordion.Item value="quantiles">
                  <Accordion.Control>Return quantiles</Accordion.Control>
                  <Accordion.Panel>
                    <Table {...tableProps} striped>
                      <Table.Tbody>
                        {Object.entries(data.return_quantiles as Record<string, unknown>).map(
                          ([k, v]) => (
                            <Table.Tr key={k}>
                              <Table.Th w="45%">{k}</Table.Th>
                              <Table.Td ff="monospace" style={TD_NUM}>
                                {formatCell(v)}
                              </Table.Td>
                            </Table.Tr>
                          ),
                        )}
                      </Table.Tbody>
                    </Table>
                  </Accordion.Panel>
                </Accordion.Item>
              )}
              {data.ff_single_factor != null && typeof data.ff_single_factor === 'object' && (
                <Accordion.Item value="ff">
                  <Accordion.Control>Fama–French single-factor regressions</Accordion.Control>
                  <Accordion.Panel>
                    {(() => {
                      const ff = data.ff_single_factor as Record<string, unknown>
                      const err = ff.error
                      if (err != null && String(err).length > 0) {
                        return (
                          <Text size="sm" c="dimmed">
                            {String(err)}
                          </Text>
                        )
                      }
                      const fac = ff.factors
                      if (!fac || typeof fac !== 'object') {
                        return (
                          <Text size="sm" c="dimmed">
                            No factor regression results.
                          </Text>
                        )
                      }
                      const rows = Object.entries(fac as Record<string, Record<string, unknown>>)
                      return (
                        <Table {...tableProps} striped>
                          <Table.Thead>
                            <Table.Tr>
                              <Table.Th>Factor</Table.Th>
                              <Table.Th ta="right">Beta</Table.Th>
                              <Table.Th ta="right">α ann %</Table.Th>
                              <Table.Th ta="right">R²</Table.Th>
                              <Table.Th ta="right">n</Table.Th>
                            </Table.Tr>
                          </Table.Thead>
                          <Table.Tbody>
                            {rows.map(([name, row]) => (
                              <Table.Tr key={name}>
                                <Table.Td ff="monospace">{name}</Table.Td>
                                <Table.Td ff="monospace" style={TD_NUM}>
                                  {formatCell(row.beta)}
                                </Table.Td>
                                <Table.Td ff="monospace" style={TD_NUM}>
                                  {formatCell(row.alpha_ann_pct)}
                                </Table.Td>
                                <Table.Td ff="monospace" style={TD_NUM}>
                                  {formatCell(row.r2)}
                                </Table.Td>
                                <Table.Td ff="monospace" style={TD_NUM}>
                                  {formatCell(row.n)}
                                </Table.Td>
                              </Table.Tr>
                            ))}
                          </Table.Tbody>
                        </Table>
                      )
                    })()}
                  </Accordion.Panel>
                </Accordion.Item>
              )}
            </Accordion>
          </Paper>
        </Box>
      )}

      {hasAnalyzerPayload && (
        <Box miw={0} style={chartBreak}>
          <AnalyzerSummaryCard
            metrics={m}
            drawdownAnalysis={data.drawdown_analysis}
            returnsAnalysis={data.returns_analysis}
            sharpeAnalysis={data.sharpe_analysis}
          />
        </Box>
      )}

          </SimpleGrid>

          {equityPts.length === 0 && turnoverPts.length === 0 && targetStack.series.length === 0 && (
            <Text c="dimmed" size="sm">
              No equity, turnover, or target series to chart.
            </Text>
          )}
        </Stack>
      )}

    </>
  )
}
