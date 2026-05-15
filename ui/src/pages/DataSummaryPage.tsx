import {
  Anchor,
  Button,
  Code,
  Group,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Text,
  Title,
  useMantineColorScheme,
  useMantineTheme,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import type { CSSProperties } from 'react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getDataDashboard } from '../api/endpoints'
import type {
  ClassificationLabelCount,
  DashboardBucketParam,
  TickerDashboardRow,
} from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'

const INTERVAL_OPTIONS = [
  '1m',
  '2m',
  '5m',
  '15m',
  '30m',
  '60m',
  '90m',
  '1h',
  '1d',
  '5d',
  '1wk',
  '1mo',
  '3mo',
] as const

const BUCKET_OPTIONS: DashboardBucketParam[] = ['auto', 'day', 'week', 'month']

const MISSING_PIE_TOP_N = 10

function collapseClassificationCounts(
  counts: ClassificationLabelCount[],
  maxSlices = 12,
): { name: string; value: number }[] {
  if (!counts.length) return []
  const sorted = [...counts].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
  if (sorted.length <= maxSlices) {
    return sorted.map((c) => ({ name: c.label, value: c.count }))
  }
  const head = sorted.slice(0, maxSlices - 1)
  const tail = sorted.slice(maxSlices - 1)
  const otherSum = tail.reduce((s, x) => s + x.count, 0)
  return [...head.map((c) => ({ name: c.label, value: c.count })), { name: 'Other', value: otherSum }]
}

export default function DataSummaryPage() {
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()
  const [interval, setInterval] = useState('1d')
  const [bucket, setBucket] = useState<DashboardBucketParam>('auto')

  const chartMuted =
    colorScheme === 'dark' ? theme.colors.dark[2] : theme.colors.gray[6]
  const chartGrid =
    colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[3]
  const borderColor =
    colorScheme === 'dark' ? theme.other.darkBorder : theme.colors.gray[4]
  const textColor = colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.dark[9]
  const strongColor =
    colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.dark[9]

  const chartAxisStyle = useMemo(
    () => ({ fontSize: 11, fill: chartMuted }),
    [chartMuted],
  )

  const tooltipStyle = useMemo(
    () => ({
      background: colorScheme === 'dark' ? theme.other.darkPanelBg : theme.white,
      border: `1px solid ${borderColor}`,
      borderRadius: theme.defaultRadius,
      color: textColor,
      fontSize: '0.8125rem',
    }),
    [borderColor, colorScheme, textColor, theme.defaultRadius, theme.other.darkPanelBg, theme.white],
  )

  const piePalette = useMemo(
    () => [
      theme.colors.yellow[6],
      theme.colors.yellow[4],
      theme.colors.teal[6],
      borderColor,
      chartMuted,
      theme.colors.dark[4],
      theme.colors.red[4],
      theme.colors.gray[5],
      theme.colors.teal[3],
      theme.colors.red[3],
      theme.colors.dark[3],
      theme.colors.gray[6],
    ],
    [borderColor, chartMuted, theme.colors],
  )

  const query = useQuery({
    queryKey: ['dataDashboard', interval, bucket],
    queryFn: () => getDataDashboard({ interval, bucket }),
  })

  const data = query.data
  const tickers = data?.tickers
  const completenessHistogram = data?.completeness_histogram

  const scatterPoints = useMemo(() => {
    if (!tickers) return []
    const out: {
      ticker: string
      x: number
      y: number
      simpleReturnPct: number | null
      sharpe: number | null
      bars: number
    }[] = []
    for (const t of tickers) {
      if (t.risk_ann_pct == null || !Number.isFinite(t.risk_ann_pct)) continue
      const logRet = t.log_return_pct
      if (logRet == null || !Number.isFinite(logRet)) continue
      out.push({
        ticker: t.ticker,
        x: t.risk_ann_pct,
        y: logRet,
        simpleReturnPct: t.return_pct,
        sharpe: t.sharpe,
        bars: t.raw_bar_count,
      })
    }
    return out
  }, [tickers])

  const histogramBars = useMemo(() => {
    if (!completenessHistogram) return []
    return completenessHistogram.map((count, i) => ({
      label: `${i * 10}–${i === 9 ? '100' : String((i + 1) * 10)}`,
      count,
    }))
  }, [completenessHistogram])

  const accentStroke = theme.colors.yellow[colorScheme === 'dark' ? 4 : 6]

  return (
    <PageScaffold>
      <Anchor component={Link} to="/" size="sm">
        ← Home
      </Anchor>

      <Stack gap="xs">
        <Title order={1}>Data integrity & analytics</Title>
        <Text c="dimmed" size="sm">
          Coverage across the database reference window (global span for this interval and source),
          plus risk versus log total return from stored closes. Refresh pulls the latest Timescale snapshot.
        </Text>
      </Stack>

      <Group align="flex-end" wrap="wrap">
        <Select
          label="Interval"
          value={interval}
          onChange={(v) => v && setInterval(v)}
          data={INTERVAL_OPTIONS.map((iv) => ({ value: iv, label: iv }))}
          w={120}
        />
        <Select
          label="Coverage buckets"
          value={bucket}
          onChange={(v) => v && setBucket(v as DashboardBucketParam)}
          data={BUCKET_OPTIONS.map((b) => ({ value: b, label: b }))}
          w={140}
        />
        <Button
          color="yellow"
          onClick={() => query.refetch()}
          disabled={query.isFetching}
        >
          {query.isFetching ? 'Loading…' : 'Refresh'}
        </Button>
      </Group>

      <ApiErrorAlert error={query.error} />

      {query.isLoading && (
        <Text c="dimmed" size="sm">
          Loading dashboard…
        </Text>
      )}

      {data && (
        <>
          <SimpleGrid cols={{ base: 1, xs: 2, md: 3, lg: 5 }} spacing="md">
            <KpiCard
              title="Reference window"
              value={`${data.reference_start.slice(0, 10)} → ${data.reference_end.slice(0, 10)}`}
            />
            <KpiCard
              title="Tickers"
              value={String(data.ticker_count)}
              hint={data.truncated ? 'truncated (see env)' : undefined}
            />
            <KpiCard
              title="Mean completeness"
              value={`${data.aggregate_mean_completeness_pct.toFixed(1)}%`}
            />
            <KpiCard
              title="Median completeness"
              value={`${data.aggregate_median_completeness_pct.toFixed(1)}%`}
            />
            <KpiCard
              title="Coverage time buckets"
              value={`${data.bucket_count} (${data.bucket_granularity}${data.bucket_auto_subsampled ? ', merged' : ''})`}
            />
          </SimpleGrid>

          <Text c="dimmed" size="xs" maw={832}>
            Metrics use consecutive stored closes in the reference window (same semantics as{' '}
            <Code>POST /data</Code>). Gaps reduce completeness but do not insert synthetic bars.
            Annualized volatility and Sharpe use bar cadence <strong>{data.bar_unit}</strong> step{' '}
            <strong>{data.bar_step}</strong> (~{data.periods_per_year.toFixed(0)} periods/year).
          </Text>

          <Stack gap="sm">
            <Title order={2} size="h4">
              Classifications
            </Title>
            <Text c="dimmed" size="xs">
              Ticker counts by latest yfinance classification row per symbol (same universe as the dashboard).
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <ClassificationPiePanel
                title="Sector"
                counts={data.sector_counts ?? []}
                piePalette={piePalette}
                tooltipStyle={tooltipStyle}
                chartMuted={chartMuted}
              />
              <ClassificationPiePanel
                title="Industry"
                counts={data.industry_counts ?? []}
                piePalette={piePalette}
                tooltipStyle={tooltipStyle}
                chartMuted={chartMuted}
              />
            </SimpleGrid>
          </Stack>

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md">
            <Paper withBorder p="md" radius="md">
              <Text fw={600} size="sm" mb="sm">
                Risk vs log return
              </Text>
              <div style={{ width: '100%', height: 280 }}>
                <ResponsiveContainer>
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                    <CartesianGrid stroke={chartGrid} strokeDasharray="3 3" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="Vol"
                      unit="%"
                      tick={chartAxisStyle}
                      label={{
                        value: 'Annualized volatility %',
                        position: 'bottom',
                        offset: 0,
                        fill: chartMuted,
                        fontSize: 11,
                      }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="Log return"
                      unit="%"
                      tick={chartAxisStyle}
                      label={{
                        value: 'Log total return %',
                        angle: -90,
                        position: 'insideLeft',
                        fill: chartMuted,
                        fontSize: 11,
                      }}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null
                        const p = payload[0].payload as (typeof scatterPoints)[0]
                        return (
                          <div style={{ ...tooltipStyle, padding: '0.35rem 0.5rem' }}>
                            <Text fw={600} size="sm" c={strongColor}>
                              {p.ticker}
                            </Text>
                            <Text size="xs">Log return: {p.y.toFixed(2)}%</Text>
                            <Text size="xs">
                              Simple return:{' '}
                              {p.simpleReturnPct != null && Number.isFinite(p.simpleReturnPct)
                                ? `${p.simpleReturnPct.toFixed(2)}%`
                                : '—'}
                            </Text>
                            <Text size="xs">Vol: {p.x.toFixed(2)}%</Text>
                            <Text size="xs">
                              Sharpe: {p.sharpe != null ? p.sharpe.toFixed(2) : '—'}
                            </Text>
                            <Text size="xs">Bars: {p.bars ?? '—'}</Text>
                          </div>
                        )
                      }}
                    />
                    <Scatter
                      name="Tickers"
                      data={scatterPoints}
                      fill={accentStroke}
                      isAnimationActive={false}
                      shape={(raw: unknown) => {
                        const p = raw as { cx?: number; cy?: number }
                        if (p.cx == null || p.cy == null) return <g />
                        return (
                          <circle
                            cx={p.cx}
                            cy={p.cy}
                            r={3}
                            fill={accentStroke}
                            stroke={accentStroke}
                            strokeWidth={1}
                            fillOpacity={0.88}
                          />
                        )
                      }}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </Paper>

            <Paper withBorder p="md" radius="md">
              <Text fw={600} size="sm" mb="sm">
                Completeness distribution
              </Text>
              <div style={{ width: '100%', height: 280 }}>
                <ResponsiveContainer>
                  <BarChart data={histogramBars} margin={{ top: 8, right: 8, bottom: 40, left: 8 }}>
                    <CartesianGrid stroke={chartGrid} strokeDasharray="3 3" />
                    <XAxis
                      dataKey="label"
                      tick={chartAxisStyle}
                      interval={0}
                      angle={-35}
                      textAnchor="end"
                      height={60}
                    />
                    <YAxis tick={chartAxisStyle} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar
                      dataKey="count"
                      fill={theme.colors.yellow[4]}
                      stroke={accentStroke}
                      radius={[2, 2, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <Text c="dimmed" size="xs" px="xs" pb="xs" mt={4}>
                Count of tickers by completeness % bucket (full span of coverage time buckets).
              </Text>
            </Paper>
          </SimpleGrid>

          <MissingCoveragePiePanel
            tickers={data.tickers}
            piePalette={piePalette}
            tooltipStyle={tooltipStyle}
            chartMuted={chartMuted}
            strongColor={strongColor}
          />
        </>
      )}
    </PageScaffold>
  )
}

type MissingCoverageSlice = { name: string; value: number; completeness: number }

function MissingCoveragePiePanel({
  tickers,
  piePalette,
  tooltipStyle,
  chartMuted,
  strongColor,
}: {
  tickers: TickerDashboardRow[]
  piePalette: string[]
  tooltipStyle: CSSProperties
  chartMuted: string
  strongColor: string
}) {
  const pieData = useMemo((): MissingCoverageSlice[] => {
    if (!tickers.length) return []
    const rows: MissingCoverageSlice[] = tickers.map((t) => ({
      name: t.ticker,
      value: Math.max(0, 100 - t.completeness_pct),
      completeness: t.completeness_pct,
    }))
    rows.sort((a, b) => b.value - a.value || a.name.localeCompare(b.name))
    return rows.slice(0, MISSING_PIE_TOP_N).filter((r) => r.value > 0)
  }, [tickers])

  const total = useMemo(() => pieData.reduce((s, x) => s + x.value, 0), [pieData])

  if (!tickers.length) {
    return (
      <Paper withBorder p="md" radius="md">
        <Text fw={600} size="sm" mb="xs">
          Missing coverage (top {MISSING_PIE_TOP_N})
        </Text>
        <Text c="dimmed" size="sm">
          No tickers in this dashboard.
        </Text>
      </Paper>
    )
  }

  if (!pieData.length) {
    return (
      <Paper withBorder p="md" radius="md">
        <Text fw={600} size="sm" mb="xs">
          Missing coverage (top {MISSING_PIE_TOP_N})
        </Text>
        <Text c="dimmed" size="sm">
          All tickers are fully complete in this window (0% inverted coverage).
        </Text>
      </Paper>
    )
  }

  return (
    <Paper withBorder p="md" radius="md">
      <Text fw={600} size="sm" mb="xs">
        Missing coverage (top {MISSING_PIE_TOP_N})
      </Text>
      <Text c="dimmed" size="xs" mb="sm">
        Inverted coverage is 100% minus completeness (share of bars missing versus the reference grid).
        Slice size is relative among these tickers only; the legend lists up to {MISSING_PIE_TOP_N}{' '}
        symbols with the largest gaps.
      </Text>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="42%"
              cy="50%"
              outerRadius={100}
              paddingAngle={1}
              stroke={piePalette[3]}
              strokeWidth={1}
              isAnimationActive={false}
            >
              {pieData.map((entry, i) => (
                <Cell key={`${entry.name}-${i}`} fill={piePalette[i % piePalette.length]} />
              ))}
            </Pie>
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null
                const row = payload[0].payload as MissingCoverageSlice
                const sharePct = total > 0 ? (row.value / total) * 100 : 0
                return (
                  <div style={{ ...tooltipStyle, padding: '0.35rem 0.5rem' }}>
                    <Text fw={600} size="sm" c={strongColor}>
                      {row.name}
                    </Text>
                    <Text size="xs">Missing (inverted): {row.value.toFixed(2)}%</Text>
                    <Text size="xs">Completeness: {row.completeness.toFixed(2)}%</Text>
                    {total > 0 ? (
                      <Text size="xs">Share among legend set: {sharePct.toFixed(1)}%</Text>
                    ) : null}
                  </div>
                )
              }}
            />
            <Legend
              layout="vertical"
              verticalAlign="middle"
              align="right"
              wrapperStyle={{ fontSize: '0.72rem', lineHeight: 1.35 }}
              formatter={(value) => <span style={{ color: chartMuted }}>{value}</span>}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Paper>
  )
}

function ClassificationPiePanel({
  title,
  counts,
  piePalette,
  tooltipStyle,
  chartMuted,
}: {
  title: string
  counts: ClassificationLabelCount[]
  piePalette: string[]
  tooltipStyle: CSSProperties
  chartMuted: string
}) {
  const pieData = useMemo(() => collapseClassificationCounts(counts), [counts])
  const total = useMemo(() => pieData.reduce((s, x) => s + x.value, 0), [pieData])

  if (!pieData.length) {
    return (
      <Paper withBorder p="md" radius="md">
        <Text fw={600} size="sm" mb="xs">
          {title}
        </Text>
        <Text c="dimmed" size="sm">
          No classification rows for this universe.
        </Text>
      </Paper>
    )
  }

  return (
    <Paper withBorder p="md" radius="md">
      <Text fw={600} size="sm" mb="xs">
        {title}
      </Text>
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={78}
              paddingAngle={1}
              stroke={piePalette[3]}
              strokeWidth={1}
              isAnimationActive={false}
            >
              {pieData.map((entry, i) => (
                <Cell key={`${entry.name}-${i}`} fill={piePalette[i % piePalette.length]} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value: number) =>
                total > 0
                  ? [`${value} (${((value / total) * 100).toFixed(1)}%)`, 'Tickers']
                  : [`${value}`, 'Tickers']
              }
            />
            <Legend
              wrapperStyle={{ fontSize: '0.72rem' }}
              formatter={(value) => (
                <span style={{ color: chartMuted }}>{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Paper>
  )
}

function KpiCard({ title, value, hint }: { title: string; value: string; hint?: string }) {
  return (
    <Paper withBorder p="sm" radius="md">
      <Text c="dimmed" size="xs" mb={4}>
        {title}
      </Text>
      <Text fw={600} size="sm">
        {value}
      </Text>
      {hint && (
        <Text c="dimmed" size="xs" mt={4}>
          {hint}
        </Text>
      )}
    </Paper>
  )
}
