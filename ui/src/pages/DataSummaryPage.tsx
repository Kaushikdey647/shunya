import {
  Anchor,
  Button,
  Code,
  Group,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
  UnstyledButton,
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
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import type { HeatmapPalette } from '../lib/monthlyReturnsHeatmap'

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

type SortKey =
  | 'ticker'
  | 'completeness_pct'
  | 'longest_run_buckets'
  | 'return_pct'
  | 'risk_ann_pct'
  | 'sharpe'
  | 'raw_bar_count'

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

function formatPct(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  return `${n.toFixed(digits)}%`
}

function formatNum(n: number | null | undefined, digits = 2): string {
  if (n == null || Number.isNaN(n)) return '—'
  return n.toFixed(digits)
}

export default function DataSummaryPage() {
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()
  const density = useMantineTableDensity()
  const [interval, setInterval] = useState('1d')
  const [bucket, setBucket] = useState<DashboardBucketParam>('auto')
  const [sortKey, setSortKey] = useState<SortKey>('ticker')
  const [sortAsc, setSortAsc] = useState(true)

  const chartMuted =
    colorScheme === 'dark' ? theme.colors.dark[2] : theme.colors.gray[6]
  const chartGrid =
    colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[3]
  const stickyColumnBg =
    colorScheme === 'dark' ? theme.other.darkPanelBg : theme.white
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

  const heatmapPalette: HeatmapPalette = useMemo(
    () => ({
      success: theme.colors.teal[6]!,
      error: theme.colors.red[6]!,
      surface:
        colorScheme === 'dark'
          ? String(theme.other.darkPanelBg)
          : String(theme.white),
      empty:
        colorScheme === 'dark' ? theme.colors.dark[6]! : theme.colors.gray[2]!,
    }),
    [colorScheme, theme.colors.dark, theme.colors.gray, theme.colors.red, theme.colors.teal, theme.other.darkPanelBg, theme.white],
  )

  const query = useQuery({
    queryKey: ['dataDashboard', interval, bucket],
    queryFn: () => getDataDashboard({ interval, bucket }),
  })

  const data = query.data
  const tickers = data?.tickers
  const completenessHistogram = data?.completeness_histogram

  const sortedTickers = useMemo(() => {
    if (!tickers?.length) return []
    const rows = [...tickers]
    rows.sort((a, b) => {
      const dir = sortAsc ? 1 : -1
      const av = a[sortKey]
      const bv = b[sortKey]
      if (sortKey === 'ticker') {
        return dir * String(av).localeCompare(String(bv))
      }
      const an = typeof av === 'number' ? av : Number.NEGATIVE_INFINITY
      const bn = typeof bv === 'number' ? bv : Number.NEGATIVE_INFINITY
      if (an === bn) return 0
      return dir * (an < bn ? -1 : 1)
    })
    return rows
  }, [tickers, sortAsc, sortKey])

  const scatterPoints = useMemo(() => {
    if (!tickers) return []
    return tickers
      .filter(
        (t) =>
          t.risk_ann_pct != null &&
          t.return_pct != null &&
          Number.isFinite(t.risk_ann_pct) &&
          Number.isFinite(t.return_pct),
      )
      .map((t) => ({
        ticker: t.ticker,
        x: t.risk_ann_pct as number,
        y: t.return_pct as number,
        sharpe: t.sharpe,
        bars: t.raw_bar_count,
      }))
  }, [tickers])

  const histogramBars = useMemo(() => {
    if (!completenessHistogram) return []
    return completenessHistogram.map((count, i) => ({
      label: `${i * 10}–${i === 9 ? '100' : String((i + 1) * 10)}`,
      count,
    }))
  }, [completenessHistogram])

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(!sortAsc)
    else {
      setSortKey(key)
      setSortAsc(key === 'ticker')
    }
  }

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
          plus risk–return positioning from stored closes. Refresh pulls the latest Timescale snapshot.
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
              title="Heatmap columns"
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
              Ticker counts by latest yfinance classification row per symbol (same universe as the heatmap).
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
                Risk vs return
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
                      name="Return"
                      unit="%"
                      tick={chartAxisStyle}
                      label={{
                        value: 'Total return %',
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
                            <Text size="xs">Return: {p.y.toFixed(2)}%</Text>
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
                Count of tickers by completeness % bucket (full span of heatmap columns).
              </Text>
            </Paper>
          </SimpleGrid>

          <Stack gap="sm">
            <Title order={2} size="h4">
              Coverage heatmap
            </Title>
            <Text c="dimmed" size="sm">
              Rows are tickers; columns are time buckets (green = at least one bar). Axis uses{' '}
              <Code>{data.interval}</Code> / <Code>{data.source}</Code>.
            </Text>
            <Heatmap
              rows={sortedTickers}
              buckets={data.buckets}
              palette={heatmapPalette}
              tickBorder={chartGrid}
              stickyBg={stickyColumnBg}
            />
          </Stack>

          <Stack gap="sm">
            <Title order={2} size="h4">
              Instruments
            </Title>
            <Table.ScrollContainer minWidth={720}>
              <Table {...density} striped highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <SortTh
                      label="Ticker"
                      k="ticker"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <SortTh
                      label="Completeness %"
                      k="completeness_pct"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <SortTh
                      label="Longest run"
                      k="longest_run_buckets"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <SortTh
                      label="Bars"
                      k="raw_bar_count"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <SortTh
                      label="Return %"
                      k="return_pct"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <SortTh
                      label="Vol ann. %"
                      k="risk_ann_pct"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <SortTh
                      label="Sharpe"
                      k="sharpe"
                      active={sortKey}
                      asc={sortAsc}
                      onToggle={toggleSort}
                    />
                    <Table.Th c="dimmed">First bar</Table.Th>
                    <Table.Th c="dimmed">Last bar</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {sortedTickers.map((row) => (
                    <Table.Tr key={row.ticker}>
                      <Table.Td ff="monospace">{row.ticker}</Table.Td>
                      <Table.Td>{formatPct(row.completeness_pct)}</Table.Td>
                      <Table.Td>{row.longest_run_buckets}</Table.Td>
                      <Table.Td>{row.raw_bar_count}</Table.Td>
                      <Table.Td>{formatPct(row.return_pct)}</Table.Td>
                      <Table.Td>{formatPct(row.risk_ann_pct)}</Table.Td>
                      <Table.Td>{formatNum(row.sharpe, 3)}</Table.Td>
                      <Table.Td ff="monospace" c="dimmed">
                        {row.first_ts?.slice(0, 10) ?? '—'}
                      </Table.Td>
                      <Table.Td ff="monospace" c="dimmed">
                        {row.last_ts?.slice(0, 10) ?? '—'}
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          </Stack>
        </>
      )}
    </PageScaffold>
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

function SortTh({
  label,
  k,
  active,
  asc,
  onToggle,
}: {
  label: string
  k: SortKey
  active: SortKey
  asc: boolean
  onToggle: (k: SortKey) => void
}) {
  const on = active === k
  return (
    <Table.Th>
      <UnstyledButton onClick={() => onToggle(k)} fw={on ? 600 : 500}>
        <Text span c={on ? 'yellow' : 'dimmed'} size="sm">
          {label}
          {on ? (asc ? ' ↑' : ' ↓') : ''}
        </Text>
      </UnstyledButton>
    </Table.Th>
  )
}

function Heatmap({
  rows,
  buckets,
  palette,
  tickBorder,
  stickyBg,
}: {
  rows: TickerDashboardRow[]
  buckets: { index: number; start: string; end: string }[]
  palette: HeatmapPalette
  tickBorder: string
  stickyBg: string
}) {
  const n = buckets.length
  if (!n || !rows.length) {
    return (
      <Text c="dimmed" size="sm">
        No coverage rows.
      </Text>
    )
  }

  return (
    <Paper withBorder p="xs" radius="md" style={{ overflow: 'auto' }}>
      <div style={{ minWidth: `${88 + n * 3}px` }}>
        <div style={{ display: 'flex', gap: 2, marginBottom: 6, paddingLeft: 88 }}>
          {buckets.map((b, i) => (
            <span
              key={b.index}
              title={`${b.start} → ${b.end}`}
              style={{
                flex: 1,
                minWidth: 2,
                height: 4,
                background: i % 12 === 0 ? tickBorder : 'transparent',
              }}
            />
          ))}
        </div>
        {rows.map((row) => (
          <div
            key={row.ticker}
            style={{
              display: 'flex',
              gap: 2,
              alignItems: 'center',
              marginBottom: 2,
            }}
          >
            <span
              style={{
                width: 80,
                flexShrink: 0,
                fontSize: '0.75rem',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                position: 'sticky',
                left: 0,
                background: stickyBg,
                zIndex: 1,
                paddingRight: 6,
                fontFamily: 'var(--mantine-font-family-monospace)',
              }}
              title={row.ticker}
            >
              {row.ticker}
            </span>
            {row.coverage.map((c, i) => {
              const meta = buckets[i]
              const titleStr = meta ? `${meta.start.slice(0, 10)}→${meta.end.slice(0, 10)}` : ''
              return (
                <span
                  key={i}
                  title={titleStr}
                  style={{
                    flex: 1,
                    minWidth: 2,
                    height: 14,
                    borderRadius: 1,
                    background: c ? palette.success : palette.error,
                    opacity: c ? 0.88 : 0.28,
                  }}
                />
              )
            })}
          </div>
        ))}
      </div>
    </Paper>
  )
}
