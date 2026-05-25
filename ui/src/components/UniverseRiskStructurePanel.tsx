import {
  Anchor,
  Box,
  Card,
  SegmentedControl,
  SimpleGrid,
  Stack,
  Table,
  Tabs,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getUniverseReturnAnalytics, instrumentDetailPath } from '../api/endpoints'
import ApiErrorAlert from './ApiErrorAlert'

type PeriodOpt = '1y' | '2y' | '5y'

const CHART_GRID_PROPS = {
  stroke: 'rgba(255, 255, 255, 0.07)',
  strokeDasharray: '4 4' as const,
}

function corrCellBg(v: number): string {
  const r = Math.max(-1, Math.min(1, v))
  if (r >= 0) {
    const a = 0.12 + r * 0.55
    return `rgba(250, 82, 82, ${a})`
  }
  const a = 0.12 + -r * 0.55
  return `rgba(34, 139, 230, ${a})`
}

function CorrelationMatrix({
  tickers,
  matrix,
}: {
  tickers: string[]
  matrix: number[][]
}) {
  const n = tickers.length
  if (!n) {
    return <Text c="dimmed" size="sm">No correlation data.</Text>
  }
  return (
    <Box mx="auto" maw="100%" style={{ width: 'min(100%, min(85vw, 72dvh))' }}>
      <Text size="xs" c="dimmed" mb={6}>
        Square heatmap (no axis labels). Hover a cell for tickers and ρ.
      </Text>
      <Box
        style={{
          aspectRatio: 1,
          width: '100%',
          maxHeight: 'min(72dvh, 85vw)',
        }}
      >
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${n}, 1fr)`,
            gridTemplateRows: `repeat(${n}, 1fr)`,
            width: '100%',
            height: '100%',
            gap: 1,
            borderRadius: 4,
            overflow: 'hidden',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          {tickers.flatMap((row, i) =>
            tickers.map((col, j) => {
              const v = matrix[i]?.[j] ?? 0
              const bg = corrCellBg(v)
              return (
                <Tooltip key={`${i}-${j}`} label={`${row} vs ${col}: ${v.toFixed(3)}`} withArrow openDelay={200}>
                  <Box
                    aria-label={`Correlation ${row} ${col} ${v.toFixed(3)}`}
                    style={{
                      background: bg,
                      minWidth: 0,
                      minHeight: 0,
                      width: '100%',
                      height: '100%',
                    }}
                  />
                </Tooltip>
              )
            })
          )}
        </div>
      </Box>
    </Box>
  )
}

function PcaLoadingsTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: ReadonlyArray<{ payload?: { ticker?: string; pc1_loading?: number; pc2_loading?: number } }>
}) {
  if (!active || !payload?.length) return null
  const p = payload[0]?.payload
  if (!p?.ticker) return null
  return (
    <Box
      px="sm"
      py={6}
      style={{
        background: 'var(--mantine-color-body)',
        border: '1px solid var(--mantine-color-default-border)',
        borderRadius: 4,
        fontSize: 12,
        minWidth: 140,
      }}
    >
      <Text fw={600} ff="monospace" size="sm">
        {p.ticker}
      </Text>
      <Text size="xs" c="dimmed" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
        PC1 {Number(p.pc1_loading).toFixed(4)} · PC2 {Number(p.pc2_loading).toFixed(4)}
      </Text>
      <Text fz="xs" c="dimmed" mt={4} style={{ lineHeight: 1.35 }}>
        Sector labels require API support on loadings.{' '}
        <Anchor component={Link} to={instrumentDetailPath(p.ticker)} size="xs" c="yellow">
          Open instrument
        </Anchor>
      </Text>
    </Box>
  )
}

export default function UniverseRiskStructurePanel({ universeId }: { universeId: string }) {
  const [period, setPeriod] = useState<PeriodOpt>('1y')

  const analyticsQ = useQuery({
    queryKey: ['universe', universeId, 'return-analytics', period],
    queryFn: () =>
      getUniverseReturnAnalytics(universeId, {
        period,
        interval: '1d',
        source: 'yfinance',
      }),
  })

  const evrChart = useMemo(() => {
    const d = analyticsQ.data?.pca_explained_variance_ratio ?? []
    return d.map((v, i) => ({ name: `PC${i + 1}`, frac: v }))
  }, [analyticsQ.data])

  const xsChart = useMemo(() => {
    return (analyticsQ.data?.cross_sectional_vol ?? []).map((p) => ({
      date: p.date,
      xs_vol: p.xs_vol,
    }))
  }, [analyticsQ.data])

  return (
    <Stack gap="md">
      <GroupWithPeriod period={period} onPeriod={setPeriod} />
      <ApiErrorAlert error={analyticsQ.error} />
      {analyticsQ.isLoading && <Text c="dimmed" size="sm">Loading risk analytics…</Text>}
      {analyticsQ.data && (
        <Stack gap="lg">
          <Text size="xs" c="dimmed">
            Window {analyticsQ.data.start_date} → {analyticsQ.data.end_date_exclusive} ·{' '}
            {analyticsQ.data.n_observations} return days · {analyticsQ.data.tickers.length} names ·{' '}
            {analyticsQ.data.alignment}
          </Text>

          <Card
            padding="md"
            radius="md"
            withBorder={false}
            style={{ backgroundColor: 'var(--mantine-color-default)' }}
          >
            <Title order={4} size="h5" mb="sm">
              Return correlations
            </Title>
            <Tabs defaultValue="simple">
              <Tabs.List mb="sm">
                <Tabs.Tab value="simple">Simple returns</Tabs.Tab>
                <Tabs.Tab value="log">Log returns</Tabs.Tab>
              </Tabs.List>
              <Tabs.Panel value="simple">
                <CorrelationMatrix
                  tickers={analyticsQ.data.tickers}
                  matrix={analyticsQ.data.correlation_simple}
                />
              </Tabs.Panel>
              <Tabs.Panel value="log">
                <CorrelationMatrix
                  tickers={analyticsQ.data.tickers}
                  matrix={analyticsQ.data.correlation_log}
                />
              </Tabs.Panel>
            </Tabs>
          </Card>

          <Card
            padding="md"
            radius="md"
            withBorder={false}
            style={{ backgroundColor: 'var(--mantine-color-default)' }}
          >
            <Title order={4} size="h5" mb={4}>
              Cross-sectional volatility
            </Title>
            <Text size="xs" c="dimmed" mb="sm">
              Per day: sample standard deviation of simple returns across universe members (ddof=1).
            </Text>
            <div style={{ width: '100%', height: 280 }}>
              <ResponsiveContainer>
                <LineChart data={xsChart} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid {...CHART_GRID_PROPS} />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={24} />
                  <YAxis tick={{ fontSize: 10 }} width={48} />
                  <RechartsTooltip formatter={(v: number) => v.toFixed(4)} />
                  <Line type="stepAfter" dataKey="xs_vol" stroke="#228be6" dot={false} strokeWidth={1.5} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card
            padding="md"
            radius="md"
            withBorder={false}
            style={{ backgroundColor: 'var(--mantine-color-default)' }}
          >
            <Title order={4} size="h5" mb={4}>
              PCA (standardized simple returns)
            </Title>
            <Text size="xs" c="dimmed" mb="sm">
              SVD on the T×N standardized return matrix; scores are PC1 projection over time; loadings are variable weights.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer>
                  <BarChart data={evrChart} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
                    <CartesianGrid {...CHART_GRID_PROPS} />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} width={40} domain={[0, 1]} />
                    <RechartsTooltip formatter={(v: number) => (v * 100).toFixed(2) + '%'} />
                    <Bar dataKey="frac" fill="#845ef7" name="Variance share" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{ width: '100%', height: 260 }}>
                <ResponsiveContainer>
                  <LineChart
                    data={analyticsQ.data.pca_pc1_scores}
                    margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid {...CHART_GRID_PROPS} />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={28} />
                    <YAxis tick={{ fontSize: 10 }} width={44} />
                    <RechartsTooltip formatter={(v: number) => Number(v).toFixed(4)} />
                    <Line type="stepAfter" dataKey="score" stroke="#40c057" dot={false} strokeWidth={1.2} name="PC1 score" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </SimpleGrid>
            {analyticsQ.data.pca_loadings_scatter.length > 0 && (
              <Box mt="md">
                <Text size="xs" c="dimmed" mb={4}>
                  PC1 vs PC2 loadings (per ticker). Tooltip shows the symbol.
                </Text>
                <div style={{ width: '100%', height: 300 }}>
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                      <CartesianGrid {...CHART_GRID_PROPS} />
                      <XAxis type="number" dataKey="pc1_loading" name="PC1" tick={{ fontSize: 10 }} />
                      <YAxis type="number" dataKey="pc2_loading" name="PC2" tick={{ fontSize: 10 }} />
                      <RechartsTooltip
                        isAnimationActive={false}
                        cursor={{ strokeDasharray: '3 3' }}
                        content={(props) => (
                          <PcaLoadingsTooltip active={props.active} payload={props.payload} />
                        )}
                      />
                      <Scatter data={analyticsQ.data.pca_loadings_scatter} fill="#fd7e14" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </Box>
            )}
          </Card>

          <Card
            padding="md"
            radius="md"
            withBorder={false}
            style={{ backgroundColor: 'var(--mantine-color-default)' }}
          >
            <Title order={4} size="h5" mb="xs">
              Concentration (latest market cap weights)
            </Title>
            <Text size="xs" c="dimmed" mb="sm">
              HHI = Σw²; CR5 / CR10 = weight in top five / ten names. Falls back to equal weights if caps are missing.
            </Text>
            <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md" mb="md">
              <div>
                <Text size="xs" c="dimmed" tt="uppercase">HHI</Text>
                <Text fw={600} ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {analyticsQ.data.concentration.hhi.toFixed(4)}
                </Text>
              </div>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase">CR5</Text>
                <Text fw={600} ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {(analyticsQ.data.concentration.cr5 * 100).toFixed(1)}%
                </Text>
              </div>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase">CR10</Text>
                <Text fw={600} ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {(analyticsQ.data.concentration.cr10 * 100).toFixed(1)}%
                </Text>
              </div>
              <div>
                <Text size="xs" c="dimmed" tt="uppercase">Weights</Text>
                <Text fw={600} ff="monospace" size="sm">
                  {analyticsQ.data.concentration.weight_mode}
                  {analyticsQ.data.concentration.mcap_weights_partial ? ' (partial)' : ''}
                </Text>
              </div>
            </SimpleGrid>
            <Table.ScrollContainer minWidth={320} type="native" mb={4}>
              <Table striped highlightOnHover withTableBorder>
                <Table.Thead>
                  <Table.Tr>
                    {analyticsQ.data.concentration.top_holdings.map((h) => (
                      <Table.Th key={h.ticker} fz="xs" ff="monospace" ta="center" style={{ whiteSpace: 'nowrap' }}>
                        {h.ticker}
                      </Table.Th>
                    ))}
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    {analyticsQ.data.concentration.top_holdings.map((h) => (
                      <Table.Td key={h.ticker} fz="xs" ta="center" fw={500} ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {(h.weight * 100).toFixed(2)}%
                      </Table.Td>
                    ))}
                  </Table.Tr>
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          </Card>
        </Stack>
      )}
    </Stack>
  )
}

function GroupWithPeriod({
  period,
  onPeriod,
}: {
  period: PeriodOpt
  onPeriod: (p: PeriodOpt) => void
}) {
  return (
    <SegmentedControl
      value={period}
      onChange={(v) => onPeriod(v as PeriodOpt)}
      data={[
        { label: '1y', value: '1y' },
        { label: '2y', value: '2y' },
        { label: '5y', value: '5y' },
      ]}
    />
  )
}
