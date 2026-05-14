import {
  Card,
  Group,
  SimpleGrid,
  Stack,
  Text,
  useComputedColorScheme,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { Line, LineChart, ResponsiveContainer, YAxis } from 'recharts'
import { postMarketSnapshot } from '../../api/endpoints'
import type { MarketSnapshotRow } from '../../api/types'
import ApiErrorAlert from '../ApiErrorAlert'
import { MacroStripSkeleton } from './homeDashboardSkeletons'
import { MACRO_STRIP_SYMBOLS } from '../../lib/macroSymbols'
import { SignedPctText } from '../../lib/signedPct'

function sparklineStroke(
  colorScheme: 'light' | 'dark',
  pct: number | null | undefined,
  first: number,
  last: number,
): string {
  const flat = pct === 0 || (pct == null && Math.abs(last - first) < 1e-9 * (Math.abs(last) || 1))
  if (flat) {
    return colorScheme === 'dark' ? '#00FFFF' : '#0891b2'
  }
  const up = pct != null ? pct > 0 : last > first
  if (up) {
    return colorScheme === 'dark' ? '#00FF00' : '#0d9488'
  }
  return colorScheme === 'dark' ? '#FF0000' : '#dc2626'
}

function MacroCard({ row }: { row: MarketSnapshotRow }) {
  const colorScheme = useComputedColorScheme('light', { getInitialValueInEffect: false })
  const scheme = colorScheme === 'dark' ? 'dark' : 'light'
  const closes = row.sparkline_close.filter((c) => Number.isFinite(c))
  const data = closes.map((close, i) => ({ i, close }))
  const pct = row.pct_change_1d
  const last = row.last
  const first = closes[0] ?? last ?? 0
  const lastClose = closes[closes.length - 1] ?? last ?? 0
  const stroke = sparklineStroke(scheme, pct ?? null, first, lastClose)

  let yDomain: [number, number] | undefined
  if (closes.length >= 1) {
    const minC = Math.min(...closes)
    const maxC = Math.max(...closes)
    const span = maxC - minC
    const pad = span > 0 ? span * 0.15 : Math.abs(minC) * 0.002 + 1e-6
    yDomain = [minC - pad, maxC + pad]
  }

  return (
    <Card padding="sm" radius={scheme === 'dark' ? 0 : 'md'} withBorder>
      <Card.Section inheritPadding pb="xs">
        <Text fw={600} ff="monospace" size="sm">
          {row.symbol}
        </Text>
        <Group justify="space-between" gap="xs" wrap="nowrap">
          <Text ff="monospace" size="sm" fw={500}>
            {last != null && Number.isFinite(last)
              ? last.toLocaleString(undefined, { maximumFractionDigits: 2 })
              : '—'}
          </Text>
          <SignedPctText v={pct} />
        </Group>
      </Card.Section>
      <div style={{ height: 52 }}>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height={52}>
            <LineChart data={data} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
              {yDomain && <YAxis domain={yDomain} hide width={0} />}
              <Line
                type="monotone"
                dataKey="close"
                stroke={stroke}
                strokeWidth={scheme === 'dark' ? 2 : 1.75}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <Text size="xs" c="dimmed" pt="sm">
            No series
          </Text>
        )}
      </div>
    </Card>
  )
}

export default function MacroStrip() {
  const symbols = [...MACRO_STRIP_SYMBOLS]
  const q = useQuery({
    queryKey: ['market', 'snapshot', 'macro', symbols.join(',')],
    queryFn: () => postMarketSnapshot({ symbols }),
    staleTime: 90_000,
  })

  const bySym = new Map((q.data?.rows ?? []).map((r) => [r.symbol, r]))

  return (
    <Stack gap="sm" aria-label="Macro overview">
      <ApiErrorAlert error={q.error} />
      {q.isLoading && <MacroStripSkeleton />}
      {!q.isLoading && (
        <>
          <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md">
            {symbols.map((sym) => {
              const row =
                bySym.get(sym) ??
                ({
                  symbol: sym,
                  last: null,
                  pct_change_1d: null,
                  volume: null,
                  sparkline_close: [],
                } satisfies MarketSnapshotRow)
              return <MacroCard key={sym} row={row} />
            })}
          </SimpleGrid>
          <Text size="xs" c="dimmed" px="xs">
            Sparklines: recent daily closes (typically 5–7 sessions).
          </Text>
        </>
      )}
    </Stack>
  )
}
