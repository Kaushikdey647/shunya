import {
  Anchor,
  Card,
  SegmentedControl,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { getMarketMovers, instrumentDetailPath } from '../../api/endpoints'
import type { MoversKind } from '../../api/types'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'
import ApiErrorAlert from '../ApiErrorAlert'
import { TableRowsSkeleton } from './homeDashboardSkeletons'
import { SignedPctText } from '../../lib/signedPct'

const TABS: { id: MoversKind; label: string }[] = [
  { id: 'gainers', label: 'Top gainers' },
  { id: 'losers', label: 'Top losers' },
  { id: 'active', label: 'Most active' },
]

function fmtVol(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`
  return String(Math.round(v))
}

export default function MoversPanel() {
  const [tab, setTab] = useState<MoversKind>('gainers')
  const tableProps = useMantineTableDensity()
  const q = useQuery({
    queryKey: ['market', 'movers', tab],
    queryFn: () => getMarketMovers({ kind: tab, limit: 25 }),
    staleTime: 90_000,
  })

  return (
    <Card padding="md" radius="md" withBorder>
      <Stack gap="md">
        <Title order={5}>Movers</Title>
        <SegmentedControl
          fullWidth
          value={tab}
          onChange={(v) => setTab(v as MoversKind)}
          data={TABS.map((t) => ({ label: t.label, value: t.id }))}
        />
        <ApiErrorAlert error={q.error} />
        {q.isLoading && (
          <TableRowsSkeleton
            tableProps={tableProps}
            columnCount={4}
            rowCount={8}
            headers={['Ticker', 'Price', '%', 'Volume']}
            minWidth={280}
            mih={200}
            mah={400}
          />
        )}
        {!q.isLoading && q.data && (
          <Table.ScrollContainer minWidth={280} mih={200} mah={400}>
            <Table {...tableProps}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Ticker</Table.Th>
                  <Table.Th>Price</Table.Th>
                  <Table.Th>%</Table.Th>
                  <Table.Th>Volume</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {q.data.rows.length === 0 ? (
                  <Table.Tr>
                    <Table.Td colSpan={4}>
                      <Text c="dimmed" size="sm">
                        No rows
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ) : (
                  q.data.rows.map((r) => (
                    <Table.Tr key={r.ticker}>
                      <Table.Td>
                        <Anchor component={Link} to={instrumentDetailPath(r.ticker)} ff="monospace" size="sm">
                          {r.ticker}
                        </Anchor>
                      </Table.Td>
                      <Table.Td>
                        <Text ff="monospace" size="sm">
                          {r.price != null && Number.isFinite(r.price)
                            ? r.price.toLocaleString(undefined, { maximumFractionDigits: 2 })
                            : '—'}
                        </Text>
                      </Table.Td>
                      <Table.Td>
                        <SignedPctText v={r.pct_change} />
                      </Table.Td>
                      <Table.Td>
                        <Text ff="monospace" size="sm">
                          {fmtVol(r.volume)}
                        </Text>
                      </Table.Td>
                    </Table.Tr>
                  ))
                )}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        )}
      </Stack>
    </Card>
  )
}
