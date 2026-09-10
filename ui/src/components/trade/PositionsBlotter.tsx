import { Table, Text } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getOmsPositions } from '../../api/endpoints'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'

function fmtUsd(v: number | null | undefined): string {
  if (v == null) return '—'
  return v.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 })
}

function fmtPct(v: number | null | undefined): string {
  if (v == null) return '—'
  return `${(v * 100).toFixed(2)}%`
}

export default function PositionsBlotter({ tradeDeskToken }: { tradeDeskToken: string }) {
  const density = useMantineTableDensity()

  const positionsQ = useQuery({
    queryKey: ['oms', 'positions', tradeDeskToken],
    queryFn: () => getOmsPositions(tradeDeskToken),
    enabled: tradeDeskToken.length > 0,
    refetchInterval: 10_000,
    staleTime: 5_000,
  })

  if (!tradeDeskToken) {
    return (
      <Text size="sm" c="dimmed">
        Apply a trade-desk token to view positions.
      </Text>
    )
  }

  const rows = positionsQ.data?.positions ?? []

  return (
    <Table.ScrollContainer minWidth={720}>
      <Table {...density} striped withTableBorder stickyHeader>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Symbol</Table.Th>
            <Table.Th ta="right">Qty</Table.Th>
            <Table.Th ta="right">Avg entry</Table.Th>
            <Table.Th ta="right">Last</Table.Th>
            <Table.Th ta="right">Mkt value</Table.Th>
            <Table.Th ta="right">Unrealized P&amp;L</Table.Th>
            <Table.Th ta="right">Working buy</Table.Th>
            <Table.Th ta="right">Working sell</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((p) => (
            <Table.Tr key={p.symbol}>
              <Table.Td>{p.symbol}</Table.Td>
              <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {p.qty}
              </Table.Td>
              <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmtUsd(p.avg_entry_price)}
              </Table.Td>
              <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmtUsd(p.current_price)}
              </Table.Td>
              <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {fmtUsd(p.market_value)}
              </Table.Td>
              <Table.Td
                ta="right"
                ff="monospace"
                c={(p.unrealized_pl ?? 0) >= 0 ? 'teal' : 'red'}
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {fmtUsd(p.unrealized_pl)} ({fmtPct(p.unrealized_plpc)})
              </Table.Td>
              <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {p.working_buy_shares || '—'}
              </Table.Td>
              <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {p.working_sell_shares || '—'}
              </Table.Td>
            </Table.Tr>
          ))}
          {rows.length === 0 && !positionsQ.isLoading && (
            <Table.Tr>
              <Table.Td colSpan={8}>
                <Text size="sm" c="dimmed">
                  No open positions.
                </Text>
              </Table.Td>
            </Table.Tr>
          )}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  )
}
