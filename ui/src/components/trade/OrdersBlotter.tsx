import { Badge, Button, Table, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import { cancelOmsOrder, getOmsOrders } from '../../api/endpoints'
import { ApiError } from '../../api/client'
import type { OmsParentOrderOut } from '../../api/types'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'
import { useOmsStream } from '../../live/omsStore'

const TERMINAL_STATES = new Set(['FILLED', 'CANCELED', 'REJECTED'])

function stateColor(state: string): string {
  if (state === 'FILLED') return 'teal'
  if (state === 'PARTIAL_FILL') return 'yellow'
  if (state === 'WORKING') return 'blue'
  if (state === 'CANCELED' || state === 'REJECTED') return 'red'
  return 'gray'
}

export default function OrdersBlotter({ tradeDeskToken }: { tradeDeskToken: string }) {
  const density = useMantineTableDensity()
  const queryClient = useQueryClient()
  const { state: omsState } = useOmsStream()

  const ordersQ = useQuery({
    queryKey: ['oms', 'orders', tradeDeskToken],
    queryFn: () => getOmsOrders(tradeDeskToken, { state: 'all', limit: 200 }),
    enabled: tradeDeskToken.length > 0,
    refetchInterval: 15_000,
    staleTime: 5_000,
  })

  const cancelMutation = useMutation({
    mutationFn: (parentId: string) => cancelOmsOrder(tradeDeskToken, parentId),
    onSuccess: () => {
      notifications.show({ color: 'blue', title: 'Cancel requested', message: 'Awaiting broker confirmation.' })
      void queryClient.invalidateQueries({ queryKey: ['oms', 'orders'] })
    },
    onError: (err) => {
      const message = err instanceof ApiError ? err.message : 'Cancel request failed.'
      notifications.show({ color: 'red', title: 'Cancel failed', message })
    },
  })

  const rows = useMemo(() => {
    const base = ordersQ.data?.orders ?? []
    const byId = new Map<string, OmsParentOrderOut>(base.map((o) => [o.parent_id, o]))
    for (const o of Object.values(omsState.orders)) {
      byId.set(o.parent_id, { ...byId.get(o.parent_id), ...o })
    }
    return Array.from(byId.values()).sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
  }, [ordersQ.data, omsState.orders])

  if (!tradeDeskToken) {
    return (
      <Text size="sm" c="dimmed">
        Apply a trade-desk token to view orders.
      </Text>
    )
  }

  return (
    <>
      <Text size="xs" c="dimmed" mb="xs">
        Shows orders placed from this desk. All broker positions (including orders placed elsewhere) appear
        under Positions.
      </Text>
      <Table.ScrollContainer minWidth={720}>
        <Table {...density} striped withTableBorder stickyHeader>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Time</Table.Th>
              <Table.Th>Symbol</Table.Th>
              <Table.Th>Side</Table.Th>
              <Table.Th>Type</Table.Th>
              <Table.Th ta="right">Qty</Table.Th>
              <Table.Th ta="right">Filled</Table.Th>
              <Table.Th ta="right">Limit</Table.Th>
              <Table.Th ta="right">Avg fill</Table.Th>
              <Table.Th>State</Table.Th>
              <Table.Th />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.map((o) => (
              <Table.Tr key={o.parent_id}>
                <Table.Td style={{ whiteSpace: 'nowrap' }}>{o.created_at ?? '—'}</Table.Td>
                <Table.Td>
                  <Link to={`/execution/${encodeURIComponent(o.parent_id)}`}>{o.symbol}</Link>
                </Table.Td>
                <Table.Td c={o.side === 'BUY' ? 'teal' : 'red'} fw={600}>
                  {o.side}
                </Table.Td>
                <Table.Td>{o.order_type ?? '—'}</Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {o.quantity_ordered}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {o.quantity_filled}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {o.limit_price != null ? o.limit_price.toFixed(2) : '—'}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {o.avg_fill_price != null ? o.avg_fill_price.toFixed(2) : '—'}
                </Table.Td>
                <Table.Td>
                  <Badge color={stateColor(o.state)} variant="light">
                    {o.state}
                  </Badge>
                </Table.Td>
                <Table.Td>
                  {!TERMINAL_STATES.has(o.state) && (
                    <Button
                      size="compact-xs"
                      variant="light"
                      color="red"
                      loading={cancelMutation.isPending && cancelMutation.variables === o.parent_id}
                      onClick={() => cancelMutation.mutate(o.parent_id)}
                    >
                      Cancel
                    </Button>
                  )}
                </Table.Td>
              </Table.Tr>
            ))}
            {rows.length === 0 && !ordersQ.isLoading && (
              <Table.Tr>
                <Table.Td colSpan={10}>
                  <Text size="sm" c="dimmed">
                    No orders yet.
                  </Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </>
  )
}
