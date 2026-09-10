import {
  Alert,
  Badge,
  Box,
  Button,
  Grid,
  Group,
  Paper,
  Progress,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { getOmsOrder, getOmsOrderFills } from '../api/endpoints'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { useTradeDeskToken } from '../hooks/useTradeDeskToken'
import { LiveL1Provider, useLiveL1 } from '../live/l1Store'
import { touchRecentParent } from '../lib/tradeDeskStore'

function parseChildClientOrderId(clientOrderId: string | null | undefined): { slice: number; urgency: number } | null {
  if (!clientOrderId) return null
  const base = clientOrderId.split('u')[0]
  if (!base || !base.includes(':')) return null
  const [, sliceStr] = base.split(':')
  const slice = Number(sliceStr)
  if (!Number.isFinite(slice)) return null
  const urgencyMatch = clientOrderId.match(/u(\d+)$/)
  return { slice, urgency: urgencyMatch ? Number(urgencyMatch[1]) : 0 }
}

function L1MicrostructurePanel({ symbol }: { symbol: string }) {
  const { state } = useLiveL1()
  const lastQuote = state.quotes[state.quotes.length - 1]
  const bid = lastQuote?.bid_price
  const ask = lastQuote?.ask_price
  const mid = bid != null && ask != null ? (bid + ask) / 2 : null

  return (
    <Paper withBorder p="md" radius="md" h="100%">
      <Text fw={600} size="sm" mb="xs">
        Market microstructure (L1) — {symbol}
      </Text>
      {state.phase !== 'live' && (
        <Text size="xs" c="dimmed">
          {state.phase === 'connecting' ? 'Connecting…' : 'Not connected.'}
        </Text>
      )}
      {bid != null && ask != null && mid != null && (
        <>
          <Group grow align="stretch" gap="md" wrap="wrap">
            <Stack gap={4}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
                Bid
              </Text>
              <Text ff="monospace" fw={700} fz="xl" c="teal" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {bid.toFixed(2)}
              </Text>
            </Stack>
            <Stack gap={4}>
              <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
                Ask
              </Text>
              <Text ff="monospace" fw={700} fz="xl" c="red" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {ask.toFixed(2)}
              </Text>
              <Text size="xs" c="dimmed">
                Spread {(ask - bid).toFixed(3)} · mid {mid.toFixed(3)}
              </Text>
            </Stack>
          </Group>
          <Box mt="md" h={6} style={{ borderRadius: 4, background: 'var(--mantine-color-dark-5)' }}>
            <Box
              h="100%"
              w={`${Math.min(100, Math.max(0, ((mid - bid) / (ask - bid || 1)) * 100))}%`}
              style={{ borderRadius: 4, background: 'var(--mantine-color-yellow-5)' }}
            />
          </Box>
        </>
      )}
      {state.lastError && (
        <Text size="xs" c="red" mt="xs">
          {state.lastError}
        </Text>
      )}
    </Paper>
  )
}

function L1Panel({ symbol }: { symbol: string }) {
  const [connected, setConnected] = useState(false)
  return (
    <LiveL1Provider symbol={symbol} streamActive={connected}>
      <Stack gap="xs">
        <Group justify="flex-end">
          <Button size="compact-xs" variant={connected ? 'default' : 'light'} onClick={() => setConnected((v) => !v)}>
            {connected ? 'Disconnect' : 'Connect'}
          </Button>
        </Group>
        <L1MicrostructurePanel symbol={symbol} />
      </Stack>
    </LiveL1Provider>
  )
}

export default function ExecutionTracerPage() {
  const { parentId: raw } = useParams<{ parentId: string }>()
  const parentId = raw ? decodeURIComponent(raw) : ''
  const density = useMantineTableDensity()
  const { appliedToken } = useTradeDeskToken()

  useEffect(() => {
    if (parentId) touchRecentParent(parentId)
  }, [parentId])

  const orderQ = useQuery({
    queryKey: ['oms', 'order', parentId, appliedToken],
    queryFn: () => getOmsOrder(appliedToken, parentId),
    enabled: Boolean(parentId) && appliedToken.length > 0,
    refetchInterval: 10_000,
  })

  const fillsQ = useQuery({
    queryKey: ['oms', 'orderFills', parentId, appliedToken],
    queryFn: () => getOmsOrderFills(appliedToken, parentId),
    enabled: Boolean(parentId) && appliedToken.length > 0,
    refetchInterval: 10_000,
  })

  const childRows = useMemo(() => {
    const fills = fillsQ.data?.fills ?? []
    return fills
      .map((f) => ({ fill: f, meta: parseChildClientOrderId(f.child_client_order_id) }))
      .sort((a, b) => (a.meta?.slice ?? 0) - (b.meta?.slice ?? 0))
  }, [fillsQ.data])

  if (!parentId) {
    return (
      <PageScaffold>
        <Text c="dimmed">Missing parent id.</Text>
      </PageScaffold>
    )
  }

  if (!appliedToken) {
    return (
      <PageScaffold>
        <Title order={1}>EMS slicer</Title>
        <Alert color="yellow" title="Trade-desk token required" mt="md">
          Apply your trade-desk token on the{' '}
          <Text component={Link} to="/trade/blotter" span fw={600}>
            Blotter
          </Text>{' '}
          page to view real order/fill data here.
        </Alert>
      </PageScaffold>
    )
  }

  const order = orderQ.data
  const fillPct = order && order.quantity_ordered > 0 ? order.quantity_filled / order.quantity_ordered : 0
  const workingShares = order ? order.quantity_ordered - order.quantity_filled : 0

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Button component={Link} to="/execution" variant="default" size="compact-sm" mb="sm">
        ← Execution hub
      </Button>
      <Title order={1}>EMS slicer</Title>
      <Text ff="monospace" c="dimmed" size="sm">
        parent {parentId}
      </Text>

      <ApiErrorAlert error={orderQ.error} />
      <ApiErrorAlert error={fillsQ.error} />

      {order && (
        <Grid gap="md" mt="md" align="stretch">
          <Grid.Col span={{ base: 12, md: 4 }}>
            <Paper withBorder p="md" radius="md" h="100%">
              <Group justify="space-between" mb="xs">
                <Text fw={600} size="sm">
                  Fill progress
                </Text>
                <Badge color={order.state === 'FILLED' ? 'teal' : 'blue'} variant="light">
                  {order.state}
                </Badge>
              </Group>
              <Stack gap="sm">
                <div>
                  <Text size="xs" mb={4}>
                    Filled {(fillPct * 100).toFixed(0)}% ({order.quantity_filled} / {order.quantity_ordered})
                  </Text>
                  <Progress value={fillPct * 100} size="sm" color="teal" />
                </div>
                <Text size="xs" c="dimmed">
                  Working shares: {workingShares}
                </Text>
                <Text size="xs" c="dimmed">
                  Avg fill price: {order.avg_fill_price != null ? order.avg_fill_price.toFixed(2) : '—'}
                </Text>
              </Stack>
            </Paper>
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 8 }}>
            <L1Panel symbol={order.symbol} />
          </Grid.Col>
        </Grid>
      )}

      <Title order={3} size="h4" mt="lg">
        Fills (child orders)
      </Title>
      <Table.ScrollContainer minWidth={520}>
        <Table {...density} striped withTableBorder>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Child cid</Table.Th>
              <Table.Th ta="right">Slice</Table.Th>
              <Table.Th ta="right">Qty</Table.Th>
              <Table.Th ta="right">Price</Table.Th>
              <Table.Th>Time</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {childRows.map(({ fill, meta }) => (
              <Table.Tr key={fill.trade_id}>
                <Table.Td ff="monospace">{fill.child_client_order_id ?? '—'}</Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {meta ? meta.slice : '—'}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {fill.quantity}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {fill.price.toFixed(2)}
                </Table.Td>
                <Table.Td>{new Date(fill.ts).toLocaleTimeString()}</Table.Td>
              </Table.Tr>
            ))}
            {childRows.length === 0 && !fillsQ.isLoading && (
              <Table.Tr>
                <Table.Td colSpan={5}>
                  <Text size="sm" c="dimmed">
                    No fills yet.
                  </Text>
                </Table.Td>
              </Table.Tr>
            )}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </PageScaffold>
  )
}
