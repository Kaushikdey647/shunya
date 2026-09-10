import { Badge, Button, Group, Stack, Text, TextInput, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { getOmsOrders } from '../api/endpoints'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useTradeDesk } from '../hooks/useTradeDesk'
import { useTradeDeskToken } from '../hooks/useTradeDeskToken'
import { touchRecentParent } from '../lib/tradeDeskStore'

export default function ExecutionHubPage() {
  const desk = useTradeDesk()
  const navigate = useNavigate()
  const [parentId, setParentId] = useState('')
  const { appliedToken } = useTradeDeskToken()

  const ordersQ = useQuery({
    queryKey: ['oms', 'orders', appliedToken, 'recent'],
    queryFn: () => getOmsOrders(appliedToken, { limit: 20 }),
    enabled: appliedToken.length > 0,
  })

  return (
    <PageScaffold>
      <Title order={1}>Execution tracer</Title>
      <Text c="dimmed" size="sm">
        Drill into EMS slicer progress for a parent order. Paste a parent id from OMS logs or pick a recent run.
      </Text>
      <Stack gap="sm" maw={480}>
        <TextInput
          label="Parent order id"
          placeholder="e.g. demo-parent-vwap-1"
          value={parentId}
          onChange={(e) => setParentId(e.currentTarget.value)}
          ff="monospace"
          styles={{
            input: {
              fontVariantNumeric: 'tabular-nums',
              transition: 'border-color 120ms ease, box-shadow 120ms ease',
            },
          }}
        />
        <Button
          color="yellow"
          disabled={!parentId.trim()}
          onClick={() => {
            const id = parentId.trim()
            touchRecentParent(id)
            navigate(`/execution/${encodeURIComponent(id)}`)
          }}
        >
          Open tracer
        </Button>
      </Stack>
      <Text c="dimmed" size="sm" mt="md">
        Last portfolio union universe:{' '}
        <Text span fw={600}>
          {desk.lastPortfolioUniverseTickers.length
            ? `${desk.lastPortfolioUniverseTickers.length} tickers`
            : 'none recorded'}
        </Text>
        {desk.lastPortfolioUniverseNote ? (
          <>
            {' '}
            <Text span c="dimmed" size="xs">
              ({desk.lastPortfolioUniverseNote})
            </Text>
          </>
        ) : null}
      </Text>
      <ApiErrorAlert error={ordersQ.error} />

      <Title order={3} size="h4" mt="xl">
        Recent orders
      </Title>
      {!appliedToken && (
        <Text size="sm" c="dimmed">
          Apply a trade-desk token on the{' '}
          <Text component={Link} to="/trade/blotter" span fw={600}>
            Blotter
          </Text>{' '}
          page to list recent manually-placed orders here.
        </Text>
      )}
      <Stack gap="xs">
        {(ordersQ.data?.orders ?? []).map((o) => (
          <Group key={o.parent_id} justify="space-between" wrap="nowrap">
            <Group gap="xs" wrap="nowrap">
              <Text ff="monospace" size="sm">
                {o.symbol}
              </Text>
              <Text size="sm" c={o.side === 'BUY' ? 'teal' : 'red'}>
                {o.side}
              </Text>
              <Badge size="sm" variant="light">
                {o.state}
              </Badge>
            </Group>
            <Button
              component={Link}
              size="compact-xs"
              variant="light"
              to={`/execution/${encodeURIComponent(o.parent_id)}`}
            >
              Trace
            </Button>
          </Group>
        ))}
      </Stack>

      <Title order={3} size="h4" mt="xl">
        Recently viewed
      </Title>
      <Stack gap="xs">
        {desk.recentParentIds.map((pid) => (
          <Group key={pid} justify="space-between" wrap="nowrap">
            <Text ff="monospace" size="sm">
              {pid}
            </Text>
            <Button component={Link} size="compact-xs" variant="light" to={`/execution/${encodeURIComponent(pid)}`}>
              Trace
            </Button>
          </Group>
        ))}
      </Stack>
    </PageScaffold>
  )
}
