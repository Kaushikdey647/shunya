import { Button, Group, Stack, Text, TextInput, Title } from '@mantine/core'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import PageScaffold from '../components/PageScaffold'
import { useTradeDesk } from '../hooks/useTradeDesk'
import { touchRecentParent } from '../lib/tradeDeskStore'

export default function ExecutionHubPage() {
  const desk = useTradeDesk()
  const navigate = useNavigate()
  const [parentId, setParentId] = useState('')

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
      <Title order={3} size="h4" mt="xl">
        Recent parents
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
