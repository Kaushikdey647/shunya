import {
  Box,
  Button,
  Group,
  Paper,
  Progress,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useEffect, useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { touchRecentParent } from '../lib/tradeDeskStore'

type ChildRow = {
  id: string
  qty: number
  limit: number
  status: string
}

function demoSlice(parentId: string): { timeFrac: number; fillFrac: number; children: ChildRow[] } {
  if (parentId === 'demo-parent-vwap-1') {
    return {
      timeFrac: 0.4,
      fillFrac: 0.1,
      children: [
        { id: 'c-001', qty: 40, limit: 182.42, status: 'FILLED' },
        { id: 'c-002', qty: 40, limit: 182.38, status: 'WORKING' },
        { id: 'c-003', qty: 40, limit: 182.35, status: 'PENDING' },
      ],
    }
  }
  let h = 0
  for (let i = 0; i < parentId.length; i++) h = Math.imul(31, h) + parentId.charCodeAt(i) | 0
  const timeFrac = 0.25 + (Math.abs(h) % 70) / 100
  const fillFrac = 0.2 + (Math.abs(h >> 5) % 60) / 100
  const children: ChildRow[] = [1, 2, 3, 4, 5].map((i) => ({
    id: `c-${i}`,
    qty: 25,
    limit: 100 + (h % 50) / 10 + i * 0.02,
    status: i === 1 ? 'FILLED' : i === 2 ? 'WORKING' : 'PENDING',
  }))
  return { timeFrac, fillFrac, children }
}

export default function ExecutionTracerPage() {
  const { parentId: raw } = useParams<{ parentId: string }>()
  const parentId = raw ? decodeURIComponent(raw) : ''
  const density = useMantineTableDensity()

  useEffect(() => {
    if (parentId) touchRecentParent(parentId)
  }, [parentId])

  const slice = useMemo(() => (parentId ? demoSlice(parentId) : null), [parentId])
  const lagging = slice && slice.fillFrac + 0.25 < slice.timeFrac

  if (!parentId) {
    return (
      <PageScaffold>
        <Text c="dimmed">Missing parent id.</Text>
      </PageScaffold>
    )
  }

  const bid = 182.36
  const ask = 182.41
  const mid = (bid + ask) / 2

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Button component={Link} to="/execution" variant="default" size="compact-sm" mb="sm">
        ← Execution hub
      </Button>
      <Title order={1}>EMS slicer</Title>
      <Text ff="monospace" c="dimmed" size="sm">
        parent {parentId}
      </Text>

      <Paper withBorder p="md" radius="md" mt="md">
        <Group justify="space-between" mb="xs">
          <Text fw={600} size="sm">
            Slicer progress
          </Text>
          {lagging ? (
            <Text size="xs" c="orange" fw={600}>
              Behind schedule (time &gt; fill)
            </Text>
          ) : (
            <Text size="xs" c="dimmed">
              On pace
            </Text>
          )}
        </Group>
        <Text size="xs" c="dimmed" mb={4}>
          Window elapsed vs filled notional — orange when time progress materially exceeds fill progress.
        </Text>
        <Stack gap="xs">
          <div>
            <Text size="xs" mb={4}>
              Time window {(slice!.timeFrac * 100).toFixed(0)}%
            </Text>
            <Progress value={slice!.timeFrac * 100} size="lg" color="gray" />
          </div>
          <div>
            <Text size="xs" mb={4}>
              Fill {(slice!.fillFrac * 100).toFixed(0)}%
            </Text>
            <Progress value={slice!.fillFrac * 100} size="lg" color={lagging ? 'orange' : 'teal'} />
          </div>
        </Stack>
      </Paper>

      <Title order={3} size="h4" mt="lg">
        Child orders (Alpaca)
      </Title>
      <Table.ScrollContainer minWidth={520}>
        <Table {...density} striped withTableBorder>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Child id</Table.Th>
              <Table.Th ta="right">Qty</Table.Th>
              <Table.Th ta="right">Limit</Table.Th>
              <Table.Th>Status</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {slice!.children.map((c) => (
              <Table.Tr key={c.id}>
                <Table.Td ff="monospace">{c.id}</Table.Td>
                <Table.Td ta="right">{c.qty}</Table.Td>
                <Table.Td ta="right" ff="monospace">
                  {c.limit.toFixed(2)}
                </Table.Td>
                <Table.Td ff="monospace">{c.status}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>

      <Title order={3} size="h4" mt="lg">
        Market microstructure (L1)
      </Title>
      <Paper withBorder p="md" radius="md" maw={420}>
        <Group grow align="stretch" gap="md">
          <Stack gap={4}>
            <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
              Bid
            </Text>
            <Text ff="monospace" fw={700} fz="lg" c="teal">
              {bid.toFixed(2)}
            </Text>
            <Text size="xs" c="dimmed">
              Peg reference for passive child
            </Text>
          </Stack>
          <Stack gap={4}>
            <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
              Ask
            </Text>
            <Text ff="monospace" fw={700} fz="lg" c="red">
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
            w={`${((mid - bid) / (ask - bid)) * 100}%`}
            style={{ borderRadius: 4, background: 'var(--mantine-color-yellow-5)' }}
          />
        </Box>
        <Text size="xs" c="dimmed" mt={6}>
          Yellow marker: child peg vs mid (illustrative).
        </Text>
      </Paper>
    </PageScaffold>
  )
}
