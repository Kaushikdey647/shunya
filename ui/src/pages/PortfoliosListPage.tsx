import { Badge, Button, Group, Paper, Stack, Table, Text, Title } from '@mantine/core'
import { Link } from 'react-router-dom'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { createPortfolio, deletePortfolio } from '../lib/tradeDeskStore'
import { mockPortfolioRegistryMetrics } from '../lib/tradeDeskMockMetrics'
import { useTradeDesk } from '../hooks/useTradeDesk'

export default function PortfoliosListPage() {
  const desk = useTradeDesk()
  const density = useMantineTableDensity()

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Group justify="space-between" align="flex-start" wrap="wrap">
        <div>
          <Title order={1}>Portfolio registry</Title>
          <Text c="dimmed" size="sm">
            Combined sleeves, live Sharpe (mock), net exposure, and activation state. Parent surface for{' '}
            <Text span ff="monospace">
              PortfolioManager
            </Text>
            .
          </Text>
        </div>
        <Button color="yellow" onClick={() => createPortfolio(`Book ${desk.portfolios.length + 1}`)}>
          New portfolio
        </Button>
      </Group>

      {desk.portfolios.length === 0 ? (
        <Paper withBorder p="lg" radius="md">
          <Text c="dimmed" size="sm" mb="md">
            No portfolios yet. Create one, then attach alphas from Studio or deploy from a backtest job.
          </Text>
          <Button variant="light" color="yellow" onClick={() => createPortfolio('Core book')}>
            Create “Core book”
          </Button>
        </Paper>
      ) : (
        <Table.ScrollContainer minWidth={720}>
          <Table
            {...density}
            verticalSpacing={4}
            horizontalSpacing="xs"
            fz="xs"
            striped
            highlightOnHover
            withTableBorder
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th py={6}>Portfolio</Table.Th>
                <Table.Th py={6} ta="right">
                  Live Sharpe
                </Table.Th>
                <Table.Th py={6} ta="right">
                  Net exposure
                </Table.Th>
                <Table.Th py={6}>Sleeves</Table.Th>
                <Table.Th py={6}>Status</Table.Th>
                <Table.Th py={6} w={120}>
                  {' '}
                </Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {desk.portfolios.map((p) => {
                const { liveSharpe, netExposure } = mockPortfolioRegistryMetrics(p.id, p.slots.length)
                return (
                  <Table.Tr key={p.id}>
                    <Table.Td py={6}>
                      <Group gap="xs" wrap="nowrap" align="baseline">
                        <Text fw={600} size="sm" lineClamp={1}>
                          {p.name}
                        </Text>
                        <Text size="xs" c="dimmed" ff="monospace" lineClamp={1}>
                          {p.id.slice(0, 8)}… · v{p.strategySpecVersion}
                        </Text>
                      </Group>
                    </Table.Td>
                    <Table.Td py={6} ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {liveSharpe == null ? '—' : liveSharpe.toFixed(2)}
                    </Table.Td>
                    <Table.Td py={6} ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {(netExposure * 100).toFixed(1)}%
                    </Table.Td>
                    <Table.Td py={6}>{p.slots.length}</Table.Td>
                    <Table.Td py={6}>
                      <Group gap={4}>
                        <Badge color={p.goLive ? 'teal' : 'gray'} variant="light" size="xs">
                          {p.goLive ? 'Live feed' : 'Research'}
                        </Badge>
                        <Badge color={p.blendMode === 'alpha_blend' ? 'blue' : 'violet'} variant="outline" size="xs">
                          {p.blendMode === 'alpha_blend' ? 'α-blend' : 'target'}
                        </Badge>
                      </Group>
                    </Table.Td>
                    <Table.Td py={6}>
                      <Button component={Link} size="compact-xs" variant="light" color="yellow" to={`/portfolios/${encodeURIComponent(p.id)}`}>
                        Workspace
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                )
              })}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}

      <Paper withBorder p="md" radius="md">
        <Text fw={600} size="sm" mb="xs">
          Danger zone
        </Text>
        <Stack gap="xs">
          {desk.portfolios.map((p) => (
            <Group key={`del-${p.id}`} justify="space-between" wrap="nowrap">
              <Text size="sm" ff="monospace">
                {p.name}
              </Text>
              <Button
                size="compact-xs"
                color="red"
                variant="subtle"
                onClick={() => {
                  if (window.confirm(`Delete portfolio “${p.name}”?`)) deletePortfolio(p.id)
                }}
              >
                Delete
              </Button>
            </Group>
          ))}
          {desk.portfolios.length === 0 && <Text c="dimmed" size="xs">Nothing to delete.</Text>}
        </Stack>
      </Paper>
    </PageScaffold>
  )
}
