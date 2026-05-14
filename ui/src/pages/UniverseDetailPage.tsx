import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Group,
  Paper,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useDebouncedValue } from '@mantine/hooks'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  addUniverseMembers,
  getUniverse,
  getUniverseSummary,
  listUniverseMembers,
  removeUniverseMembers,
  searchInstruments,
} from '../api/endpoints'
import type { InstrumentSearchQuote } from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import {
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

const PIE_COLORS = ['#f59f00', '#228be6', '#40c057', '#845ef7', '#fd7e14', '#15aabf', '#e64980', '#868e96']

export default function UniverseDetailPage() {
  const { id } = useParams<{ id: string }>()
  const qc = useQueryClient()
  const density = useMantineTableDensity()
  const [qInput, setQInput] = useState('')
  const [debounced] = useDebouncedValue(qInput.trim(), 280)
  const [pick, setPick] = useState<InstrumentSearchQuote | null>(null)

  const universeQ = useQuery({
    queryKey: ['universe', id],
    queryFn: () => getUniverse(id!),
    enabled: Boolean(id),
  })

  const membersQ = useQuery({
    queryKey: ['universe', id, 'members'],
    queryFn: () => listUniverseMembers(id!, { limit: 5000, offset: 0 }),
    enabled: Boolean(id),
  })

  const summaryQ = useQuery({
    queryKey: ['universe', id, 'summary'],
    queryFn: () => getUniverseSummary(id!),
    enabled: Boolean(id),
  })

  const searchQ = useQuery({
    queryKey: ['instrument-search-mini', debounced],
    queryFn: () => searchInstruments(debounced),
    enabled: debounced.length >= 1,
  })

  const equitySuggestions = useMemo(() => {
    const quotes = searchQ.data?.quotes ?? []
    return quotes.filter((r) => {
      const qt = (r.quote_type || '').toUpperCase()
      return !qt || qt === 'EQUITY'
    })
  }, [searchQ.data])

  const addMut = useMutation({
    mutationFn: (sym: string) => addUniverseMembers(id!, { tickers: [sym] }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['universe', id] })
      void qc.invalidateQueries({ queryKey: ['universe', id, 'summary'] })
      void qc.invalidateQueries({ queryKey: ['universes'] })
      setPick(null)
      setQInput('')
    },
  })

  const removeMut = useMutation({
    mutationFn: (sym: string) => removeUniverseMembers(id!, [sym]),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['universe', id] })
      void qc.invalidateQueries({ queryKey: ['universe', id, 'summary'] })
      void qc.invalidateQueries({ queryKey: ['universes'] })
    },
  })

  if (!id) {
    return (
      <PageScaffold>
        <Text c="dimmed">Missing universe id.</Text>
      </PageScaffold>
    )
  }

  const u = universeQ.data
  const summary = summaryQ.data

  const sectorPie = (summary?.sector_breakdown ?? []).map((s, i) => ({
    name: s.label,
    value: s.count,
    fill: PIE_COLORS[i % PIE_COLORS.length],
  }))
  const industryPie = (summary?.industry_breakdown ?? []).slice(0, 12).map((s, i) => ({
    name: s.label,
    value: s.count,
    fill: PIE_COLORS[i % PIE_COLORS.length],
  }))

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Button component={Link} to="/universe" variant="default" size="compact-sm" mb="xs">
        ← Universes
      </Button>
      <ApiErrorAlert error={universeQ.error} />
      {universeQ.isLoading && <Text c="dimmed">Loading…</Text>}
      {!universeQ.isLoading && !u && <Text c="dimmed">Universe not found.</Text>}
      {u && (
        <Stack gap="lg">
          <Group justify="space-between" align="flex-start" wrap="wrap">
            <div>
              <Title order={1}>{u.name}</Title>
              <Group gap="xs" mt={4}>
                <Badge variant="light" color="gray" ff="monospace" size="sm">
                  {u.id}
                </Badge>
                <Badge variant="outline" color="yellow" size="sm">
                  {u.member_count} members
                </Badge>
              </Group>
              {u.description && (
                <Text size="sm" c="dimmed" mt={4}>
                  {u.description}
                </Text>
              )}
            </div>
          </Group>

          <Paper withBorder p="md" radius="md" maw={560}>
            <Text fw={600} size="sm" mb="xs">
              Add ticker
            </Text>
            <Text size="xs" c="dimmed" mb="sm">
              Search lists equity-style names; server rejects non-equity classifications when present.
            </Text>
            <Stack gap="sm">
              <TextInput
                placeholder="Search symbol…"
                value={qInput}
                onChange={(e) => {
                  setQInput(e.currentTarget.value)
                  setPick(null)
                }}
                autoComplete="off"
              />
              {equitySuggestions.length > 0 && (
                <Stack gap={4}>
                  {equitySuggestions.slice(0, 8).map((r) => (
                    <Button
                      key={r.symbol}
                      variant={pick?.symbol === r.symbol ? 'light' : 'subtle'}
                      color="yellow"
                      size="compact-xs"
                      justify="flex-start"
                      onClick={() => setPick(r)}
                    >
                      <Text span ff="monospace">
                        {r.symbol}
                      </Text>
                      <Text span c="dimmed" size="xs" ml="xs">
                        {r.shortname ?? r.longname ?? ''}
                      </Text>
                    </Button>
                  ))}
                </Stack>
              )}
              <Group>
                <Button
                  color="yellow"
                  disabled={!pick || addMut.isPending}
                  onClick={() => pick && addMut.mutate(pick.symbol)}
                >
                  {addMut.isPending ? 'Adding…' : 'Add to universe'}
                </Button>
              </Group>
              <ApiErrorAlert error={addMut.error} />
            </Stack>
          </Paper>

          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
            <Card withBorder padding="md" radius="md">
              <Text fw={600} size="sm" mb="xs">
                Sector mix (excl. unknown / other)
              </Text>
              {summaryQ.isLoading && <Text c="dimmed" size="xs">Loading…</Text>}
              {sectorPie.length === 0 && !summaryQ.isLoading && (
                <Text c="dimmed" size="xs">
                  No classified sectors for members yet.
                </Text>
              )}
              {sectorPie.length > 0 && (
                <div style={{ width: '100%', height: 280 }}>
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie dataKey="value" data={sectorPie} nameKey="name" outerRadius={100} label />
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </Card>
            <Card withBorder padding="md" radius="md">
              <Text fw={600} size="sm" mb="xs">
                Industry mix (top 12, excl. unknown / other)
              </Text>
              {summaryQ.isLoading && <Text c="dimmed" size="xs">Loading…</Text>}
              {industryPie.length === 0 && !summaryQ.isLoading && (
                <Text c="dimmed" size="xs">
                  No classified industries for members yet.
                </Text>
              )}
              {industryPie.length > 0 && (
                <div style={{ width: '100%', height: 280 }}>
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie dataKey="value" data={industryPie} nameKey="name" outerRadius={100} label />
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </Card>
          </SimpleGrid>

          <Card withBorder padding="md" radius="md">
            <Text fw={600} size="sm" mb="md">
              Aggregate fundamentals (latest daily row per member, where available)
            </Text>
            {summary && (
              <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md">
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase">
                    Median market cap
                  </Text>
                  <Text fw={600}>
                    {summary.median_market_cap != null && Number.isFinite(summary.median_market_cap)
                      ? summary.median_market_cap.toLocaleString(undefined, { maximumFractionDigits: 0 })
                      : '—'}
                  </Text>
                </div>
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase">
                    Mean trailing P/E
                  </Text>
                  <Text fw={600}>{summary.mean_trailing_pe?.toFixed(2) ?? '—'}</Text>
                </div>
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase">
                    Median beta
                  </Text>
                  <Text fw={600}>{summary.median_beta?.toFixed(3) ?? '—'}</Text>
                </div>
                <div>
                  <Text size="xs" c="dimmed" tt="uppercase">
                    Fundamentals cells
                  </Text>
                  <Text fw={600}>{summary.fundamentals_coverage_count}</Text>
                </div>
              </SimpleGrid>
            )}
          </Card>

          <Title order={3} size="h4">
            Members
          </Title>
          <ApiErrorAlert error={membersQ.error} />
          {membersQ.isLoading && <Text c="dimmed" size="sm">Loading members…</Text>}
          {(membersQ.data?.length ?? 0) > 0 && (
            <Table.ScrollContainer minWidth={480}>
              <Table {...density} striped>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Ticker</Table.Th>
                    <Table.Th>Name</Table.Th>
                    <Table.Th>Sector</Table.Th>
                    <Table.Th w="3rem" />
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {(membersQ.data ?? []).map((m) => (
                    <Table.Tr key={m.ticker}>
                      <Table.Td ff="monospace">{m.ticker}</Table.Td>
                      <Table.Td>{m.long_name ?? '—'}</Table.Td>
                      <Table.Td>{m.sector_disp ?? '—'}</Table.Td>
                      <Table.Td>
                        <ActionIcon
                          variant="subtle"
                          color="red"
                          aria-label={`Remove ${m.ticker}`}
                          disabled={removeMut.isPending}
                          onClick={() => removeMut.mutate(m.ticker)}
                        >
                          ×
                        </ActionIcon>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          )}
          {!membersQ.isLoading && (membersQ.data?.length ?? 0) === 0 && (
            <Text c="dimmed" size="sm">
              No members yet. Use the search box above or add from an instrument page.
            </Text>
          )}
          <ApiErrorAlert error={removeMut.error} />
        </Stack>
      )}
    </PageScaffold>
  )
}
