import {
  Anchor,
  Button,
  Group,
  Paper,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { createUniverse, listUniverses } from '../api/endpoints'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'

export default function UniverseListPage() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const density = useMantineTableDensity()
  const [name, setName] = useState('')

  const q = useQuery({
    queryKey: ['universes', 'list'],
    queryFn: () => listUniverses({ limit: 500, offset: 0 }),
  })

  const rows = useMemo(() => q.data ?? [], [q.data])

  const createMut = useMutation({
    mutationFn: () => createUniverse({ name: name.trim() }),
    onSuccess: (row) => {
      void qc.invalidateQueries({ queryKey: ['universes'] })
      setName('')
      navigate(`/universe/${encodeURIComponent(row.id)}`)
    },
  })

  return (
    <PageScaffold>
      <Group justify="space-between" align="flex-start" wrap="wrap">
        <div>
          <Title order={1}>Universes</Title>
          <Text c="dimmed" size="sm">
            Saved equity ticker sets for backtests and portfolio union views.
          </Text>
        </div>
        <Anchor component={Link} to="/studio" size="sm">
          Alpha Studio
        </Anchor>
      </Group>

      <Paper withBorder p="md" radius="md" maw={520}>
        <Text fw={600} size="sm" mb="xs">
          New universe
        </Text>
        <ApiErrorAlert error={createMut.error} />
        <Group wrap="nowrap" align="flex-end">
          <TextInput
            label="Name"
            placeholder="e.g. my_watchlist"
            value={name}
            onChange={(e) => setName(e.currentTarget.value)}
            style={{ flex: 1 }}
            autoComplete="off"
          />
          <Button
            color="yellow"
            disabled={!name.trim() || createMut.isPending}
            onClick={() => createMut.mutate()}
          >
            Create
          </Button>
        </Group>
      </Paper>

      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text c="dimmed" size="sm">
          Loading…
        </Text>
      )}
      {rows.length > 0 && (
        <Table.ScrollContainer minWidth={480}>
          <Table {...density} striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Members</Table.Th>
                <Table.Th>Updated</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((u) => (
                <Table.Tr key={u.id}>
                  <Table.Td>
                    <Anchor component={Link} to={`/universe/${encodeURIComponent(u.id)}`}>
                      {u.name}
                    </Anchor>
                  </Table.Td>
                  <Table.Td>{u.member_count}</Table.Td>
                  <Table.Td>{new Date(u.updated_at).toLocaleString()}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}
      {!q.isLoading && rows.length === 0 && (
        <Text c="dimmed" size="sm">
          No universes yet. Create one above, then add tickers on the detail page or from an instrument page.
        </Text>
      )}
    </PageScaffold>
  )
}
