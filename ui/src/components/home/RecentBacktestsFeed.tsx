import { Badge, Card, Stack, Table, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { listBacktests } from '../../api/endpoints'
import type { BacktestJobOut } from '../../api/types'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'
import ApiErrorAlert from '../ApiErrorAlert'
import { TableRowsSkeleton } from './homeDashboardSkeletons'

function primaryMetric(summary: Record<string, unknown> | null): string {
  if (!summary) return '—'
  const cagr = summary.cagr_pct
  if (typeof cagr === 'number' && Number.isFinite(cagr)) return `${cagr.toFixed(2)}% CAGR`
  const sh = summary.sharpe_ratio
  if (typeof sh === 'number' && Number.isFinite(sh)) return `Sharpe ${sh.toFixed(2)}`
  return '—'
}

function statusBadge(status: BacktestJobOut['status']) {
  if (status === 'succeeded') return { color: 'green' as const, label: status }
  if (status === 'failed') return { color: 'red' as const, label: status }
  return { color: 'gray' as const, label: status }
}

export default function RecentBacktestsFeed() {
  const navigate = useNavigate()
  const tableProps = useMantineTableDensity()
  const q = useQuery({
    queryKey: ['backtests', 'home-recent'],
    queryFn: () => listBacktests({ limit: 10, offset: 0 }),
    staleTime: 30_000,
  })

  return (
    <Card padding="md" radius="md" withBorder>
      <Stack gap="md">
        <Title order={5}>Recent backtests</Title>
        <ApiErrorAlert error={q.error} />
        {q.isLoading && (
          <TableRowsSkeleton
            tableProps={tableProps}
            columnCount={3}
            rowCount={6}
            headers={['Alpha', 'Status', 'Metric']}
            minWidth={260}
            mih={120}
            mah={320}
          />
        )}
        {!q.isLoading && q.data && (
          <Table.ScrollContainer minWidth={260} mih={120} mah={320}>
            <Table {...tableProps}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Alpha</Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Metric</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {q.data.length === 0 ? (
                  <Table.Tr>
                    <Table.Td colSpan={3}>
                      <Text c="dimmed" size="sm">
                        No runs yet
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                ) : (
                  q.data.map((job) => {
                    const b = statusBadge(job.status)
                    return (
                      <Table.Tr
                        key={job.id}
                        style={{ cursor: 'pointer' }}
                        onClick={() => navigate(`/backtests/${encodeURIComponent(job.id)}`)}
                      >
                        <Table.Td>
                          <Text size="sm">{job.alpha_name?.trim() || job.alpha_id}</Text>
                        </Table.Td>
                        <Table.Td>
                          <Badge variant="light" color={b.color} size="sm">
                            {b.label}
                          </Badge>
                        </Table.Td>
                        <Table.Td>
                          <Text ff="monospace" size="sm">
                            {primaryMetric(job.result_summary)}
                          </Text>
                        </Table.Td>
                      </Table.Tr>
                    )
                  })
                )}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        )}
      </Stack>
    </Card>
  )
}
