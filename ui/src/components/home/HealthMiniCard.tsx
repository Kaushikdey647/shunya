import { Alert, Box, Card, Group, Stack, Table, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getHealth } from '../../api/endpoints'
import { isHealthResponse } from '../../api/types'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'
import ApiErrorAlert from '../ApiErrorAlert'
import { HealthMiniCardBodySkeleton } from './homeDashboardSkeletons'

export default function HealthMiniCard() {
  const tableProps = useMantineTableDensity()
  const q = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    staleTime: 30_000,
  })

  return (
    <Card padding="md" radius="md" withBorder>
      <Stack gap="md">
        <Title order={5}>System health</Title>
        <ApiErrorAlert error={q.error} variant="outline" compact />
        {q.isLoading && <HealthMiniCardBodySkeleton tableProps={tableProps} />}
        {q.data && !isHealthResponse(q.data) && (
          <Alert color="red" variant="outline" title="Unexpected health response" p="xs">
            <Text size="sm">
              GET /health did not return the expected JSON. Confirm{' '}
              <Text span ff="monospace">
                VITE_API_BASE
              </Text>{' '}
              points at your FastAPI service (public URL), not the SPA or /healthz plaintext.
            </Text>
          </Alert>
        )}
        {q.data && isHealthResponse(q.data) && (
          <Stack gap="sm">
            <Group gap="sm" wrap="wrap" align="center">
              <Box
                w={8}
                h={8}
                style={{
                  borderRadius: 999,
                  flexShrink: 0,
                  backgroundColor:
                    q.data.status === 'ok'
                      ? 'var(--mantine-color-teal-filled)'
                      : q.data.status === 'degraded'
                        ? 'var(--mantine-color-yellow-filled)'
                        : 'var(--mantine-color-red-filled)',
                }}
              />
              <Text size="sm" ff="monospace" fw={600}>
                GET /health
              </Text>
              <Text size="sm" fw={600}>
                {q.data.status}
              </Text>
            </Group>
            <Table.ScrollContainer minWidth={260}>
              <Table {...tableProps}>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>Backend</Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.backend.status}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.backend.latency_ms.toFixed(1)} ms
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Database</Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.database.status}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.database.latency_ms.toFixed(1)} ms
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Yahoo</Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.yfinance.status}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.yfinance.latency_ms.toFixed(1)} ms
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                  <Table.Tr>
                    <Table.Td>Alpaca</Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.alpaca.status}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {q.data.alpaca.status === 'skipped'
                          ? '—'
                          : `${q.data.alpaca.latency_ms.toFixed(1)} ms`}
                      </Text>
                    </Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          </Stack>
        )}
      </Stack>
    </Card>
  )
}
