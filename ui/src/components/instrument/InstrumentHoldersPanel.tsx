import { Stack, Table, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getInstrumentHolders } from '../../api/endpoints'
import ApiErrorAlert from '../ApiErrorAlert'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'

type Props = {
  symbol: string
  enabled: boolean
}

export default function InstrumentHoldersPanel({ symbol, enabled }: Props) {
  const tableProps = useMantineTableDensity()
  const q = useQuery({
    queryKey: ['instrument-holders', symbol],
    queryFn: () => getInstrumentHolders(symbol),
    enabled: enabled && symbol.length > 0,
    staleTime: 120_000,
  })

  return (
    <Stack gap="lg">
      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text size="sm" c="dimmed">
          Loading holders…
        </Text>
      )}
      {q.data && (
        <>
          <Stack gap="xs">
            <Title order={5}>Institutional</Title>
            {!q.data.available_institutional || q.data.institutional.length === 0 ? (
              <Text size="sm" c="dimmed">
                No institutional holder data.
              </Text>
            ) : (
              <Table.ScrollContainer minWidth={400}>
                <Table {...tableProps} striped>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Holder</Table.Th>
                      <Table.Th>% held</Table.Th>
                      <Table.Th>Shares</Table.Th>
                      <Table.Th>Value</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {q.data.institutional.map((r) => (
                      <Table.Tr key={`${r.holder}-${r.date_reported ?? ''}`}>
                        <Table.Td>{r.holder}</Table.Td>
                        <Table.Td ff="monospace">
                          {r.percent_held != null && Number.isFinite(r.percent_held)
                            ? `${r.percent_held.toFixed(2)}%`
                            : '—'}
                        </Table.Td>
                        <Table.Td ff="monospace">
                          {r.shares != null && Number.isFinite(r.shares)
                            ? r.shares.toLocaleString(undefined, { maximumFractionDigits: 0 })
                            : '—'}
                        </Table.Td>
                        <Table.Td ff="monospace">
                          {r.value != null && Number.isFinite(r.value)
                            ? r.value.toLocaleString(undefined, { maximumFractionDigits: 0 })
                            : '—'}
                        </Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </Table.ScrollContainer>
            )}
          </Stack>
          <Stack gap="xs">
            <Title order={5}>Mutual funds</Title>
            {!q.data.available_mutual_funds || q.data.mutual_funds.length === 0 ? (
              <Text size="sm" c="dimmed">
                No mutual fund holder data.
              </Text>
            ) : (
              <Table.ScrollContainer minWidth={400}>
                <Table {...tableProps} striped>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Fund</Table.Th>
                      <Table.Th>% held</Table.Th>
                      <Table.Th>Shares</Table.Th>
                      <Table.Th>Value</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {q.data.mutual_funds.map((r) => (
                      <Table.Tr key={`${r.holder}-${r.date_reported ?? ''}`}>
                        <Table.Td>{r.holder}</Table.Td>
                        <Table.Td ff="monospace">
                          {r.percent_held != null && Number.isFinite(r.percent_held)
                            ? `${r.percent_held.toFixed(2)}%`
                            : '—'}
                        </Table.Td>
                        <Table.Td ff="monospace">
                          {r.shares != null && Number.isFinite(r.shares)
                            ? r.shares.toLocaleString(undefined, { maximumFractionDigits: 0 })
                            : '—'}
                        </Table.Td>
                        <Table.Td ff="monospace">
                          {r.value != null && Number.isFinite(r.value)
                            ? r.value.toLocaleString(undefined, { maximumFractionDigits: 0 })
                            : '—'}
                        </Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </Table.ScrollContainer>
            )}
          </Stack>
        </>
      )}
    </Stack>
  )
}
