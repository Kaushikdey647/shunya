import { SegmentedControl, Stack, Table, Text } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { getInstrumentFinancials } from '../../api/endpoints'
import type { InstrumentFinancialFrequency, InstrumentStatement } from '../../api/types'
import ApiErrorAlert from '../ApiErrorAlert'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'

type Props = {
  symbol: string
  enabled: boolean
}

export default function InstrumentFinancialsPanel({ symbol, enabled }: Props) {
  const tableProps = useMantineTableDensity()
  const [statement, setStatement] = useState<InstrumentStatement>('income')
  const [frequency, setFrequency] = useState<InstrumentFinancialFrequency>('quarterly')

  const q = useQuery({
    queryKey: ['instrument-financials', symbol, statement, frequency],
    queryFn: () =>
      getInstrumentFinancials(symbol, {
        statement,
        frequency,
        periods: 8,
      }),
    enabled: enabled && symbol.length > 0,
    staleTime: 120_000,
  })

  return (
    <Stack gap="md">
      <SegmentedControl
        value={frequency}
        onChange={(v) => setFrequency(v as InstrumentFinancialFrequency)}
        data={[
          { label: 'Quarterly', value: 'quarterly' },
          { label: 'Annual', value: 'annual' },
        ]}
      />
      <SegmentedControl
        value={statement}
        onChange={(v) => setStatement(v as InstrumentStatement)}
        data={[
          { label: 'Income', value: 'income' },
          { label: 'Balance sheet', value: 'balance' },
          { label: 'Cash flow', value: 'cashflow' },
        ]}
      />
      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text size="sm" c="dimmed">
          Loading statements…
        </Text>
      )}
      {q.data && !q.data.available && (
        <Text size="sm" c="dimmed">
          No statement data for this instrument.
        </Text>
      )}
      {q.data?.available && q.data.rows.length === 0 && !q.isLoading && (
        <Text size="sm" c="dimmed">
          Empty statement table.
        </Text>
      )}
      {q.data?.available && q.data.rows.length > 0 && (
        <Table.ScrollContainer minWidth={480}>
          <Table {...tableProps} striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th style={{ minWidth: 200 }}>Line item</Table.Th>
                {q.data.periods.map((p) => (
                  <Table.Th key={p} style={{ whiteSpace: 'nowrap' }}>
                    {p.slice(0, 10)}
                  </Table.Th>
                ))}
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {q.data.rows.map((row) => (
                <Table.Tr key={row.label}>
                  <Table.Td>{row.label}</Table.Td>
                  {row.values.map((v, i) => (
                    <Table.Td key={`${row.label}-${i}`} ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {v != null && Number.isFinite(v)
                        ? v.toLocaleString(undefined, { maximumFractionDigits: 2 })
                        : '—'}
                    </Table.Td>
                  ))}
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}
      {q.data?.truncated && (
        <Text size="xs" c="dimmed">
          Table truncated for size.
        </Text>
      )}
    </Stack>
  )
}
