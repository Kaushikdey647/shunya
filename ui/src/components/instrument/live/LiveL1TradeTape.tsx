import { ScrollArea, Stack, Table, Text } from '@mantine/core'
import { useMemo } from 'react'
import { useLiveL1 } from '../../../live/l1Store'

export function LiveL1TradeTape() {
  const { state } = useLiveL1()
  const rows = useMemo(() => {
    return [...state.tape].reverse().slice(0, 40)
  }, [state.tape])

  return (
    <Stack gap="xs">
      <Text size="sm">Tape (trades, cancels, corrections)</Text>
      <ScrollArea h={200} type="auto">
        <Table striped highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Time</Table.Th>
              <Table.Th>Kind</Table.Th>
              <Table.Th>Detail</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.length === 0 ? (
              <Table.Tr>
                <Table.Td colSpan={3}>
                  <Text size="xs" c="dimmed">
                    No events yet.
                  </Text>
                </Table.Td>
              </Table.Tr>
            ) : (
              rows.map((r, i) => {
                if (r.kind === 'trade') {
                  const t = r.t
                  return (
                    <Table.Tr key={`${t.time}-${t.price}-${i}`}>
                      <Table.Td style={{ whiteSpace: 'nowrap' }}>{t.time}</Table.Td>
                      <Table.Td>trade</Table.Td>
                      <Table.Td>
                        {t.price} × {t.size}
                      </Table.Td>
                    </Table.Tr>
                  )
                }
                if (r.kind === 'correction') {
                  return (
                    <Table.Tr key={`c-${r.time}-${i}`}>
                      <Table.Td style={{ whiteSpace: 'nowrap' }}>{r.time}</Table.Td>
                      <Table.Td>correction</Table.Td>
                      <Table.Td>{r.summary}</Table.Td>
                    </Table.Tr>
                  )
                }
                return (
                  <Table.Tr key={`x-${r.time}-${i}`}>
                    <Table.Td style={{ whiteSpace: 'nowrap' }}>{r.time}</Table.Td>
                    <Table.Td>cancel</Table.Td>
                    <Table.Td>{r.summary}</Table.Td>
                  </Table.Tr>
                )
              })
            )}
          </Table.Tbody>
        </Table>
      </ScrollArea>
    </Stack>
  )
}
