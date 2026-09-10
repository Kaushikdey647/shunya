import { Paper, ScrollArea, Stack, Text, Title } from '@mantine/core'
import { useMemo } from 'react'
import type { OmsFillOut } from '../../api/types'
import { useOmsStream } from '../../live/omsStore'
import OrderStreamLine from './OrderStreamLine'

function fillToLine(f: OmsFillOut): string {
  const time = new Date(f.ts).toLocaleTimeString()
  return `${time} ${f.symbol} ${f.side} ${f.quantity} @ ${f.price.toFixed(2)} FILLED (parent ${f.parent_order_id.slice(0, 8)})`
}

export default function FillsTape() {
  const { state } = useOmsStream()
  const lines = useMemo(() => [...state.fills].reverse().map(fillToLine), [state.fills])

  return (
    <Paper withBorder p="md" radius="md" h="100%">
      <Title order={4} mb="xs">
        Fills
      </Title>
      <ScrollArea h={280} type="auto">
        <Stack gap={0}>
          {lines.map((line, i) => (
            <OrderStreamLine key={i} line={line} />
          ))}
          {lines.length === 0 && (
            <Text size="sm" c="dimmed">
              No fills yet this session.
            </Text>
          )}
        </Stack>
      </ScrollArea>
    </Paper>
  )
}
