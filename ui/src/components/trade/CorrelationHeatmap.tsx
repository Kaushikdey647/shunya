import { Box, Group, Paper, Table, Text, Tooltip } from '@mantine/core'

function pairCorr(a: string, b: string): number {
  if (a === b) return 1
  let h = 0
  const s = `${a}::${b}`
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(31, h) + s.charCodeAt(i) | 0
  }
  const x = (Math.abs(h) % 1000) / 1000
  return Math.round((0.15 + x * 0.65) * 1000) / 1000
}

function corrColor(v: number): string {
  if (v >= 0.85) return 'var(--mantine-color-red-8)'
  if (v >= 0.55) return 'var(--mantine-color-orange-7)'
  if (v >= 0.35) return 'var(--mantine-color-yellow-7)'
  return 'var(--mantine-color-teal-8)'
}

type Props = {
  labels: string[]
}

/** Compact rolling-correlation style matrix (synthetic until series API exists). */
export default function CorrelationHeatmap({ labels }: Props) {
  if (labels.length === 0) {
    return (
      <Text c="dimmed" size="sm">
        Add at least one alpha slot to render the correlation grid.
      </Text>
    )
  }

  return (
    <Paper withBorder p="md" radius="md">
      <Group justify="space-between" mb="sm" wrap="wrap">
        <Text fw={600} size="sm">
          Rolling correlation (synthetic)
        </Text>
        <Text size="xs" c="dimmed">
          High overlap warns blend dampening — wire real returns for production.
        </Text>
      </Group>
      <Table.ScrollContainer minWidth={200 + labels.length * 44}>
        <Table withTableBorder withColumnBorders horizontalSpacing={4} verticalSpacing={4}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th w={120}> </Table.Th>
              {labels.map((lab) => (
                <Table.Th key={lab} ta="center" fz={10} ff="monospace" px={4}>
                  {lab.slice(0, 8)}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {labels.map((row) => (
              <Table.Tr key={row}>
                <Table.Td ff="monospace" fz={10}>
                  {row.slice(0, 14)}
                </Table.Td>
                {labels.map((col) => {
                  const v = pairCorr(row, col)
                  const bg = corrColor(v)
                  return (
                    <Table.Td key={`${row}-${col}`} p={4}>
                      <Tooltip label={`ρ ≈ ${v.toFixed(3)}`} withArrow>
                        <Box
                          h={32}
                          w="100%"
                          miw={36}
                          style={{
                            borderRadius: 4,
                            background: bg,
                            opacity: 0.75 + v * 0.2,
                          }}
                        />
                      </Tooltip>
                    </Table.Td>
                  )
                })}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Group gap="md" mt="sm" wrap="wrap">
        <Text size="xs" c="dimmed">
          Teal &lt; 0.35 · Yellow 0.35–0.55 · Orange 0.55–0.85 · Red ≥ 0.85
        </Text>
      </Group>
    </Paper>
  )
}
