import {
  Paper,
  Table,
  Text,
  Title,
  useMantineColorScheme,
  useMantineTheme,
} from '@mantine/core'
import { useMemo } from 'react'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { heatmapCellColor, monthlyReturnsFromEquityCurve } from '../lib/monthlyReturnsHeatmap'

export default function MonthlyReturnsHeatmap({
  equityCurve,
}: {
  equityCurve: Record<string, unknown>[]
}) {
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()
  const density = useMantineTableDensity()
  const grid = monthlyReturnsFromEquityCurve(equityCurve)

  const palette = useMemo(
    () => ({
      success: theme.colors.teal[6]!,
      error: theme.colors.red[6]!,
      surface:
        colorScheme === 'dark'
          ? String(theme.other.darkPanelBg)
          : String(theme.white),
      empty:
        colorScheme === 'dark' ? theme.colors.dark[6]! : theme.colors.gray[2]!,
    }),
    [
      colorScheme,
      theme.colors.dark,
      theme.colors.gray,
      theme.colors.red,
      theme.colors.teal,
      theme.other.darkPanelBg,
      theme.white,
    ],
  )

  const strong =
    colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.dark[9]

  const monthLabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

  if (grid.years.length === 0) {
    return (
      <Text c="dimmed" size="sm">
        Not enough monthly data for a returns heatmap.
      </Text>
    )
  }

  return (
    <Paper withBorder p="md" radius="md">
      <Title order={4} size="h5" mb="sm">
        Monthly returns %
      </Title>
      <Table.ScrollContainer minWidth={480}>
        <Table {...density} striped={false} highlightOnHover={false} withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th style={{ width: '3.5rem' }}>Year</Table.Th>
              {grid.months.map((m) => (
                <Table.Th key={m} ta="center" style={{ minWidth: '2rem' }}>
                  {monthLabels[m - 1]}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {grid.years.map((y) => (
              <Table.Tr key={y}>
                <Table.Th scope="row" ta="center" ff="monospace">
                  {y}
                </Table.Th>
                {grid.months.map((mo) => {
                  const key = `${y}-${mo}`
                  const v = grid.values.get(key)
                  const label = v !== undefined ? `${v >= 0 ? '+' : ''}${v.toFixed(1)}%` : '—'
                  return (
                    <Table.Td
                      key={key}
                      ta="center"
                      ff="monospace"
                      fz="xs"
                      fw={600}
                      style={{
                        background: heatmapCellColor(v, grid.normAbs, palette),
                        color: strong,
                      }}
                      title={v !== undefined ? label : 'No data'}
                    >
                      {v !== undefined ? v.toFixed(1) : ''}
                    </Table.Td>
                  )
                })}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Paper>
  )
}
