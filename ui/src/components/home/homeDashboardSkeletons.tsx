import { Card, Divider, SimpleGrid, Skeleton, Stack, Table, Text } from '@mantine/core'
import type { MantineTableDensityProps } from '../../hooks/useMantineTableDensity'
import { MACRO_STRIP_SYMBOLS } from '../../lib/macroSymbols'

export function MacroStripSkeleton() {
  const count = MACRO_STRIP_SYMBOLS.length

  return (
    <SimpleGrid cols={{ base: 2, sm: 4 }} spacing="md" aria-busy="true">
      {Array.from({ length: count }, (_, i) => (
        <Card key={i} padding="sm" radius="md" withBorder>
          <Card.Section inheritPadding pb="xs">
            <Skeleton height={16} width="45%" mb={8} />
            <Stack gap={6}>
              <Skeleton height={14} width="70%" />
              <Skeleton height={14} width="40%" />
            </Stack>
          </Card.Section>
          <Skeleton height={52} radius="sm" />
        </Card>
      ))}
    </SimpleGrid>
  )
}

export function HeadlineListSkeleton({ rows = 6 }: { rows?: number }) {
  return (
    <Stack gap={0} aria-busy="true">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i}>
          {i > 0 && <Divider my="sm" />}
          <Stack gap={6}>
            <Skeleton height={18} width="92%" />
            <Stack gap={4}>
              <Skeleton height={12} width={120} />
              <Skeleton height={12} width={100} />
            </Stack>
          </Stack>
        </div>
      ))}
    </Stack>
  )
}

type TableRowsSkeletonProps = {
  tableProps: MantineTableDensityProps
  columnCount: number
  rowCount: number
  /** When set, renders a thead with these labels (matches loaded tables). */
  headers?: string[]
  minWidth?: number
  mih?: number
  mah?: number
  lastHeaderWidth?: number
}

export function TableRowsSkeleton({
  tableProps,
  columnCount,
  rowCount,
  headers,
  minWidth = 280,
  mih,
  mah,
  lastHeaderWidth,
}: TableRowsSkeletonProps) {
  const showHead = headers != null && headers.length === columnCount

  return (
    <Table.ScrollContainer minWidth={minWidth} mih={mih} mah={mah} aria-busy="true">
      <Table {...tableProps}>
        {showHead && (
          <Table.Thead>
            <Table.Tr>
              {headers!.map((label, j) => (
                <Table.Th key={j} {...(j === headers!.length - 1 && lastHeaderWidth ? { w: lastHeaderWidth } : {})}>
                  {label}
                </Table.Th>
              ))}
            </Table.Tr>
          </Table.Thead>
        )}
        <Table.Tbody>
          {Array.from({ length: rowCount }, (_, r) => (
            <Table.Tr key={r}>
              {Array.from({ length: columnCount }, (_, c) => (
                <Table.Td key={c}>
                  <Skeleton height={14} radius="sm" />
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  )
}

export function HealthMiniCardBodySkeleton({ tableProps }: { tableProps: MantineTableDensityProps }) {
  return (
    <Stack gap="sm" aria-busy="true">
      <Skeleton height={56} radius="md" />
      <Table.ScrollContainer minWidth={260}>
        <Table {...tableProps}>
          <Table.Tbody>
            {['Backend', 'Database', 'Yahoo'].map((label) => (
              <Table.Tr key={label}>
                <Table.Td>
                  <Text size="sm">{label}</Text>
                </Table.Td>
                <Table.Td>
                  <Skeleton height={14} radius="sm" />
                </Table.Td>
                <Table.Td>
                  <Skeleton height={14} width={56} radius="sm" />
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Stack>
  )
}
