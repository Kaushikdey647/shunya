import {
  ActionIcon,
  Anchor,
  Button,
  Card,
  Group,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  useComputedColorScheme,
} from '@mantine/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { postMarketSnapshot, instrumentDetailPath } from '../../api/endpoints'
import {
  addWatchlistSymbol,
  readWatchlist,
  removeWatchlistSymbol,
} from '../../lib/watchlist'
import { SignedPctText } from '../../lib/signedPct'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'
import ApiErrorAlert from '../ApiErrorAlert'
import { TableRowsSkeleton } from './homeDashboardSkeletons'

export default function WatchlistCard() {
  const qc = useQueryClient()
  const tableProps = useMantineTableDensity()
  const colorScheme = useComputedColorScheme('light', { getInitialValueInEffect: false })
  const [tickers, setTickers] = useState<string[]>(() => readWatchlist())
  const [input, setInput] = useState('')

  const snap = useQuery({
    queryKey: ['market', 'snapshot', 'watchlist', tickers.join(',')],
    queryFn: () => postMarketSnapshot({ symbols: tickers }),
    enabled: tickers.length > 0,
    staleTime: 90_000,
  })

  const bySym = new Map((snap.data?.rows ?? []).map((r) => [r.symbol, r]))

  const onAdd = (e: React.FormEvent) => {
    e.preventDefault()
    const next = addWatchlistSymbol(input)
    setInput('')
    setTickers(next)
    void qc.invalidateQueries({ queryKey: ['market', 'snapshot', 'watchlist'] })
  }

  return (
    <Card padding="md" radius={colorScheme === 'dark' ? 0 : 'md'} withBorder>
      <Stack gap="md">
        <Title order={5}>Watchlist</Title>
        <form onSubmit={onAdd}>
          <Group gap="xs" wrap="nowrap" align="flex-end">
            <TextInput
              flex={1}
              placeholder="Ticker"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              maxLength={32}
              aria-label="Add ticker"
            />
            <Button type="submit" color="yellow">
              Add
            </Button>
          </Group>
        </form>
        {tickers.length === 0 ? (
          <Text c="dimmed" size="sm">
            Add symbols to track daily moves (stored in this browser).
          </Text>
        ) : (
          <>
            <ApiErrorAlert error={snap.error} />
            {snap.isLoading && (
              <TableRowsSkeleton
                tableProps={tableProps}
                columnCount={4}
                rowCount={tickers.length}
                headers={['Ticker', 'Last', 'Day %', '']}
                minWidth={280}
                mih={120}
                mah={360}
                lastHeaderWidth={90}
              />
            )}
            {!snap.isLoading && (
              <Table.ScrollContainer minWidth={280} mih={120} mah={360}>
                <Table {...tableProps}>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Ticker</Table.Th>
                      <Table.Th ta="right">Last</Table.Th>
                      <Table.Th ta="right">Day %</Table.Th>
                      <Table.Th w={90} />
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {tickers.map((sym) => {
                      const row = bySym.get(sym)
                      return (
                        <Table.Tr key={sym}>
                          <Table.Td>
                            <Anchor component={Link} to={instrumentDetailPath(sym)} ff="monospace" size="sm">
                              {sym}
                            </Anchor>
                          </Table.Td>
                          <Table.Td ta="right">
                            <Text ff="monospace" size="sm" style={{ fontVariantNumeric: 'tabular-nums' }}>
                              {row?.last != null && Number.isFinite(row.last)
                                ? row.last.toLocaleString(undefined, { maximumFractionDigits: 2 })
                                : '—'}
                            </Text>
                          </Table.Td>
                          <Table.Td ta="right">
                            <SignedPctText v={row?.pct_change_1d} />
                          </Table.Td>
                          <Table.Td>
                            <ActionIcon
                              type="button"
                              variant="subtle"
                              color="gray"
                              size="sm"
                              aria-label={`Remove ${sym}`}
                              onClick={() => {
                                setTickers(removeWatchlistSymbol(sym))
                                void qc.invalidateQueries({
                                  queryKey: ['market', 'snapshot', 'watchlist'],
                                })
                              }}
                              styles={{
                                root: {
                                  '&:hover': {
                                    color: 'var(--mantine-color-red-filled)',
                                    backgroundColor: 'rgba(255, 0, 0, 0.08)',
                                  },
                                },
                              }}
                            >
                              ×
                            </ActionIcon>
                          </Table.Td>
                        </Table.Tr>
                      )
                    })}
                  </Table.Tbody>
                </Table>
              </Table.ScrollContainer>
            )}
          </>
        )}
      </Stack>
    </Card>
  )
}
