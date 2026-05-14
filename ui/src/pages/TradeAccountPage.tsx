import {
  Alert,
  Badge,
  Button,
  Group,
  Paper,
  PasswordInput,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  getAppSettings,
  getTradeAccountActivities,
  getTradeAccountEquity,
  getTradePortfolioHistory,
} from '../api/endpoints'
import type { AlpacaEquityAccountOut } from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'

function fmtUsd(raw: string | null | undefined): string {
  if (raw == null || raw === '') return '—'
  const n = Number(raw)
  if (!Number.isFinite(n)) return String(raw)
  return n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 })
}

function fmtBool(v: boolean | null | undefined): string {
  if (v == null) return '—'
  return v ? 'Yes' : 'No'
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Paper withBorder p="sm" radius="md">
      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
        {label}
      </Text>
      <Text size="md" fw={600}>
        {value}
      </Text>
    </Paper>
  )
}

function equityStats(a: AlpacaEquityAccountOut) {
  return (
    <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
      <Stat label="Equity" value={fmtUsd(a.equity)} />
      <Stat label="Cash" value={fmtUsd(a.cash)} />
      <Stat label="Portfolio value" value={fmtUsd(a.portfolio_value)} />
      <Stat label="Buying power" value={fmtUsd(a.buying_power)} />
      <Stat label="Reg T buying power" value={fmtUsd(a.regt_buying_power)} />
      <Stat label="Daytrading BP" value={fmtUsd(a.daytrading_buying_power)} />
      <Stat label="Effective BP" value={fmtUsd(a.effective_buying_power)} />
      <Stat label="Non-marginable BP" value={fmtUsd(a.non_marginable_buying_power)} />
      <Stat label="Long market value" value={fmtUsd(a.long_market_value)} />
      <Stat label="Short market value" value={fmtUsd(a.short_market_value)} />
      <Stat label="Position market value" value={fmtUsd(a.position_market_value)} />
      <Stat label="Initial margin" value={fmtUsd(a.initial_margin)} />
      <Stat label="Maintenance margin" value={fmtUsd(a.maintenance_margin)} />
      <Stat label="SMA" value={fmtUsd(a.sma)} />
      <Stat label="Accrued fees" value={fmtUsd(a.accrued_fees)} />
      <Stat label="BOD DTBP" value={fmtUsd(a.bod_dtbp)} />
      <Stat label="Multiplier" value={a.multiplier ?? '—'} />
      <Stat label="Daytrade count" value={a.daytrade_count != null ? String(a.daytrade_count) : '—'} />
      <Stat label="Balance as-of" value={a.balance_asof ?? '—'} />
      <Stat label="Pattern day trader" value={fmtBool(a.pattern_day_trader)} />
      <Stat label="Shorting enabled" value={fmtBool(a.shorting_enabled)} />
      <Stat label="Trading blocked" value={fmtBool(a.trading_blocked)} />
      <Stat label="Transfers blocked" value={fmtBool(a.transfers_blocked)} />
      <Stat label="Account blocked" value={fmtBool(a.account_blocked)} />
      <Stat label="Trade suspended (user)" value={fmtBool(a.trade_suspended_by_user)} />
    </SimpleGrid>
  )
}

const PERIOD_OPTIONS = ['1D', '1W', '1M', '3M', '6M', '1A'].map((v) => ({ value: v, label: v }))
const TIMEFRAME_OPTIONS = [
  { value: '', label: 'Default' },
  { value: '1Min', label: '1Min' },
  { value: '5Min', label: '5Min' },
  { value: '15Min', label: '15Min' },
  { value: '1H', label: '1H' },
  { value: '1D', label: '1D' },
]

export default function TradeAccountPage() {
  const [tradeToken, setTradeToken] = useState('')
  const [period, setPeriod] = useState('1M')
  const [timeframe, setTimeframe] = useState<string | null>(null)

  const settingsQ = useQuery({ queryKey: ['appSettings'], queryFn: getAppSettings })
  const alpacaEnabled = settingsQ.data?.environment.alpaca_enabled ?? false
  const tokenReady = tradeToken.trim().length > 0
  const fetchEnabled = alpacaEnabled && tokenReady

  const equityQ = useQuery({
    queryKey: ['tradeAccountEquity', tradeToken],
    queryFn: () => getTradeAccountEquity(tradeToken.trim()),
    enabled: fetchEnabled,
  })

  const historyQ = useQuery({
    queryKey: ['tradePortfolioHistory', tradeToken, period, timeframe],
    queryFn: () =>
      getTradePortfolioHistory(tradeToken.trim(), {
        period,
        timeframe: timeframe || null,
      }),
    enabled: fetchEnabled,
  })

  const activitiesQ = useInfiniteQuery({
    queryKey: ['tradeAccountActivities', tradeToken],
    queryFn: ({ pageParam }) =>
      getTradeAccountActivities(tradeToken.trim(), {
        page_size: 50,
        page_token: pageParam ?? undefined,
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_page_token ?? undefined,
    enabled: fetchEnabled,
  })

  const chartRows = useMemo(() => {
    const h = historyQ.data
    if (!h?.timestamp?.length) return []
    return h.timestamp.map((ts, i) => ({
      t: ts * 1000,
      label: new Date(ts * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric' }),
      equity: h.equity[i] ?? 0,
    }))
  }, [historyQ.data])

  const flatActivities = useMemo(
    () => activitiesQ.data?.pages.flatMap((p) => p.activities) ?? [],
    [activitiesQ.data],
  )

  return (
    <PageScaffold>
      <Title order={2}>Account</Title>
      <Text size="sm" c="dimmed">
        Alpaca brokerage snapshot (equities). Requires the API trade desk token and Alpaca enabled on the server.
      </Text>

      <ApiErrorAlert error={settingsQ.error} />
      <ApiErrorAlert error={equityQ.error} />
      <ApiErrorAlert error={historyQ.error} />
      <ApiErrorAlert error={activitiesQ.error} />

      {!alpacaEnabled && (
        <Alert color="yellow" title="Alpaca disabled">
          The API reports Alpaca as disabled (<code>SHUNYA_API_ALPACA_ENABLED</code>). Broker routes return 503 until
          it is enabled and keys are configured.
        </Alert>
      )}

      <Paper withBorder p="md" radius="md">
        <Stack gap="sm">
          <PasswordInput
            label="X-Shunya-Trade-Desk-Token"
            description="Same value as SHUNYA_API_TRADE_DESK_TOKEN on the API. Stored only in this component state."
            value={tradeToken}
            onChange={(e) => setTradeToken(e.currentTarget.value)}
            disabled={!alpacaEnabled}
          />
          <Group>
            <Button
              variant="light"
              disabled={!fetchEnabled}
              onClick={() => {
                void equityQ.refetch()
                void historyQ.refetch()
                void activitiesQ.refetch()
              }}
            >
              Refresh
            </Button>
          </Group>
        </Stack>
      </Paper>

      {equityQ.data && (
        <Stack gap="md">
          <Group justify="space-between" align="flex-end">
            <div>
              <Title order={4}>Balances</Title>
              <Text size="sm" c="dimmed">
                Account {equityQ.data.account_number ?? equityQ.data.id ?? '—'} · {equityQ.data.status ?? '—'}
              </Text>
            </div>
            {equityQ.data.currency && <Badge variant="light">{equityQ.data.currency}</Badge>}
          </Group>
          {equityStats(equityQ.data)}
        </Stack>
      )}

      {fetchEnabled && (
        <Paper withBorder p="md" radius="md">
          <Stack gap="md">
            <Title order={4}>Portfolio history</Title>
            <Group>
              <Select label="Period" data={PERIOD_OPTIONS} value={period} onChange={(v) => setPeriod(v ?? '1M')} />
              <Select
                label="Timeframe"
                data={TIMEFRAME_OPTIONS}
                value={timeframe ?? ''}
                onChange={(v) => setTimeframe(v || null)}
                clearable
              />
            </Group>
            {historyQ.isLoading && <Text size="sm">Loading chart…</Text>}
            {chartRows.length > 0 && (
              <div style={{ width: '100%', height: 320 }}>
                <ResponsiveContainer>
                  <LineChart data={chartRows} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="label" minTickGap={24} tick={{ fontSize: 11 }} />
                    <YAxis
                      domain={['auto', 'auto']}
                      tickFormatter={(v) =>
                        Number(v).toLocaleString(undefined, { notation: 'compact', maximumFractionDigits: 1 })
                      }
                      width={56}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      formatter={(value: number) =>
                        value.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 })
                      }
                    />
                    <Line type="monotone" dataKey="equity" stroke="var(--mantine-color-blue-6)" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
            {!historyQ.isLoading && chartRows.length === 0 && (
              <Text size="sm" c="dimmed">
                No portfolio history points for this range.
              </Text>
            )}
          </Stack>
        </Paper>
      )}

      {fetchEnabled && (
        <Paper withBorder p="md" radius="md">
          <Title order={4} mb="sm">
            Recent activity
          </Title>
          {activitiesQ.isLoading && <Text size="sm">Loading…</Text>}
          <ScrollArea h={360} type="auto">
            <Table striped highlightOnHover withTableBorder stickyHeader>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Time</Table.Th>
                  <Table.Th>Type</Table.Th>
                  <Table.Th>Symbol</Table.Th>
                  <Table.Th>Side</Table.Th>
                  <Table.Th>Qty</Table.Th>
                  <Table.Th>Price</Table.Th>
                  <Table.Th>Net</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {flatActivities.map((row) => (
                  <Table.Tr key={row.id}>
                    <Table.Td style={{ whiteSpace: 'nowrap' }}>
                      {row.transaction_time ?? row.date ?? '—'}
                    </Table.Td>
                    <Table.Td>{row.activity_type}</Table.Td>
                    <Table.Td>{row.symbol ?? '—'}</Table.Td>
                    <Table.Td>{row.side ?? '—'}</Table.Td>
                    <Table.Td>{row.qty != null ? String(row.qty) : '—'}</Table.Td>
                    <Table.Td>{row.price != null ? String(row.price) : '—'}</Table.Td>
                    <Table.Td>{row.net_amount != null ? String(row.net_amount) : '—'}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
          {activitiesQ.hasNextPage && (
            <Button
              variant="default"
              mt="sm"
              loading={activitiesQ.isFetchingNextPage}
              onClick={() => void activitiesQ.fetchNextPage()}
            >
              Load more
            </Button>
          )}
        </Paper>
      )}
    </PageScaffold>
  )
}
