import {
  Alert,
  Anchor,
  Badge,
  Button,
  Card,
  Divider,
  Group,
  SimpleGrid,
  Stack,
  Table,
  Tabs,
  Text,
  Title,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import {
  getInstrumentNews,
  getInstrumentOhlcv,
  getInstrumentOverview,
  instrumentDetailPath,
} from '../api/endpoints'
import type { InstrumentOverviewResponse } from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import InstrumentChart from '../components/InstrumentChart'
import InstrumentContextFeed from '../components/InstrumentContextFeed'
import InstrumentFinancialsPanel from '../components/instrument/InstrumentFinancialsPanel'
import InstrumentHoldersPanel from '../components/instrument/InstrumentHoldersPanel'
import InstrumentOptionsChainPanel from '../components/instrument/InstrumentOptionsChainPanel'
import InstrumentYfinanceResearchTab from '../components/instrument/InstrumentYfinanceResearchTab'
import PageScaffold from '../components/PageScaffold'
import { barTimesUtcSeconds, snapCrosshairToBarTime } from '../utils/chartBarTimes'

type Preset = { id: string; label: string; interval: string; period: string }

const TIMEFRAMES: Preset[] = [
  { id: '1d', label: '1D', interval: '5m', period: '1d' },
  { id: '5d', label: '5D', interval: '15m', period: '5d' },
  { id: '1m', label: '1M', interval: '1h', period: '1mo' },
  { id: '3m', label: '3M', interval: '1d', period: '3mo' },
  { id: '6m', label: '6M', interval: '1d', period: '6mo' },
  { id: '1y', label: '1Y', interval: '1d', period: '1y' },
  { id: '5y', label: '5Y', interval: '1wk', period: '5y' },
  { id: 'max', label: 'All', interval: '1mo', period: 'max' },
]

function normalizeSymbol(raw: string | undefined): string | null {
  if (!raw) return null
  const s = decodeURIComponent(raw).trim().toUpperCase()
  if (!/^[A-Z0-9^.-]{1,32}$/.test(s)) return null
  return s
}

function fmtCap(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  return n.toLocaleString()
}

function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return n.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function OverviewBody({ o }: { o: InstrumentOverviewResponse }) {
  const v = o.valuation
  return (
    <Stack gap="lg">
      <SimpleGrid cols={{ base: 2, sm: 3, md: 4 }} spacing="md">
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            Market cap
          </Text>
          <Text fw={600} ff="monospace">
            {fmtCap(o.market_cap)}
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            Beta
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(o.beta, 3)}
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            Trailing P/E
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(v.trailing_pe, 2)}
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            Forward P/E
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(v.forward_pe, 2)}
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            EPS (trail / fwd)
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(v.trailing_eps, 2)} / {fmtNum(v.forward_eps, 2)}
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            ROE / ROA
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(v.return_on_equity != null ? v.return_on_equity * 100 : null, 1)}% /{' '}
            {fmtNum(v.return_on_assets != null ? v.return_on_assets * 100 : null, 1)}%
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            P/B · P/S
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(v.price_to_book, 2)} · {fmtNum(v.price_to_sales, 2)}
          </Text>
        </Card>
        <Card withBorder padding="sm" radius="md">
          <Text size="xs" c="dimmed" tt="uppercase">
            Debt / Equity
          </Text>
          <Text fw={600} ff="monospace">
            {fmtNum(v.debt_to_equity, 2)}
          </Text>
        </Card>
      </SimpleGrid>

      {o.instrument_kind === 'option' && o.option_contract && (
        <Card withBorder padding="md" radius="md">
          <Title order={5} mb="sm">
            Option contract
          </Title>
          <Stack gap="xs">
            {o.option_contract.underlying_symbol && (
              <Group gap="xs">
                <Text size="sm" c="dimmed">
                  Underlying
                </Text>
                <Anchor
                  component={Link}
                  to={instrumentDetailPath(o.option_contract.underlying_symbol)}
                  ff="monospace"
                  size="sm"
                >
                  {o.option_contract.underlying_symbol}
                </Anchor>
              </Group>
            )}
            <Text size="sm">
              <Text span c="dimmed">
                Strike / expiry / type:{' '}
              </Text>
              <Text span ff="monospace">
                {fmtNum(o.option_contract.strike, 2)} · {o.option_contract.expire_date ?? '—'} ·{' '}
                {o.option_contract.contract_type ?? '—'}
              </Text>
            </Text>
            <Text size="sm">
              <Text span c="dimmed">
                Last / bid / ask:{' '}
              </Text>
              <Text span ff="monospace">
                {fmtNum(o.option_contract.last_price, 2)} / {fmtNum(o.option_contract.bid, 2)} /{' '}
                {fmtNum(o.option_contract.ask, 2)}
              </Text>
            </Text>
            <Text size="sm">
              <Text span c="dimmed">
                Volume / OI / IV:{' '}
              </Text>
              <Text span ff="monospace">
                {o.option_contract.volume ?? '—'} / {o.option_contract.open_interest ?? '—'} /{' '}
                {o.option_contract.implied_volatility != null
                  ? `${(o.option_contract.implied_volatility * 100).toFixed(1)}%`
                  : '—'}
              </Text>
            </Text>
          </Stack>
        </Card>
      )}

      {o.company && (
        <Card withBorder padding="md" radius="md">
          <Title order={5} mb="sm">
            Company
          </Title>
          <Stack gap="xs">
            <Text size="sm">
              <Text span c="dimmed">
                Sector / industry:{' '}
              </Text>
              {o.company.sector ?? '—'} · {o.company.industry ?? '—'}
            </Text>
            {o.company.website && (
              <Anchor href={o.company.website} target="_blank" rel="noopener noreferrer" size="sm">
                {o.company.website}
              </Anchor>
            )}
            {(o.company.address_line1 || o.company.city) && (
              <Text size="sm" c="dimmed">
                {[o.company.address_line1, o.company.city, o.company.state, o.company.zip_code, o.company.country]
                  .filter(Boolean)
                  .join(', ')}
              </Text>
            )}
            {o.company.long_business_summary && (
              <Text size="sm" style={{ whiteSpace: 'pre-wrap' }}>
                {o.company.long_business_summary}
              </Text>
            )}
          </Stack>
        </Card>
      )}

      {o.fund && (
        <Card withBorder padding="md" radius="md">
          <Title order={5} mb="sm">
            Fund profile
          </Title>
          <Stack gap="sm">
            <Text size="sm">
              {o.fund.fund_family ?? '—'} · {o.fund.category ?? '—'}
            </Text>
            <Group gap="xl">
              <Text size="sm">
                <Text span c="dimmed">
                  Expense ratio:{' '}
                </Text>
                {o.fund.expense_ratio != null ? `${fmtNum(o.fund.expense_ratio, 3)}%` : '—'}
              </Text>
              <Text size="sm">
                <Text span c="dimmed">
                  Yield:{' '}
                </Text>
                {o.fund.yield_pct != null ? `${fmtNum(o.fund.yield_pct, 2)}%` : '—'}
              </Text>
            </Group>
            {(o.fund.top_holdings?.length ?? 0) > 0 && (
              <Table.ScrollContainer minWidth={320}>
                <Table striped verticalSpacing="xs">
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Symbol</Table.Th>
                      <Table.Th>Name</Table.Th>
                      <Table.Th>%</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {(o.fund.top_holdings ?? []).map((h) => (
                      <Table.Tr key={h.symbol}>
                        <Table.Td ff="monospace">{h.symbol}</Table.Td>
                        <Table.Td>{h.name ?? '—'}</Table.Td>
                        <Table.Td ff="monospace">{h.holding_percent != null ? fmtNum(h.holding_percent, 2) : '—'}</Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </Table.ScrollContainer>
            )}
          </Stack>
        </Card>
      )}

      {o.executives.length > 0 && (
        <Card withBorder padding="md" radius="md">
          <Title order={5} mb="sm">
            Key executives
          </Title>
          <Table striped>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Name</Table.Th>
                <Table.Th>Title</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {o.executives.map((e, i) => (
                <Table.Tr key={`${e.name ?? ''}-${i}`}>
                  <Table.Td>{e.name ?? '—'}</Table.Td>
                  <Table.Td>{e.title ?? '—'}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Card>
      )}
    </Stack>
  )
}

export default function InstrumentDetailPage() {
  const { symbol: symbolParam } = useParams<{ symbol: string }>()
  const [searchParams] = useSearchParams()
  const qtHint = searchParams.get('qt')
  const symbol = useMemo(() => normalizeSymbol(symbolParam), [symbolParam])
  const [mainTab, setMainTab] = useState<string>('overview')
  const [preset, setPreset] = useState<Preset>(() => TIMEFRAMES[5]!)
  const [focusBarUnix, setFocusBarUnix] = useState<number | null>(null)

  const overview = useQuery({
    queryKey: ['instrument-overview', symbol],
    queryFn: () => getInstrumentOverview(symbol!),
    enabled: symbol != null,
    staleTime: 60_000,
  })

  const ohlcv = useQuery({
    queryKey: ['instrument-ohlcv', symbol, preset.interval, preset.period],
    queryFn: () =>
      getInstrumentOhlcv(symbol!, {
        interval: preset.interval,
        period: preset.period,
      }),
    enabled: symbol != null,
  })

  const newsQuery = useQuery({
    queryKey: ['instrument-news', symbol],
    queryFn: () => getInstrumentNews(symbol!),
    enabled: symbol != null,
    staleTime: 60_000,
  })

  const newsRows = newsQuery.data?.news ?? []

  const barTimes = useMemo(
    () => barTimesUtcSeconds(ohlcv.data?.bars ?? []),
    [ohlcv.data?.bars],
  )

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset when ticker changes
    setFocusBarUnix(null)
  }, [symbol])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset tab on navigation
    setMainTab('overview')
  }, [symbol])

  const onCrosshairBarUtc = useCallback(
    (utc: number | null) => {
      setFocusBarUnix(snapCrosshairToBarTime(utc, barTimes))
    },
    [barTimes],
  )

  const features = overview.data?.features
  const showFinancials = features?.financials ?? false
  const showHolders = features?.holders ?? false
  const showOptions = (features?.options_chain ?? false) && overview.data?.instrument_kind !== 'option'
  const showResearch =
    overview.data != null &&
    (overview.data.instrument_kind === 'equity' ||
      overview.data.instrument_kind === 'etf' ||
      overview.data.instrument_kind === 'mutualfund')

  if (!symbol) {
    return (
      <PageScaffold>
        <Title order={1}>Invalid symbol</Title>
        <Text c="dimmed" size="sm">
          Use a valid ticker (letters, numbers, . - ^).
        </Text>
        <Anchor component={Link} to="/search">
          Back to search
        </Anchor>
      </PageScaffold>
    )
  }

  const yahooUrl = `https://finance.yahoo.com/quote/${encodeURIComponent(symbol)}`
  const titleName = overview.data?.long_name ?? overview.data?.short_name ?? symbol

  return (
    <PageScaffold>
      <Group justify="space-between" align="flex-start" wrap="wrap">
        <Stack gap={4}>
          <Group gap="sm" align="center">
            <Title order={1} ff="monospace">
              {symbol}
            </Title>
            {overview.data && (
              <Badge variant="light" color="yellow" size="sm">
                {overview.data.instrument_kind}
              </Badge>
            )}
            {qtHint && (
              <Badge variant="outline" color="gray" size="xs">
                search: {qtHint}
              </Badge>
            )}
          </Group>
          {overview.data && (
            <Text size="sm" c="dimmed">
              {titleName}
              {overview.data.exchange ? ` · ${overview.data.exchange}` : ''}
            </Text>
          )}
        </Stack>
        <Group gap="md">
          <Anchor href={yahooUrl} target="_blank" rel="noopener noreferrer" size="sm">
            Yahoo Finance
          </Anchor>
          <Anchor component={Link} to={`/search?q=${encodeURIComponent(symbol)}`} size="sm">
            Search again
          </Anchor>
        </Group>
      </Group>

      <ApiErrorAlert error={overview.error} />

      <Tabs value={mainTab} onChange={(v) => setMainTab(v ?? 'overview')} mt="md">
        <Tabs.List>
          <Tabs.Tab value="overview">Overview</Tabs.Tab>
          <Tabs.Tab value="chart">Chart & news</Tabs.Tab>
          {showFinancials && <Tabs.Tab value="financials">Financials</Tabs.Tab>}
          {showHolders && <Tabs.Tab value="holders">Holders</Tabs.Tab>}
          {showResearch && <Tabs.Tab value="research">Research</Tabs.Tab>}
          {showOptions && <Tabs.Tab value="options">Options</Tabs.Tab>}
        </Tabs.List>

        <Tabs.Panel value="overview" pt="lg">
          {overview.isLoading && (
            <Text c="dimmed" size="sm">
              Loading overview…
            </Text>
          )}
          {overview.data && <OverviewBody o={overview.data} />}
        </Tabs.Panel>

        <Tabs.Panel value="chart" pt="lg">
          <Button.Group mb="md">
            {TIMEFRAMES.map((p) => (
              <Button
                key={p.id}
                variant={p.id === preset.id ? 'filled' : 'default'}
                color={p.id === preset.id ? 'yellow' : undefined}
                size="compact-sm"
                onClick={() => {
                  setPreset(p)
                  setFocusBarUnix(null)
                }}
              >
                {p.label}
              </Button>
            ))}
          </Button.Group>
          {ohlcv.isLoading && (
            <Text c="dimmed" size="sm">
              Loading chart…
            </Text>
          )}
          <ApiErrorAlert error={ohlcv.error} />
          <ApiErrorAlert error={newsQuery.error} />
          {ohlcv.data?.storage_error && (
            <Alert color="red" variant="light">
              {ohlcv.data.storage_error}
            </Alert>
          )}
          {ohlcv.data?.storage_status === 'deferred' && ohlcv.data.storage_job_id != null && (
            <Text c="dimmed" size="sm">
              Database sync queued (job {ohlcv.data.storage_job_id}).
            </Text>
          )}
          {ohlcv.data && ohlcv.data.bars.length === 0 && !ohlcv.isLoading && (
            <Text c="dimmed" size="sm">
              No bars returned for this range.
            </Text>
          )}
          {ohlcv.data && ohlcv.data.bars.length > 0 && (
            <Group align="flex-start" gap="md" wrap="wrap" grow>
              <Stack style={{ flex: '2 1 420px', minWidth: 280 }}>
                <InstrumentChart
                  bars={ohlcv.data.bars}
                  news={newsRows}
                  onCrosshairBarUtc={onCrosshairBarUtc}
                  compactNewsTooltip
                  key={`${preset.id}-${symbol}`}
                />
              </Stack>
              <Stack style={{ flex: '1 1 280px', minWidth: 260 }}>
                <InstrumentContextFeed
                  news={newsRows}
                  barTimes={barTimes}
                  focusBarUnix={focusBarUnix}
                />
              </Stack>
            </Group>
          )}
        </Tabs.Panel>

        {showFinancials && (
          <Tabs.Panel value="financials" pt="lg">
            <InstrumentFinancialsPanel symbol={symbol} enabled={mainTab === 'financials'} />
          </Tabs.Panel>
        )}

        {showHolders && (
          <Tabs.Panel value="holders" pt="lg">
            <InstrumentHoldersPanel symbol={symbol} enabled={mainTab === 'holders'} />
          </Tabs.Panel>
        )}

        {showResearch && (
          <Tabs.Panel value="research" pt="lg">
            <InstrumentYfinanceResearchTab symbol={symbol} enabled={mainTab === 'research'} />
          </Tabs.Panel>
        )}

        {showOptions && (
          <Tabs.Panel value="options" pt="lg">
            <InstrumentOptionsChainPanel symbol={symbol} enabled={mainTab === 'options'} />
          </Tabs.Panel>
        )}
      </Tabs>

      <Divider my="xl" />
      <Text size="xs" c="dimmed">
        Market data and fundamentals are sourced from Yahoo Finance via yfinance; fields may be missing or delayed
        depending on the symbol.
      </Text>
    </PageScaffold>
  )
}
