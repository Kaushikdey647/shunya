import {
  Accordion,
  ScrollArea,
  SimpleGrid,
  Stack,
  Table,
  Text,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import {
  getInstrumentAnalystPriceTargets,
  getInstrumentCalendar,
  getInstrumentEarningsEstimate,
  getInstrumentEarningsHistory,
  getInstrumentEpsRevisions,
  getInstrumentEpsTrend,
  getInstrumentGrowthEstimates,
  getInstrumentInsiderPurchases,
  getInstrumentInsiderRosterHolders,
  getInstrumentInsiderTransactions,
  getInstrumentMajorHoldersYf,
  getInstrumentRecommendations,
  getInstrumentRecommendationsSummary,
  getInstrumentRevenueEstimate,
  getInstrumentSecFilings,
  getInstrumentSustainability,
  getInstrumentUpgradesDowngrades,
  getInstrumentValuationMeasures,
} from '../../api/endpoints'
import type {
  InstrumentAnalystPriceTargetsResponse,
  InstrumentJsonBlobResponse,
  InstrumentValuationMeasuresPayload,
  InstrumentYfinanceTableResponse,
} from '../../api/types'
import ApiErrorAlert from '../ApiErrorAlert'
import { useMantineTableDensity } from '../../hooks/useMantineTableDensity'

type Props = {
  symbol: string
  enabled: boolean
}

function fmtNum(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return n.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function cellStr(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'number' && !Number.isFinite(v)) return '—'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

function YfDataTable({ payload }: { payload: InstrumentYfinanceTableResponse['data'] }) {
  const tableProps = useMantineTableDensity()
  const { columns, records } = payload
  if (columns.length === 0 || records.length === 0) {
    return (
      <Text size="sm" c="dimmed">
        Empty table.
      </Text>
    )
  }
  return (
    <Table.ScrollContainer minWidth={480}>
      <Table {...tableProps} striped highlightOnHover>
        <Table.Thead>
          <Table.Tr>
            {columns.map((c) => (
              <Table.Th key={c} style={{ whiteSpace: 'nowrap' }}>
                {c}
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {records.map((row, ri) => (
            <Table.Tr key={ri}>
              {columns.map((c) => (
                <Table.Td key={c} ff="monospace" style={{ fontSize: '0.8rem' }}>
                  {cellStr(row[c])}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  )
}

function ValuationMeasuresBlock({ symbol, load }: { symbol: string; load: boolean }) {
  const q = useQuery({
    queryKey: ['instrument-valuation-measures', symbol],
    queryFn: () => getInstrumentValuationMeasures(symbol),
    enabled: load,
    staleTime: 120_000,
  })
  const p: InstrumentValuationMeasuresPayload | undefined = q.data
  return (
    <Stack gap="sm">
      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text size="sm" c="dimmed">
          Loading…
        </Text>
      )}
      {p && !p.available && (
        <Text size="sm" c="dimmed">
          No valuation table for this symbol.
        </Text>
      )}
      {p && p.available && <YfDataTable payload={{ columns: p.columns, records: p.records }} />}
    </Stack>
  )
}

function PriceTargetsBlock({ symbol, load }: { symbol: string; load: boolean }) {
  const q = useQuery({
    queryKey: ['instrument-analyst-price-targets', symbol],
    queryFn: () => getInstrumentAnalystPriceTargets(symbol),
    enabled: load,
    staleTime: 120_000,
  })
  const p: InstrumentAnalystPriceTargetsResponse | undefined = q.data
  return (
    <Stack gap="sm">
      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text size="sm" c="dimmed">
          Loading…
        </Text>
      )}
      {p && !p.available && (
        <Text size="sm" c="dimmed">
          No price target consensus for this symbol.
        </Text>
      )}
      {p && p.available && (
        <SimpleGrid cols={{ base: 2, sm: 3 }} spacing="md">
          {(
            [
              ['Current', p.current],
              ['Low', p.low],
              ['High', p.high],
              ['Mean', p.mean],
              ['Median', p.median],
            ] as const
          ).map(([label, v]) => (
            <Stack key={label} gap={4}>
              <Text size="xs" c="dimmed" tt="uppercase">
                {label}
              </Text>
              <Text fw={600} ff="monospace">
                {fmtNum(v, 2)}
              </Text>
            </Stack>
          ))}
        </SimpleGrid>
      )}
    </Stack>
  )
}

function YfTableBlock({
  symbol,
  load,
  queryKey,
  fetcher,
}: {
  symbol: string
  load: boolean
  queryKey: string
  fetcher: (s: string) => Promise<InstrumentYfinanceTableResponse>
}) {
  const q = useQuery({
    queryKey: ['instrument-research-table', queryKey, symbol],
    queryFn: () => fetcher(symbol),
    enabled: load,
    staleTime: 120_000,
  })
  const p = q.data
  return (
    <Stack gap="sm">
      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text size="sm" c="dimmed">
          Loading…
        </Text>
      )}
      {p && !p.available && (
        <Text size="sm" c="dimmed">
          No data for this section.
        </Text>
      )}
      {p && p.available && <YfDataTable payload={p.data} />}
    </Stack>
  )
}

function JsonBlobBlock({
  symbol,
  load,
  queryKey,
  fetcher,
}: {
  symbol: string
  load: boolean
  queryKey: string
  fetcher: (s: string) => Promise<InstrumentJsonBlobResponse>
}) {
  const q = useQuery({
    queryKey: ['instrument-research-json', queryKey, symbol],
    queryFn: () => fetcher(symbol),
    enabled: load,
    staleTime: 120_000,
  })
  const p = q.data
  const text = useMemo(
    () => (p?.data != null ? JSON.stringify(p.data, null, 2) : ''),
    [p?.data],
  )
  return (
    <Stack gap="sm">
      <ApiErrorAlert error={q.error} />
      {q.isLoading && (
        <Text size="sm" c="dimmed">
          Loading…
        </Text>
      )}
      {p && !p.available && (
        <Text size="sm" c="dimmed">
          No data for this section.
        </Text>
      )}
      {p && p.available && text.length > 0 && (
        <ScrollArea.Autosize mah={360}>
          <Text component="pre" size="xs" ff="monospace" style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
            {text}
          </Text>
        </ScrollArea.Autosize>
      )}
      {p && p.available && text.length === 0 && (
        <Text size="sm" c="dimmed">
          Empty payload.
        </Text>
      )}
    </Stack>
  )
}

/** Yahoo-backed analyst, estimates, insider, calendar, and filings; loads each accordion section on first expand. */
export default function InstrumentYfinanceResearchTab({ symbol, enabled }: Props) {
  const [opened, setOpened] = useState<string[]>([])

  const isOpen = (id: string) => enabled && opened.includes(id)

  return (
    <Stack gap="md">
      <Text size="sm" c="dimmed">
        Open a section to fetch that dataset (cached on the server). Options and IV heatmap stay on the Options tab.
      </Text>
      <Accordion multiple value={opened} onChange={setOpened}>
        <Accordion.Item value="valuation">
          <Accordion.Control>Valuation measures</Accordion.Control>
          <Accordion.Panel>
            <ValuationMeasuresBlock symbol={symbol} load={isOpen('valuation')} />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="price-targets">
          <Accordion.Control>Analyst price targets</Accordion.Control>
          <Accordion.Panel>
            <PriceTargetsBlock symbol={symbol} load={isOpen('price-targets')} />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="earnings-estimate">
          <Accordion.Control>Earnings estimate</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('earnings-estimate')}
              queryKey="earnings-estimate"
              fetcher={getInstrumentEarningsEstimate}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="revenue-estimate">
          <Accordion.Control>Revenue estimate</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('revenue-estimate')}
              queryKey="revenue-estimate"
              fetcher={getInstrumentRevenueEstimate}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="earnings-history">
          <Accordion.Control>Earnings history</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('earnings-history')}
              queryKey="earnings-history"
              fetcher={getInstrumentEarningsHistory}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="eps-trend">
          <Accordion.Control>EPS trend</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('eps-trend')}
              queryKey="eps-trend"
              fetcher={getInstrumentEpsTrend}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="eps-revisions">
          <Accordion.Control>EPS revisions</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('eps-revisions')}
              queryKey="eps-revisions"
              fetcher={getInstrumentEpsRevisions}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="growth-estimates">
          <Accordion.Control>Growth estimates</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('growth-estimates')}
              queryKey="growth-estimates"
              fetcher={getInstrumentGrowthEstimates}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="recommendations">
          <Accordion.Control>Recommendations</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('recommendations')}
              queryKey="recommendations"
              fetcher={getInstrumentRecommendations}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="recommendations-summary">
          <Accordion.Control>Recommendations summary</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('recommendations-summary')}
              queryKey="recommendations-summary"
              fetcher={getInstrumentRecommendationsSummary}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="upgrades-downgrades">
          <Accordion.Control>Upgrades / downgrades</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('upgrades-downgrades')}
              queryKey="upgrades-downgrades"
              fetcher={getInstrumentUpgradesDowngrades}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="sustainability">
          <Accordion.Control>Sustainability (ESG)</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('sustainability')}
              queryKey="sustainability"
              fetcher={getInstrumentSustainability}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="insider-purchases">
          <Accordion.Control>Insider purchases</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('insider-purchases')}
              queryKey="insider-purchases"
              fetcher={getInstrumentInsiderPurchases}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="insider-transactions">
          <Accordion.Control>Insider transactions</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('insider-transactions')}
              queryKey="insider-transactions"
              fetcher={getInstrumentInsiderTransactions}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="insider-roster">
          <Accordion.Control>Insider roster / holders</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('insider-roster')}
              queryKey="insider-roster"
              fetcher={getInstrumentInsiderRosterHolders}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="major-holders-yf">
          <Accordion.Control>Major holders (Yahoo breakdown)</Accordion.Control>
          <Accordion.Panel>
            <YfTableBlock
              symbol={symbol}
              load={isOpen('major-holders-yf')}
              queryKey="major-holders-yf"
              fetcher={getInstrumentMajorHoldersYf}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="calendar">
          <Accordion.Control>Calendar</Accordion.Control>
          <Accordion.Panel>
            <JsonBlobBlock
              symbol={symbol}
              load={isOpen('calendar')}
              queryKey="calendar"
              fetcher={getInstrumentCalendar}
            />
          </Accordion.Panel>
        </Accordion.Item>

        <Accordion.Item value="sec-filings">
          <Accordion.Control>SEC filings</Accordion.Control>
          <Accordion.Panel>
            <JsonBlobBlock
              symbol={symbol}
              load={isOpen('sec-filings')}
              queryKey="sec-filings"
              fetcher={getInstrumentSecFilings}
            />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Stack>
  )
}
