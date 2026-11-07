import {
  Alert,
  Badge,
  Button,
  Group,
  Modal,
  NumberInput,
  Paper,
  PasswordInput,
  ScrollArea,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import {
  createLiveBook,
  getLiveRun,
  getLiveStatus,
  listAlphas,
  listLiveRuns,
  listUniverses,
  liveDeploy,
  liveKill,
  liveResume,
  patchLiveSleeve,
} from '../api/endpoints'
import type { LiveBook, LiveKillLevel, LiveKillResult, LiveSleeve } from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useTradeDeskToken } from '../hooks/useTradeDeskToken'

const POLL_MS = 15_000
/** Cron fires every 15 min; older than this means the runner stopped ticking. */
const HEARTBEAT_STALE_S = 20 * 60

function fmtUsd(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
}

function fmtAge(s: number | null | undefined): string {
  if (s == null) return 'never'
  if (s < 90) return `${Math.round(s)}s ago`
  if (s < 5400) return `${Math.round(s / 60)}m ago`
  return `${(s / 3600).toFixed(1)}h ago`
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <Paper withBorder p="sm" radius="md">
      <Text size="xs" c="dimmed" tt="uppercase" fw={600}>
        {label}
      </Text>
      <Text size="md" fw={600} ff="monospace" c={color} style={{ fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </Text>
    </Paper>
  )
}

function KillPanel({ token, halted }: { token: string; halted: boolean }) {
  const qc = useQueryClient()
  const [level, setLevel] = useState<LiveKillLevel>('cancel')
  const [reason, setReason] = useState('')
  const [confirm, setConfirm] = useState(false)
  const [result, setResult] = useState<LiveKillResult | null>(null)
  const kill = useMutation({
    mutationFn: () => liveKill(token, { level, reason: reason.trim() }),
    onSuccess: (r) => {
      setResult(r)
      setConfirm(false)
      void qc.invalidateQueries({ queryKey: ['liveStatus'] })
    },
  })
  const resume = useMutation({
    mutationFn: () => liveResume(token, reason.trim()),
    onSuccess: () => {
      setResult(null)
      void qc.invalidateQueries({ queryKey: ['liveStatus'] })
    },
  })
  return (
    <Paper withBorder p="md" radius="md" style={{ borderColor: 'var(--mantine-color-red-6)' }}>
      <Stack gap="sm">
        <Title order={4}>Emergency kill</Title>
        <SegmentedControl
          value={level}
          onChange={(v) => setLevel(v as LiveKillLevel)}
          data={[
            { value: 'halt', label: 'Halt (no new orders)' },
            { value: 'cancel', label: 'Halt + cancel orders' },
            { value: 'flatten', label: 'Halt + flatten' },
          ]}
        />
        <TextInput
          label="Reason"
          required
          value={reason}
          onChange={(e) => setReason(e.currentTarget.value)}
          placeholder="Why are you stopping trading?"
        />
        <Group>
          <Button color="red" size="md" disabled={!reason.trim()} onClick={() => setConfirm(true)}>
            KILL
          </Button>
          {halted && (
            <Button variant="light" disabled={!reason.trim()} loading={resume.isPending} onClick={() => resume.mutate()}>
              Resume trading
            </Button>
          )}
        </Group>
        <ApiErrorAlert error={kill.error} />
        <ApiErrorAlert error={resume.error} />
        {result && (
          <Alert color={result.route === 'direct_alpaca' ? 'orange' : 'red'} title={`Kill sent (${result.route ?? 'hosted'})`}>
            <Text size="sm">
              Orders cancelled: {result.orders_cancelled ?? 0}
              {result.positions_closed != null ? ` · positions closed: ${result.positions_closed}` : ''}
            </Text>
            {result.warning && <Text size="sm">{result.warning}</Text>}
            {result.hosted_error && <Text size="xs" c="dimmed">Hosted: {result.hosted_error}</Text>}
            {result.broker_error && <Text size="sm" c="red">Broker: {result.broker_error}</Text>}
          </Alert>
        )}
      </Stack>
      <Modal opened={confirm} onClose={() => setConfirm(false)} title="Confirm kill" centered>
        <Stack>
          <Text>
            Level <strong>{level}</strong>: {level === 'flatten' ? 'cancel all orders and close all positions.' : level === 'cancel' ? 'stop the runner and cancel all open orders.' : 'stop the runner from sending new orders.'}
          </Text>
          <Group justify="flex-end">
            <Button variant="default" onClick={() => setConfirm(false)}>
              Back
            </Button>
            <Button color="red" loading={kill.isPending} onClick={() => kill.mutate()}>
              Confirm {level}
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Paper>
  )
}

function SleeveRow({ token, book, s }: { token: string; book: LiveBook; s: LiveSleeve }) {
  const qc = useQueryClient()
  const patch = useMutation({
    mutationFn: (p: Partial<Pick<LiveSleeve, 'weight' | 'status'>>) => patchLiveSleeve(token, book.id, s.id, p),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['liveStatus'] }),
  })
  return (
    <Table.Tr>
      <Table.Td>{s.name}</Table.Td>
      <Table.Td>{s.n_tickers}</Table.Td>
      <Table.Td w={110}>
        <NumberInput
          size="xs"
          min={0}
          max={1}
          step={0.05}
          decimalScale={3}
          defaultValue={s.weight}
          onBlur={(e) => {
            const w = Number(e.currentTarget.value)
            if (Number.isFinite(w) && w !== s.weight) patch.mutate({ weight: w })
          }}
        />
      </Table.Td>
      <Table.Td w={140}>
        <Select
          size="xs"
          value={s.status}
          data={['active', 'paused', 'draining', 'retired']}
          onChange={(v) => v && v !== s.status && patch.mutate({ status: v as LiveSleeve['status'] })}
        />
      </Table.Td>
      <Table.Td ff="monospace">{fmtUsd(s.market_value)}</Table.Td>
      <Table.Td>{Object.keys(s.positions).length}</Table.Td>
      <Table.Td>
        {s.runs_completed}/{s.ramp_days}
        {patch.error && <Text size="xs" c="red">{String((patch.error as Error).message)}</Text>}
      </Table.Td>
    </Table.Tr>
  )
}

function DeployPanel({ token, book }: { token: string; book: LiveBook | undefined }) {
  const qc = useQueryClient()
  const [alphaId, setAlphaId] = useState<string | null>(null)
  const [universeId, setUniverseId] = useState<string | null>(null)
  const [weight, setWeight] = useState<number>(0.5)
  const [lookback, setLookback] = useState<number>(400)
  const alphasQ = useQuery({ queryKey: ['alphas', 'all'], queryFn: () => listAlphas({ limit: 200 }) })
  const universesQ = useQuery({ queryKey: ['universes', 'all'], queryFn: () => listUniverses({ limit: 200 }) })
  const deploy = useMutation({
    mutationFn: () =>
      liveDeploy(token, {
        alpha_id: alphaId!,
        universe_id: universeId!,
        lookback_days: lookback,
        book_id: book?.id,
        weight,
      }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['liveStatus'] }),
  })
  return (
    <Paper withBorder p="md" radius="md">
      <Stack gap="sm">
        <Title order={4}>Deploy alpha as sleeve</Title>
        <Text size="xs" c="dimmed">
          Freezes the alpha source, FinStrat config and universe tickers and pushes them to the hosted runner. Live v1
          supports neutralization none/market and no fundamentals. Live bars come from the Alpaca IEX feed (split/dividend adjusted).
        </Text>
        <SimpleGrid cols={{ base: 1, md: 4 }}>
          <Select
            label="Alpha"
            searchable
            value={alphaId}
            onChange={setAlphaId}
            data={(alphasQ.data ?? []).map((a) => ({ value: a.id, label: a.name }))}
          />
          <Select
            label="Universe"
            searchable
            value={universeId}
            onChange={setUniverseId}
            data={(universesQ.data ?? []).map((u) => ({ value: u.id, label: `${u.name} (${u.member_count})` }))}
          />
          <NumberInput label="Sleeve weight" min={0} max={1} step={0.05} value={weight} onChange={(v) => setWeight(Number(v))} />
          <NumberInput label="Lookback (days)" min={30} max={3650} value={lookback} onChange={(v) => setLookback(Number(v))} />
        </SimpleGrid>
        <Group>
          <Button disabled={!alphaId || !universeId || !book} loading={deploy.isPending} onClick={() => deploy.mutate()}>
            Deploy to {book?.name ?? '(create a book first)'}
          </Button>
          {deploy.data && (
            <Text size="sm" c="green">
              Deployed {deploy.data.n_tickers} tickers (sleeve {deploy.data.sleeve_id ?? 'not attached'})
            </Text>
          )}
        </Group>
        {(deploy.data?.warnings ?? []).map((w) => (
          <Alert key={w} color="yellow">
            {w}
          </Alert>
        ))}
        <ApiErrorAlert error={deploy.error} />
      </Stack>
    </Paper>
  )
}

function CreateBookPanel({ token }: { token: string }) {
  const qc = useQueryClient()
  const [name, setName] = useState('main')
  const [fraction, setFraction] = useState<number>(0.5)
  const [maxDd, setMaxDd] = useState<number>(0.15)
  const create = useMutation({
    mutationFn: () =>
      createLiveBook(token, { name, capital_fraction: fraction, risk_config: { max_drawdown: maxDd } }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['liveStatus'] }),
  })
  return (
    <Paper withBorder p="md" radius="md">
      <Stack gap="sm">
        <Title order={4}>Create book</Title>
        <SimpleGrid cols={{ base: 1, md: 3 }}>
          <TextInput label="Name" value={name} onChange={(e) => setName(e.currentTarget.value)} />
          <NumberInput label="Capital fraction of equity" min={0.01} max={1} step={0.05} value={fraction} onChange={(v) => setFraction(Number(v))} />
          <NumberInput label="Max drawdown (auto-halt)" min={0.01} max={0.9} step={0.01} value={maxDd} onChange={(v) => setMaxDd(Number(v))} />
        </SimpleGrid>
        <Group>
          <Button onClick={() => create.mutate()} loading={create.isPending} disabled={!name.trim()}>
            Create
          </Button>
        </Group>
        <ApiErrorAlert error={create.error} />
      </Stack>
    </Paper>
  )
}

function RunsPanel({ token, bookId }: { token: string; bookId: string | undefined }) {
  const [openRun, setOpenRun] = useState<string | null>(null)
  const runsQ = useQuery({
    queryKey: ['liveRuns', token, bookId],
    queryFn: () => listLiveRuns(token, bookId),
    enabled: Boolean(bookId),
    refetchInterval: POLL_MS * 4,
  })
  const runQ = useQuery({
    queryKey: ['liveRun', token, openRun],
    queryFn: () => getLiveRun(token, openRun!),
    enabled: Boolean(openRun),
  })
  return (
    <Paper withBorder p="md" radius="md">
      <Stack gap="sm">
        <Title order={4}>Runs</Title>
        <ApiErrorAlert error={runsQ.error} />
        <ScrollArea>
          <Table striped highlightOnHover>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Session</Table.Th>
                <Table.Th>Signal</Table.Th>
                <Table.Th>Status</Table.Th>
                <Table.Th>Orders</Table.Th>
                <Table.Th>Equity</Table.Th>
                <Table.Th>Error</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {(runsQ.data ?? []).map((r) => (
                <Table.Tr key={r.id} onClick={() => setOpenRun(r.id)} style={{ cursor: 'pointer' }}>
                  <Table.Td>{r.session_date}</Table.Td>
                  <Table.Td>{r.signal_date}</Table.Td>
                  <Table.Td>
                    <Badge color={r.status === 'failed' ? 'red' : r.status === 'reconciled' ? 'green' : 'blue'}>{r.status}</Badge>
                  </Table.Td>
                  <Table.Td>{String(r.detail?.n_orders ?? '—')}</Table.Td>
                  <Table.Td ff="monospace">{fmtUsd(r.equity)}</Table.Td>
                  <Table.Td>
                    <Text size="xs" c="red" lineClamp={2}>
                      {r.error ?? ''}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Stack>
      <Modal opened={Boolean(openRun)} onClose={() => setOpenRun(null)} title="Run orders" size="xl">
        <ApiErrorAlert error={runQ.error} />
        <Table striped>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Symbol</Table.Th>
              <Table.Th>Side</Table.Th>
              <Table.Th>Qty</Table.Th>
              <Table.Th>Status</Table.Th>
              <Table.Th>Filled</Table.Th>
              <Table.Th>Avg px</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {(runQ.data?.orders ?? []).map((o) => (
              <Table.Tr key={o.client_order_id}>
                <Table.Td>{o.symbol}</Table.Td>
                <Table.Td>{o.side}</Table.Td>
                <Table.Td>{o.qty}</Table.Td>
                <Table.Td>{o.status}{o.error ? ` — ${o.error}` : ''}</Table.Td>
                <Table.Td>{o.filled_qty}</Table.Td>
                <Table.Td ff="monospace">{o.filled_avg_price?.toFixed(2) ?? '—'}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Modal>
    </Paper>
  )
}

/** Local dashboard for the hosted daily runner: polls status, kill switch, books/sleeves, deploy, runs. */
export default function LivePage() {
  const { draftToken, setDraftToken, appliedToken, applyToken } = useTradeDeskToken()
  const statusQ = useQuery({
    queryKey: ['liveStatus', appliedToken],
    queryFn: () => getLiveStatus(appliedToken),
    enabled: appliedToken.length > 0,
    refetchInterval: POLL_MS,
  })
  const st = statusQ.data
  const halted = st?.control?.state === 'halted'
  const hbAge = st?.heartbeat?.age_seconds ?? null
  const hbStale = st?.hosted_reachable && (hbAge == null || hbAge > HEARTBEAT_STALE_S)
  const activeBook = st?.books?.find((b) => b.status === 'active')
  const mismatches = Object.entries(st?.mismatches ?? {})

  return (
    <PageScaffold>
      <Title order={2}>Live</Title>
      <Text size="sm" c="dimmed">
        Daily market-on-open runner on the hosted service. This page polls it every {POLL_MS / 1000}s through the local
        API.
      </Text>

      <Paper withBorder p="md" radius="md">
        <Group align="flex-end">
          <PasswordInput
            label="X-Shunya-Trade-Desk-Token"
            value={draftToken}
            onChange={(e) => setDraftToken(e.currentTarget.value)}
            style={{ flex: 1 }}
          />
          <Button onClick={applyToken} disabled={!draftToken.trim()}>
            Apply token
          </Button>
        </Group>
      </Paper>

      <ApiErrorAlert error={statusQ.error} />
      {st && !st.hosted_reachable && (
        <Alert color="red" title="Hosted runner unreachable">
          {st.hosted_error}. The kill button still works: it falls back to cancelling directly at Alpaca.
        </Alert>
      )}
      {hbStale && (
        <Alert color="red" title="Runner heartbeat stale">
          Last tick {fmtAge(hbAge)} (expected every 15 min). Check the cron service.
        </Alert>
      )}
      {mismatches.length > 0 && (
        <Alert color="orange" title="Sleeve ledger differs from Alpaca positions">
          {mismatches.map(([sym, m]) => `${sym}: ledger ${m.ledger} vs broker ${m.broker}`).join(' · ')}
        </Alert>
      )}
      {(st?.errors ?? []).length > 0 && (
        <Alert color="yellow" title="Partial status">
          {(st?.errors ?? []).join(' · ')}
        </Alert>
      )}

      {st && (
        <SimpleGrid cols={{ base: 2, md: 6 }} spacing="sm">
          <Stat
            label="Trading"
            value={st.control ? (halted ? `HALTED (${st.control.level ?? 'halt'})` : 'ARMED') : '—'}
            color={halted ? 'red' : 'green'}
          />
          <Stat label="Heartbeat" value={`${fmtAge(hbAge)} · ${st.heartbeat?.last_result ?? '—'}`} color={hbStale ? 'red' : undefined} />
          <Stat
            label="Market"
            value={st.clock ? (st.clock.is_open ? 'OPEN' : `opens ${new Date(st.clock.next_open).toLocaleString()}`) : '—'}
          />
          <Stat label="Equity" value={fmtUsd(st.account?.equity)} />
          <Stat label="Buying power" value={fmtUsd(st.account?.buying_power)} />
          <Stat label="Open orders" value={String(st.open_orders?.length ?? 0)} />
        </SimpleGrid>
      )}
      {halted && st?.control?.reason && (
        <Text size="sm" c="red">
          Halted by {st.control.actor ?? '?'}: {st.control.reason}
        </Text>
      )}

      {appliedToken && <KillPanel token={appliedToken} halted={halted} />}

      {appliedToken && st?.hosted_reachable && (
        <>
          {activeBook ? (
            <Paper withBorder p="md" radius="md">
              <Stack gap="sm">
                <Group justify="space-between">
                  <Title order={4}>Book: {activeBook.name}</Title>
                  <Text size="sm" c="dimmed">
                    {Math.round(activeBook.capital_fraction * 100)}% of equity · HWM {fmtUsd(activeBook.hwm_equity)}
                  </Text>
                </Group>
                <ScrollArea>
                  <Table>
                    <Table.Thead>
                      <Table.Tr>
                        <Table.Th>Sleeve</Table.Th>
                        <Table.Th>Tickers</Table.Th>
                        <Table.Th>Weight</Table.Th>
                        <Table.Th>Status</Table.Th>
                        <Table.Th>Market value</Table.Th>
                        <Table.Th>Names held</Table.Th>
                        <Table.Th>Ramp</Table.Th>
                      </Table.Tr>
                    </Table.Thead>
                    <Table.Tbody>
                      {activeBook.sleeves.map((s) => (
                        <SleeveRow key={s.id} token={appliedToken} book={activeBook} s={s} />
                      ))}
                    </Table.Tbody>
                  </Table>
                </ScrollArea>
              </Stack>
            </Paper>
          ) : (
            <CreateBookPanel token={appliedToken} />
          )}
          <DeployPanel token={appliedToken} book={activeBook} />
          <RunsPanel token={appliedToken} bookId={activeBook?.id} />
        </>
      )}

      {st?.positions && st.positions.length > 0 && (
        <Paper withBorder p="md" radius="md">
          <Title order={4} mb="sm">
            Alpaca positions
          </Title>
          <ScrollArea>
            <Table striped>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Symbol</Table.Th>
                  <Table.Th>Qty</Table.Th>
                  <Table.Th>Price</Table.Th>
                  <Table.Th>Market value</Table.Th>
                  <Table.Th>Avg entry</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {st.positions.map((p) => (
                  <Table.Tr key={p.symbol}>
                    <Table.Td>{p.symbol}</Table.Td>
                    <Table.Td>{p.qty}</Table.Td>
                    <Table.Td ff="monospace">{p.current_price.toFixed(2)}</Table.Td>
                    <Table.Td ff="monospace">{fmtUsd(p.market_value)}</Table.Td>
                    <Table.Td ff="monospace">{p.avg_entry_price.toFixed(2)}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </ScrollArea>
        </Paper>
      )}
    </PageScaffold>
  )
}
