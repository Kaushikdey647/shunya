import {
  Anchor,
  Button,
  Checkbox,
  Group,
  NumberInput,
  Select,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  deleteBacktest,
  deleteBacktestsBatch,
  listAlphas,
  listBacktests,
} from '../api/endpoints'
import type { BacktestJobStatus } from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'

const STATUS_OPTIONS: { value: string; label: string }[] = [
  { value: 'queued', label: 'queued' },
  { value: 'running', label: 'running' },
  { value: 'succeeded', label: 'succeeded' },
  { value: 'failed', label: 'failed' },
]

function formatSummaryNumber(v: unknown, opts: { suffix?: string; decimals?: number } = {}): string {
  if (v === null || v === undefined || typeof v !== 'number' || Number.isNaN(v)) return '—'
  const { suffix = '', decimals = 2 } = opts
  return `${v.toFixed(decimals)}${suffix}`
}

export default function BacktestsListPage() {
  const density = useMantineTableDensity()
  const [limit, setLimit] = useState(50)
  const [offset, setOffset] = useState(0)
  const [alphaIdFilter, setAlphaIdFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState<BacktestJobStatus | ''>('')
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const qc = useQueryClient()

  const alphasQ = useQuery({
    queryKey: ['alphas', 'for-filter'],
    queryFn: () => listAlphas({ limit: 500, offset: 0 }),
  })

  const alphaFilterParam = useMemo(() => {
    return alphaIdFilter.trim() || null
  }, [alphaIdFilter])

  const q = useQuery({
    queryKey: ['backtests', limit, offset, alphaFilterParam, statusFilter],
    queryFn: () =>
      listBacktests({
        limit,
        offset,
        alpha_id: alphaFilterParam,
        status: statusFilter || null,
      }),
  })

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- selection is page-scoped; clear when filters/paging change
    setSelected(new Set())
  }, [limit, offset, alphaFilterParam, statusFilter])

  const rows = useMemo(() => q.data ?? [], [q.data])

  const pageIds = useMemo(() => rows.map((j) => j.id), [rows])
  const allOnPageSelected =
    pageIds.length > 0 && pageIds.every((id) => selected.has(id))
  const someOnPageSelected = pageIds.some((id) => selected.has(id))

  const delMut = useMutation({
    mutationFn: async (ids: string[]) => {
      if (ids.length === 1) {
        await deleteBacktest(ids[0])
        return { deleted: 1 }
      }
      return deleteBacktestsBatch(ids)
    },
    onSuccess: (_, ids) => {
      void qc.invalidateQueries({ queryKey: ['backtests'] })
      setSelected((prev) => {
        const next = new Set(prev)
        ids.forEach((id) => next.delete(id))
        return next
      })
    },
  })

  const toggleAllOnPage = () => {
    if (!rows.length) return
    const ids = rows.map((j) => j.id)
    const allSelected = ids.every((id) => selected.has(id))
    if (allSelected) {
      setSelected((prev) => {
        const next = new Set(prev)
        ids.forEach((id) => next.delete(id))
        return next
      })
    } else {
      setSelected((prev) => {
        const next = new Set(prev)
        ids.forEach((id) => next.add(id))
        return next
      })
    }
  }

  const toggleOne = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const confirmDeleteJobs = (ids: string[]) => {
    if (
      !window.confirm(
        `Delete ${ids.length} backtest job(s)? This cannot be undone.`,
      )
    ) {
      return
    }
    delMut.mutate(ids)
  }

  const alphaOptions = useMemo(
    () => (alphasQ.data ?? []).map((a) => ({ value: a.id, label: a.name })),
    [alphasQ.data],
  )

  return (
    <PageScaffold>
      <Group justify="space-between" align="center" wrap="wrap">
        <Title order={1} m={0}>
          Backtests
        </Title>
        <Button component={Link} to="/backtests/new" color="yellow">
          New backtest
        </Button>
      </Group>

      <Group align="flex-end" gap="md" wrap="wrap">
        <Select
          label="Alpha"
          placeholder="All alphas"
          data={alphaOptions.filter((o) => o.value !== '')}
          value={alphaIdFilter || null}
          clearable
          onChange={(v) => {
            setAlphaIdFilter(v ?? '')
            setOffset(0)
          }}
          disabled={alphasQ.isLoading}
          searchable
          w={260}
        />
        <Select
          label="Status"
          data={STATUS_OPTIONS}
          value={statusFilter === '' ? null : statusFilter}
          clearable
          onChange={(v) => {
            setStatusFilter((v ?? '') as BacktestJobStatus | '')
            setOffset(0)
          }}
          w={160}
        />
        <NumberInput
          label="Limit"
          value={limit}
          onChange={(v) => {
            setLimit(Number(v) || 50)
            setOffset(0)
          }}
          min={1}
          max={200}
          w={88}
        />
        <NumberInput
          label="Offset"
          value={offset}
          onChange={(v) => setOffset(Number(v) || 0)}
          min={0}
          w={88}
        />
        <Button variant="default" disabled={offset === 0} onClick={() => setOffset((o) => Math.max(0, o - limit))}>
          Previous page
        </Button>
        <Button
          variant="default"
          disabled={!q.data || q.data.length < limit}
          onClick={() => setOffset((o) => o + limit)}
        >
          Next page
        </Button>
      </Group>

      {selected.size > 0 && (
        <Group gap="md" wrap="wrap">
          <Text size="sm" c="dimmed">
            {selected.size} selected
          </Text>
          <Button
            color="red"
            variant="light"
            disabled={delMut.isPending}
            onClick={() => {
              const ids = rows.filter((j) => selected.has(j.id)).map((j) => j.id)
              if (ids.length === 0) return
              confirmDeleteJobs(ids)
            }}
          >
            Delete selected
          </Button>
        </Group>
      )}

      <ApiErrorAlert error={q.error} />
      <ApiErrorAlert error={delMut.error} />
      {q.isLoading && (
        <Text c="dimmed" size="sm">
          Loading…
        </Text>
      )}

      {q.data && (
        <>
          <Table.ScrollContainer minWidth={980}>
            <Table {...density} highlightOnHover striped>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th w={40}>
                    <Checkbox
                      aria-label="Select all on this page"
                      checked={allOnPageSelected}
                      indeterminate={someOnPageSelected && !allOnPageSelected}
                      disabled={rows.length === 0 || delMut.isPending}
                      onChange={toggleAllOnPage}
                    />
                  </Table.Th>
                  <Table.Th>Status</Table.Th>
                  <Table.Th>Job ID</Table.Th>
                  <Table.Th>Alpha</Table.Th>
                  <Table.Th>Index</Table.Th>
                  <Table.Th style={{ textAlign: 'right' }}>CAGR</Table.Th>
                  <Table.Th style={{ textAlign: 'right' }}>Sharpe</Table.Th>
                  <Table.Th style={{ textAlign: 'right' }}>Max DD</Table.Th>
                  <Table.Th style={{ textAlign: 'right' }}>End value</Table.Th>
                  <Table.Th>Created</Table.Th>
                  <Table.Th w={100} />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rows.map((j) => {
                  const s = j.status === 'succeeded' && j.result_summary ? j.result_summary : null
                  const cagr = s && typeof s.cagr_pct === 'number' ? formatSummaryNumber(s.cagr_pct, { suffix: '%' }) : '—'
                  const sharpe =
                    s && typeof s.sharpe_ratio === 'number' ? formatSummaryNumber(s.sharpe_ratio, { decimals: 3 }) : '—'
                  const maxDd =
                    s && typeof s.max_drawdown_pct === 'number'
                      ? formatSummaryNumber(s.max_drawdown_pct, { suffix: '%' })
                      : '—'
                  const endVal =
                    s && typeof s.end_value === 'number' ? formatSummaryNumber(s.end_value, { decimals: 0 }) : '—'
                  return (
                  <Table.Tr key={j.id}>
                    <Table.Td>
                      <Checkbox
                        checked={selected.has(j.id)}
                        disabled={delMut.isPending}
                        aria-label={`Select job ${j.id}`}
                        onChange={() => toggleOne(j.id)}
                      />
                    </Table.Td>
                    <Table.Td>{j.status}</Table.Td>
                    <Table.Td>
                      <Anchor component={Link} to={`/backtests/${j.id}`} ff="monospace" size="sm">
                        {j.id}
                      </Anchor>
                    </Table.Td>
                    <Table.Td>
                      {j.alpha_name ?? (
                        <Text ff="monospace" size="sm" c="dimmed">
                          {j.alpha_id}
                        </Text>
                      )}
                    </Table.Td>
                    <Table.Td>
                      <Text ff="monospace" size="sm">
                        {j.index_code ?? '—'}
                      </Text>
                    </Table.Td>
                    <Table.Td style={{ textAlign: 'right' }}>
                      <Text size="sm" ff="monospace">
                        {cagr}
                      </Text>
                    </Table.Td>
                    <Table.Td style={{ textAlign: 'right' }}>
                      <Text size="sm" ff="monospace">
                        {sharpe}
                      </Text>
                    </Table.Td>
                    <Table.Td style={{ textAlign: 'right' }}>
                      <Text size="sm" ff="monospace">
                        {maxDd}
                      </Text>
                    </Table.Td>
                    <Table.Td style={{ textAlign: 'right' }}>
                      <Text size="sm" ff="monospace">
                        {endVal}
                      </Text>
                    </Table.Td>
                    <Table.Td>{new Date(j.created_at).toLocaleString()}</Table.Td>
                    <Table.Td>
                      <Button
                        size="compact-xs"
                        color="red"
                        variant="light"
                        disabled={delMut.isPending}
                        onClick={() => confirmDeleteJobs([j.id])}
                      >
                        Delete
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                  )
                })}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          {q.data.length === 0 && (
            <Text c="dimmed" size="sm">
              No jobs.
            </Text>
          )}
        </>
      )}
    </PageScaffold>
  )
}
