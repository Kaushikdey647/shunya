import {
  Alert,
  Anchor,
  Button,
  Checkbox,
  Code,
  Group,
  List,
  NumberInput,
  Paper,
  ScrollArea,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Tabs,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useForm } from 'react-hook-form'
import { Link, Outlet, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  createAlpha,
  deleteAlpha,
  getAlpha,
  getBacktest,
  getBacktestLogs,
  getBacktestResult,
  listAlphas,
  listBacktests,
  listUniverses,
  patchAlpha,
  postAlphaAssistBacktestReview,
} from '../api/endpoints'
import type { AlphaAssistIssue, BacktestJobOut, FinStratConfig } from '../api/types'
import { ApiError } from '../api/client'
import { DEFAULT_ALPHA_BODY } from '../alphaEditor/defaults'
import { unwrapAlphaSource, wrapAlphaBody } from '../alphaEditor/wrapAlphaBody'
import { defaultFinStratConfig } from '../api/defaultConfigs'
import ApiErrorAlert from '../components/ApiErrorAlert'
import AddToPortfolioModal from '../components/trade/AddToPortfolioModal'
import AlphaSourceEditor from '../components/AlphaSourceEditor'
import BacktestConfigPanel from '../components/BacktestConfigPanel'
import BacktestResultCharts from '../components/BacktestResultCharts'
import FinStratConfigForm from '../components/FinStratConfigForm'
import PageScaffold from '../components/PageScaffold'
import { isInsideAriaModal } from '../keyboard/domGuards'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { useRovingTableKeyboard } from '../keyboard/useRovingList'
import {
  alphaDetailsSchema,
  finstratFromServer,
  type AlphaDetailsFormValues,
} from './alphaStudioForms'
import { z } from 'zod'

const BT_FORM_ID = 'studio-backtest-config-form'

type RailTab = 'details' | 'strategy' | 'config' | 'console'

export default function AlphaStudioLayout() {
  return <Outlet />
}

export function StudioAlphaHub() {
  const density = useMantineTableDensity()
  const navigate = useNavigate()
  const [limit, setLimit] = useState(100)
  const [offset, setOffset] = useState(0)
  const [selected, setSelected] = useState<Set<string>>(() => new Set())
  const qc = useQueryClient()

  const q = useQuery({
    queryKey: ['alphas', limit, offset],
    queryFn: () => listAlphas({ limit, offset }),
  })

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- selection is page-scoped; clear when paging changes
    setSelected(new Set())
  }, [limit, offset])

  const rows = useMemo(() => q.data ?? [], [q.data])

  const pageIds = useMemo(() => rows.map((a) => a.id), [rows])
  const allOnPageSelected =
    pageIds.length > 0 && pageIds.every((id) => selected.has(id))
  const someOnPageSelected = pageIds.some((id) => selected.has(id))

  const delMut = useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) {
        await deleteAlpha(id)
      }
    },
    onSuccess: (_, ids) => {
      void qc.invalidateQueries({ queryKey: ['alphas'] })
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
    const ids = rows.map((a) => a.id)
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

  const onActivateAlphaRow = useCallback(
    (index: number) => {
      const a = rows[index]
      if (a) navigate(`/studio/${encodeURIComponent(a.id)}`)
    },
    [navigate, rows],
  )

  const alphaTableKbd = useRovingTableKeyboard({
    rowCount: rows.length,
    onActivate: onActivateAlphaRow,
  })

  const confirmDeleteAlphas = (ids: string[], label: string) => {
    if (
      !window.confirm(
        `Delete ${ids.length} alpha(s): ${label}? This also removes all backtest jobs for each alpha.`,
      )
    ) {
      return
    }
    delMut.mutate(ids)
  }

  return (
    <PageScaffold>
      <Group justify="space-between" align="flex-start" wrap="wrap">
        <Title order={1}>Alpha Studio</Title>
        <Button color="yellow" onClick={() => navigate('/studio/new')}>
          New alpha
        </Button>
      </Group>
      <Text c="dimmed" size="sm">
        Select an alpha to open the unified workspace (metadata, strategy, source, backtest, console,
        results).
      </Text>

      <Group align="flex-end" wrap="wrap">
        <NumberInput
          label="Limit"
          min={1}
          max={500}
          value={limit}
          onChange={(v) => {
            setLimit(typeof v === 'number' && v > 0 ? v : 100)
            setOffset(0)
          }}
          w={100}
        />
        <NumberInput
          label="Offset"
          min={0}
          value={offset}
          onChange={(v) => setOffset(typeof v === 'number' && v >= 0 ? v : 0)}
          w={100}
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
        <Group align="center" wrap="wrap" gap="sm">
          <Text c="dimmed" size="sm">
            {selected.size} selected
          </Text>
          <Button
            color="red"
            variant="light"
            disabled={delMut.isPending}
            onClick={() => {
              const ids = rows.filter((a) => selected.has(a.id)).map((a) => a.id)
              if (ids.length === 0) return
              const label = rows
                .filter((a) => selected.has(a.id))
                .map((a) => a.name)
                .join(', ')
              confirmDeleteAlphas(ids, label)
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
        <Stack gap="sm">
          <Table.ScrollContainer minWidth={640} {...alphaTableKbd.scrollContainerProps}>
            <Table {...density} striped highlightOnHover>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th w="2.5rem">
                    <Checkbox
                      aria-label="Select all on this page"
                      checked={allOnPageSelected}
                      indeterminate={someOnPageSelected && !allOnPageSelected}
                      disabled={rows.length === 0 || delMut.isPending}
                      onChange={toggleAllOnPage}
                    />
                  </Table.Th>
                  <Table.Th>Name</Table.Th>
                  <Table.Th>ID</Table.Th>
                  <Table.Th>Import ref</Table.Th>
                  <Table.Th>Updated</Table.Th>
                  <Table.Th w="6rem" />
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {rows.map((a, rowIndex) => (
                  <Table.Tr key={a.id} {...alphaTableKbd.rowProps(rowIndex)}>
                    <Table.Td>
                      <Checkbox
                        checked={selected.has(a.id)}
                        disabled={delMut.isPending}
                        aria-label={`Select ${a.name}`}
                        onChange={() => toggleOne(a.id)}
                      />
                    </Table.Td>
                    <Table.Td>
                      <Anchor component={Link} to={`/studio/${encodeURIComponent(a.id)}`}>
                        {a.name}
                      </Anchor>
                    </Table.Td>
                    <Table.Td ff="monospace">{a.id}</Table.Td>
                    <Table.Td ff="monospace">{a.import_ref}</Table.Td>
                    <Table.Td>{new Date(a.updated_at).toLocaleString()}</Table.Td>
                    <Table.Td>
                      <Button
                        color="red"
                        variant="light"
                        size="compact-sm"
                        disabled={delMut.isPending}
                        onClick={() => confirmDeleteAlphas([a.id], a.name)}
                      >
                        Delete
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          {q.data.length === 0 && (
            <Text c="dimmed" size="sm">
              No alphas.
            </Text>
          )}
        </Stack>
      )}
    </PageScaffold>
  )
}

const createAlphaFormSchema = z.object({
  name: z
    .string()
    .min(1, 'Required')
    .max(128)
    .regex(/^[a-zA-Z0-9_-]+$/, 'Alphanumeric, underscore, hyphen only'),
  description: z.string().max(2048).optional(),
})

type CreateAlphaFormValues = z.infer<typeof createAlphaFormSchema>

export function StudioAlphaCreate() {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [code, setCode] = useState(DEFAULT_ALPHA_BODY)
  const [finstratDraft, setFinstratDraft] = useState<FinStratConfig>(defaultFinStratConfig)

  const form = useForm<CreateAlphaFormValues>({
    resolver: zodResolver(createAlphaFormSchema),
    defaultValues: {
      name: '',
      description: '',
    },
  })

  const mutation = useMutation({
    mutationFn: (values: CreateAlphaFormValues) => {
      if (code.trim() === '') {
        return Promise.reject(
          new Error('Add non-empty alpha source code, or use an existing example from the list.'),
        )
      }
      return createAlpha({
        name: values.name,
        description: values.description || undefined,
        import_ref: null,
        source_code: wrapAlphaBody(code),
        finstrat_config: finstratDraft,
      })
    },
    onSuccess: (row) => {
      void qc.invalidateQueries({ queryKey: ['alphas'] })
      navigate(`/studio/${encodeURIComponent(row.id)}`, { replace: true })
    },
  })

  return (
    <PageScaffold>
      <Button component={Link} to="/studio" variant="default">
        ← Studio
      </Button>
      <Title order={1}>New alpha</Title>

      <ApiErrorAlert error={mutation.error} />
      {mutation.error instanceof Error && !('status' in (mutation.error as object)) && (
        <Alert color="red" variant="light">
          {mutation.error.message}
        </Alert>
      )}
      {mutation.error instanceof ApiError && mutation.error.status === 409 && (
        <Text c="dimmed" size="sm">
          Choose a different name (duplicate).
        </Text>
      )}

      <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
        <Stack gap="md">
          <Title order={2} size="h4">
            Alpha source (Python + JAX)
          </Title>
          <Text c="dimmed" size="sm">
            Edit only the <Code>alpha</Code> function body (<Code>ts</Code>, <Code>cs</Code>, <Code>fun</Code>,{' '}
            <Code>ctx.close</Code>, …). Import and signature are added when you save.
          </Text>
          <AlphaSourceEditor value={code} onChange={setCode} height="52vh" />
        </Stack>

        <Stack gap="md">
          <Title order={2} size="h4">
            Details
          </Title>
          <Stack component="form" gap="sm" onSubmit={form.handleSubmit((v) => mutation.mutate(v))}>
            <TextInput label="Name" autoComplete="off" {...form.register('name')} error={form.formState.errors.name?.message} />
            <TextInput label="Description (optional)" {...form.register('description')} />
            <Title order={3} size="h5" mt="sm">
              Strategy config
            </Title>
            <Text c="dimmed" size="xs">
              Adjust below, then create — values are sent with the request.
            </Text>
          </Stack>
          <FinStratConfigForm
            resetKey="new"
            config={defaultFinStratConfig}
            onValidChange={setFinstratDraft}
            isPending={false}
            submitLabel="Apply strategy to draft (optional)"
            onSubmit={setFinstratDraft}
          />
          <Button
            color="yellow"
            disabled={mutation.isPending}
            onClick={() => form.handleSubmit((v) => mutation.mutate(v))()}
          >
            {mutation.isPending ? 'Creating…' : 'Create alpha'}
          </Button>
        </Stack>
      </SimpleGrid>
    </PageScaffold>
  )
}

function syntheticConsoleLines(job: BacktestJobOut | undefined): string[] {
  if (!job) return []
  const lines: string[] = []
  lines.push(`[${job.status}] job ${job.id.slice(0, 8)}…`)
  lines.push(`created ${new Date(job.created_at).toLocaleString()}`)
  if (job.started_at) lines.push(`started ${new Date(job.started_at).toLocaleString()}`)
  if (job.finished_at) lines.push(`finished ${new Date(job.finished_at).toLocaleString()}`)
  if (job.error_message) lines.push(`error:\n${job.error_message}`)
  if (job.result_summary && typeof job.result_summary === 'object') {
    lines.push(`summary: ${JSON.stringify(job.result_summary)}`)
  }
  return lines
}

export function AlphaStudioWorkspace() {
  const { alphaId } = useParams<{ alphaId: string }>()
  if (!alphaId) {
    return (
      <PageScaffold>
        <Text c="dimmed">Missing alpha id.</Text>
        <Button component={Link} to="/studio" variant="default">
          ← Studio
        </Button>
      </PageScaffold>
    )
  }
  return <AlphaStudioWorkspaceInner alphaId={alphaId} />
}

function AlphaStudioWorkspaceInner({ alphaId }: { alphaId: string }) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const [searchParams, setSearchParams] = useSearchParams()

  const [code, setCode] = useState('')
  const [assistIssues, setAssistIssues] = useState<AlphaAssistIssue[]>([])
  const [assistNonce, setAssistNonce] = useState(0)
  const [railTab, setRailTab] = useState<RailTab>('details')
  const [portfolioModalOpen, setPortfolioModalOpen] = useState(false)
  const activeJobId = searchParams.get('job')

  const alphaQ = useQuery({
    queryKey: ['alpha', alphaId],
    queryFn: () => getAlpha(alphaId),
    enabled: Boolean(alphaId),
  })

  const universesPickQ = useQuery({
    queryKey: ['universes', 'alpha-studio'],
    queryFn: () => listUniverses({ limit: 300, offset: 0 }),
    enabled: Boolean(alphaId),
  })

  const detailsForm = useForm<AlphaDetailsFormValues>({
    resolver: zodResolver(alphaDetailsSchema),
    defaultValues: {},
  })

  useEffect(() => {
    if (!alphaQ.data) return
    detailsForm.reset({
      name: alphaQ.data.name,
      description: alphaQ.data.description ?? '',
    })
    // eslint-disable-next-line react-hooks/set-state-in-effect -- controlled editor reset from API
    setCode(
      alphaQ.data.source_code != null && alphaQ.data.source_code.trim() !== ''
        ? unwrapAlphaSource(alphaQ.data.source_code)
        : DEFAULT_ALPHA_BODY,
    )
  }, [alphaQ.data, detailsForm])

  useEffect(() => {
    if (!alphaQ.data) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key !== 'Enter') return
      if (portfolioModalOpen) return
      if (isInsideAriaModal(e.target)) return
      const form = document.getElementById(BT_FORM_ID) as HTMLFormElement | null
      const submitter = document.querySelector(
        `button[type="submit"][form="${BT_FORM_ID}"]`,
      ) as HTMLButtonElement | null
      if (!form || !submitter || submitter.disabled) return
      e.preventDefault()
      e.stopPropagation()
      form.requestSubmit(submitter)
    }
    window.addEventListener('keydown', onKeyDown, { capture: true })
    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [alphaQ.data, portfolioModalOpen])

  const detailsMut = useMutation({
    mutationFn: (body: { name?: string; description?: string | null }) =>
      patchAlpha(alphaId, body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['alpha', alphaId] })
      void qc.invalidateQueries({ queryKey: ['alphas'] })
    },
  })

  const codeMut = useMutation({
    mutationFn: (body: string) =>
      patchAlpha(alphaId, {
        source_code: body.trim() === '' ? null : wrapAlphaBody(body),
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['alpha', alphaId] })
      void qc.invalidateQueries({ queryKey: ['alphas'] })
    },
  })

  const finstratMut = useMutation({
    mutationFn: (finstrat_config: FinStratConfig) => patchAlpha(alphaId, { finstrat_config }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['alpha', alphaId] })
      void qc.invalidateQueries({ queryKey: ['alphas'] })
    },
  })

  const defaultUniverseMut = useMutation({
    mutationFn: (uid: string | null) => patchAlpha(alphaId, { default_universe_id: uid }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['alpha', alphaId] })
      void qc.invalidateQueries({ queryKey: ['alphas'] })
    },
  })

  const delMut = useMutation({
    mutationFn: () => deleteAlpha(alphaId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['alphas'] })
      navigate('/studio')
    },
  })

  const jobQ = useQuery({
    queryKey: ['backtest', activeJobId],
    queryFn: () => getBacktest(activeJobId!),
    enabled: Boolean(activeJobId),
    refetchInterval: (query) => {
      const s = query.state.data?.status
      return s === 'queued' || s === 'running' ? 2000 : false
    },
  })

  const logsQ = useQuery({
    queryKey: ['backtest-logs', activeJobId],
    queryFn: () => getBacktestLogs(activeJobId!),
    enabled: Boolean(activeJobId),
    refetchInterval: () => {
      const st = qc.getQueryData<BacktestJobOut>(['backtest', activeJobId])?.status
      return st === 'queued' || st === 'running' ? 2500 : false
    },
  })

  const lastSuccessQ = useQuery({
    queryKey: ['backtests', 'last-success', alphaId],
    queryFn: () => listBacktests({ alpha_id: alphaId, status: 'succeeded', limit: 1 }),
    enabled: Boolean(alphaId),
  })

  const resultJobId = useMemo(() => {
    if (activeJobId && jobQ.data?.status === 'succeeded') return activeJobId
    const last = lastSuccessQ.data?.[0]?.id
    return last ?? null
  }, [activeJobId, jobQ.data?.status, lastSuccessQ.data])

  const resultJobQ = useQuery({
    queryKey: ['backtest', resultJobId],
    queryFn: () => getBacktest(resultJobId!),
    enabled: Boolean(resultJobId),
  })

  const resultQ = useQuery({
    queryKey: ['backtest-result', resultJobId],
    queryFn: () => getBacktestResult(resultJobId!),
    enabled: Boolean(resultJobId),
  })

  const btReviewMut = useMutation({
    mutationFn: () =>
      postAlphaAssistBacktestReview({
        source_body: code,
        alpha_name: alphaQ.data?.name ?? null,
        alpha_description: alphaQ.data?.description ?? null,
        metrics: (resultQ.data?.metrics as Record<string, unknown>) ?? {},
        result_summary: (resultJobQ.data?.result_summary as Record<string, unknown> | null) ?? null,
      }),
  })

  const onEnqueueSuccess = (job: BacktestJobOut) => {
    setSearchParams({ job: job.id }, { replace: true })
    setRailTab('console')
    void qc.invalidateQueries({ queryKey: ['backtests'] })
    void qc.invalidateQueries({ queryKey: ['backtests', 'last-success', alphaId] })
  }

  const job = jobQ.data
  const resultJob = resultJobQ.data
  const showResultsBanner = Boolean(
    resultJobId && (!activeJobId || (jobQ.data && jobQ.data.status !== 'succeeded')),
  )
  const synth = syntheticConsoleLines(job)
  const remoteLogs = logsQ.data ?? []
  const consoleText =
    remoteLogs.length > 0
      ? [...remoteLogs.map((l) => `[${l.ts}] ${l.message}`), '', '--- poll snapshot ---', ...synth].join(
          '\n',
        )
      : synth.join('\n')

  return (
    <PageScaffold size="xl">
      <Button component={Link} to="/studio" variant="default">
        ← Studio
      </Button>

      <ApiErrorAlert error={alphaQ.error} />
      {alphaQ.isLoading && (
        <Text c="dimmed" size="sm">
          Loading alpha…
        </Text>
      )}
      {!alphaQ.isLoading && !alphaQ.data && (
        <Text c="dimmed" size="sm">
          Alpha not found.
        </Text>
      )}

      {alphaQ.data && (
        <Stack gap="lg">
          <Group justify="space-between" align="flex-start" wrap="wrap">
            <div>
              <Title order={1}>{alphaQ.data.name}</Title>
              <Text ff="monospace" c="dimmed" size="xs" style={{ fontVariantNumeric: 'tabular-nums' }}>
                Studio · {alphaQ.data.id}
              </Text>
            </div>
            <Button variant="light" color="yellow" onClick={() => setPortfolioModalOpen(true)}>
              Add to portfolio
            </Button>
          </Group>
          <AddToPortfolioModal
            opened={portfolioModalOpen}
            onClose={() => setPortfolioModalOpen(false)}
            alphaId={alphaId}
            alphaName={alphaQ.data.name}
          />

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
            <Stack gap="sm">
              <Title order={2} size="h4">
                Alpha source
              </Title>
              <Text c="dimmed" size="xs">
                Body only — saved as a full module with <Code>ts</Code>, <Code>cs</Code>, and <Code>fun</Code>{' '}
                injected automatically.
              </Text>
              <ApiErrorAlert error={codeMut.error} />
              <AlphaSourceEditor
                value={code}
                onChange={setCode}
                height="52vh"
                alphaMeta={{
                  name: alphaQ.data.name,
                  description: alphaQ.data.description,
                }}
                enableAssist
                assistNonce={assistNonce}
                onAssistIssues={setAssistIssues}
              />
              {assistIssues.length > 0 && (
                <Paper withBorder p="sm" radius="sm">
                  <Text size="xs" fw={600} mb="xs">
                    Assist issues
                  </Text>
                  <Stack gap="xs">
                    {assistIssues.map((issue) => (
                      <Group key={issue.id} justify="space-between" align="flex-start" wrap="nowrap">
                        <Text size="sm" style={{ flex: 1 }}>
                          {issue.message}
                        </Text>
                        {issue.corrected_body ? (
                          <Button
                            size="xs"
                            variant="light"
                            color="yellow"
                            onClick={() => {
                              setCode(issue.corrected_body!)
                              setAssistNonce((n) => n + 1)
                            }}
                          >
                            Fix
                          </Button>
                        ) : null}
                      </Group>
                    ))}
                  </Stack>
                </Paper>
              )}
              <Button color="yellow" disabled={codeMut.isPending} onClick={() => codeMut.mutate(code)}>
                {codeMut.isPending ? 'Saving…' : 'Save code'}
              </Button>
            </Stack>

            <Paper withBorder p="md" radius="md">
              <Tabs value={railTab} onChange={(v) => v && setRailTab(v as RailTab)}>
                <Tabs.List grow>
                  <Tabs.Tab value="details">Details</Tabs.Tab>
                  <Tabs.Tab value="strategy">Strategy</Tabs.Tab>
                  <Tabs.Tab value="config">Backtest</Tabs.Tab>
                  <Tabs.Tab value="console">Console</Tabs.Tab>
                </Tabs.List>

                <Tabs.Panel value="details" pt="md">
                  <Stack gap="md">
                    <Title order={3} size="h5">
                      Metadata
                    </Title>
                    <ApiErrorAlert error={detailsMut.error} />
                    <Stack
                      component="form"
                      gap="sm"
                      onSubmit={detailsForm.handleSubmit((v) => {
                        if (!alphaQ.data) return
                        const body: { name?: string; description?: string | null } = {}
                        if (v.name != null && v.name !== alphaQ.data.name) body.name = v.name
                        const desc = v.description === '' ? null : v.description
                        if (desc !== (alphaQ.data.description ?? null)) body.description = desc
                        if (Object.keys(body).length === 0) return
                        detailsMut.mutate(body)
                      })}
                    >
                      <TextInput label="Name" {...detailsForm.register('name')} />
                      <TextInput label="Description" {...detailsForm.register('description')} />
                      <ApiErrorAlert error={defaultUniverseMut.error} />
                      <Select
                        label="Default universe"
                        description="Used for portfolio union views and optional saved-universe backtests."
                        data={[
                          { value: '', label: universesPickQ.isLoading ? 'Loading…' : 'None' },
                          ...(universesPickQ.data ?? []).map((u) => ({
                            value: u.id,
                            label: `${u.name} (${u.member_count})`,
                          })),
                        ]}
                        value={alphaQ.data.default_universe_id ?? ''}
                        onChange={(v) =>
                          defaultUniverseMut.mutate(v && v.trim() ? v.trim() : null)
                        }
                        searchable
                        clearable
                        disabled={defaultUniverseMut.isPending}
                      />
                      {alphaQ.data.import_ref && (
                        <TextInput
                          label="Module import (read-only; overridden when inline source is saved)"
                          readOnly
                          ff="monospace"
                          value={alphaQ.data.import_ref}
                          onChange={() => {}}
                        />
                      )}
                      {detailsForm.formState.errors.name && (
                        <Text size="sm" c="red">
                          {detailsForm.formState.errors.name.message}
                        </Text>
                      )}
                      <Button type="submit" variant="default" disabled={detailsMut.isPending}>
                        {detailsMut.isPending ? 'Saving…' : 'Save metadata'}
                      </Button>
                    </Stack>

                    <Title order={3} size="h5">
                      Delete alpha
                    </Title>
                    <ApiErrorAlert error={delMut.error} />
                    {delMut.error instanceof ApiError && delMut.error.status === 409 && (
                      <Text c="dimmed" size="sm">
                        Cannot delete while backtest jobs reference this alpha.
                      </Text>
                    )}
                    <Button
                      color="red"
                      variant="light"
                      disabled={delMut.isPending}
                      onClick={() => {
                        if (
                          window.confirm(
                            'Delete this alpha? This cannot be undone if the server allows it.',
                          )
                        ) {
                          delMut.mutate()
                        }
                      }}
                    >
                      {delMut.isPending ? 'Deleting…' : 'Delete alpha'}
                    </Button>
                  </Stack>
                </Tabs.Panel>

                <Tabs.Panel value="strategy" pt="md">
                  <ApiErrorAlert error={finstratMut.error} />
                  <FinStratConfigForm
                    config={finstratFromServer(alphaQ.data.finstrat_config)}
                    resetKey={alphaQ.data.updated_at}
                    isPending={finstratMut.isPending}
                    submitLabel="Update strategy config"
                    onSubmit={(c) => finstratMut.mutate(c)}
                  />
                </Tabs.Panel>

                <Tabs.Panel value="config" pt="md">
                  <BacktestConfigPanel
                    alphaId={alphaId}
                    formId={BT_FORM_ID}
                    hideInlineSubmit
                    onEnqueueSuccess={onEnqueueSuccess}
                  />
                </Tabs.Panel>

                <Tabs.Panel value="console" pt="md">
                  <Stack gap="sm">
                    <Text c="dimmed" size="xs">
                      Live worker lines refresh while the job runs; status lines always reflect the latest
                      poll.
                    </Text>
                    <ScrollArea h={360}>
                      <Code block ff="monospace" fz="xs" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        {consoleText}
                      </Code>
                    </ScrollArea>
                    <ApiErrorAlert error={logsQ.error} />
                  </Stack>
                </Tabs.Panel>
              </Tabs>

              <Group mt="md" justify="flex-start" wrap="wrap">
                <Button type="submit" form={BT_FORM_ID} color="yellow">
                  Run backtest
                </Button>
                {activeJobId && (
                  <Button
                    component={Link}
                    variant="default"
                    to={`/backtests/${encodeURIComponent(activeJobId)}`}
                  >
                    Open job page
                  </Button>
                )}
              </Group>
            </Paper>
          </SimpleGrid>

          <Paper withBorder p="md" radius="md">
            <Title order={2} size="h4" mb="md">
              Backtest results
            </Title>
            <Stack gap="xl">
              {showResultsBanner && (
                <Alert variant="light" color="blue" title="Results source">
                  {!activeJobId
                    ? 'Showing metrics and charts from the latest successful backtest for this alpha.'
                    : `Showing metrics and charts from your latest successful run while the selected job is ${jobQ.data?.status ?? 'unknown'}.`}
                </Alert>
              )}
              <ApiErrorAlert error={resultQ.error} />
              {!resultJobId && lastSuccessQ.isLoading && (
                <Text c="dimmed" size="sm">
                  Looking up past runs…
                </Text>
              )}
              {!resultJobId && lastSuccessQ.isSuccess && (
                <Text c="dimmed" size="sm">
                  Run a backtest to see metrics and charts below. Nothing has succeeded yet for this alpha.
                </Text>
              )}
              {resultJobId && !resultQ.data && (
                <Text c="dimmed" size="sm">
                  {resultQ.isLoading ? 'Loading result payload…' : 'Could not load result payload for this job.'}
                </Text>
              )}
              {resultQ.data && (
                <>
                  <BacktestResultCharts data={resultQ.data} show="metrics" metricsStrip />
                  {resultJob?.status === 'succeeded' && (
                    <Paper withBorder p="md" radius="md">
                      <Title order={4} size="h5" mb="xs">
                        AI review (metrics)
                      </Title>
                      <Text c="dimmed" size="xs" mb="md">
                        Uses numeric summary and result summary only (no chart series).
                      </Text>
                      <ApiErrorAlert error={btReviewMut.error} />
                      <Group gap="sm" mb="md">
                        <Button
                          size="xs"
                          variant="light"
                          loading={btReviewMut.isPending}
                          onClick={() => btReviewMut.mutate()}
                        >
                          Run AI review
                        </Button>
                        {btReviewMut.data?.suggested_body ? (
                          <Button
                            size="xs"
                            color="yellow"
                            onClick={() => {
                              setCode(btReviewMut.data.suggested_body!)
                              setAssistNonce((n) => n + 1)
                            }}
                          >
                            Apply suggested body
                          </Button>
                        ) : null}
                      </Group>
                      {(() => {
                        const sp = btReviewMut.data?.summary_points ?? []
                        const rp = btReviewMut.data?.risk_points ?? []
                        const md = btReviewMut.data?.summary_markdown
                        if (sp.length === 0 && rp.length === 0 && md) {
                          return (
                            <ScrollArea h={200}>
                              <Code block ff="monospace" fz="xs" style={{ whiteSpace: 'pre-wrap' }}>
                                {md}
                              </Code>
                            </ScrollArea>
                          )
                        }
                        return (
                          <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                            <Stack gap="xs">
                              <Text fw={600} size="sm">
                                Summary
                              </Text>
                              {sp.length === 0 ? (
                                <Text size="sm" c="dimmed">
                                  Run AI review to populate this list.
                                </Text>
                              ) : (
                                <List size="sm" spacing="xs" listStyleType="disc">
                                  {sp.map((item, i) => (
                                    <List.Item key={`s-${i}`}>{item}</List.Item>
                                  ))}
                                </List>
                              )}
                            </Stack>
                            <Stack gap="xs">
                              <Text fw={600} size="sm">
                                Risks
                              </Text>
                              {rp.length === 0 ? (
                                <Text size="sm" c="dimmed">
                                  Run AI review to populate this list.
                                </Text>
                              ) : (
                                <List size="sm" spacing="xs" listStyleType="disc">
                                  {rp.map((item, i) => (
                                    <List.Item key={`r-${i}`}>{item}</List.Item>
                                  ))}
                                </List>
                              )}
                            </Stack>
                          </SimpleGrid>
                        )
                      })()}
                    </Paper>
                  )}
                  <BacktestResultCharts
                    data={resultQ.data}
                    show="charts"
                    balancedChartColumns
                  />
                </>
              )}
            </Stack>
          </Paper>
        </Stack>
      )}
    </PageScaffold>
  )
}
