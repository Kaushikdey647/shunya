import {
  Anchor,
  Button,
  Group,
  Paper,
  Select,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { Link, Navigate, useSearchParams } from 'react-router-dom'
import { ApiError } from '../api/client'
import { createAlpha, listAlphas } from '../api/endpoints'
import { defaultFinStratConfig } from '../api/defaultConfigs'
import type { FinStratConfig } from '../api/types'
import { DEFAULT_ALPHA_BODY } from '../alphaEditor/defaults'
import { wrapAlphaBody } from '../alphaEditor/wrapAlphaBody'
import ApiErrorAlert from '../components/ApiErrorAlert'
import AlphaSourceEditor from '../components/AlphaSourceEditor'
import BacktestConfigPanel from '../components/BacktestConfigPanel'
import FinStratConfigForm from '../components/FinStratConfigForm'
import PageScaffold from '../components/PageScaffold'

export default function BacktestNewPage() {
  const [params] = useSearchParams()
  const alphaFromQuery = params.get('alphaId')?.trim()
  if (alphaFromQuery) {
    return (
      <Navigate
        to={`/studio/${encodeURIComponent(alphaFromQuery)}`}
        replace
      />
    )
  }
  return <BacktestNewPageCore />
}

function BacktestNewPageCore() {
  const qc = useQueryClient()
  const [alphaId, setAlphaId] = useState('')
  const [draftOpen, setDraftOpen] = useState(false)
  const [code, setCode] = useState(DEFAULT_ALPHA_BODY)
  const [draftName, setDraftName] = useState('')
  const [finstratDraft, setFinstratDraft] = useState<FinStratConfig>(defaultFinStratConfig)

  const alphasQ = useQuery({
    queryKey: ['alphas', 'all-for-backtest'],
    queryFn: () => listAlphas({ limit: 500, offset: 0 }),
  })

  const alphaSelectData = useMemo(() => {
    const placeholder =
      alphasQ.isLoading ? 'Loading…' : alphasQ.data?.length ? 'Select…' : 'No alphas'
    return [
      { value: '', label: placeholder },
      ...(alphasQ.data ?? []).map((a) => ({ value: a.id, label: a.name })),
    ]
  }, [alphasQ.isLoading, alphasQ.data])

  const createDraftMut = useMutation({
    mutationFn: async () => {
      const name = draftName.trim()
      const sc = code.trim()
      if (!name) throw new Error('Draft name is required.')
      if (!sc) throw new Error('Add alpha body (function implementation) for the draft.')
      return createAlpha({
        name,
        description: undefined,
        import_ref: null,
        source_code: wrapAlphaBody(code),
        finstrat_config: finstratDraft,
      })
    },
    onSuccess: (row) => {
      void qc.invalidateQueries({ queryKey: ['alphas'] })
      setAlphaId(row.id)
      setDraftOpen(false)
      setDraftName('')
      setCode(DEFAULT_ALPHA_BODY)
    },
  })

  const formId = 'page-backtest-config'

  return (
    <PageScaffold>
      <Group gap="xs">
        <Button component={Link} to="/backtests" variant="default">
          ← Backtests
        </Button>
        <Button component={Link} to="/studio" variant="default">
          Alpha Studio
        </Button>
      </Group>
      <Title order={1}>New backtest</Title>

      <Text size="sm" c="dimmed">
        Prefer the unified{' '}
        <Anchor component={Link} to="/studio">
          Alpha Studio
        </Anchor>{' '}
        (editor + config + results). This page remains for quick runs from the list.
      </Text>

      <Select
        label="Alpha"
        data={alphaSelectData}
        value={alphaId || null}
        onChange={(v) => setAlphaId(v ?? '')}
        searchable
        disabled={alphasQ.isLoading || !alphasQ.data?.length}
        maw={440}
      />

      {!alphasQ.isLoading && !alphasQ.data?.length && (
        <Stack gap="sm">
          <Text c="dimmed" size="sm">
            No alphas yet. Save a draft below or create one from Studio.
          </Text>
          <Button variant="default" onClick={() => setDraftOpen((v) => !v)}>
            {draftOpen ? 'Hide draft creator' : 'Create draft alpha here'}
          </Button>
          {draftOpen && (
            <Paper withBorder p="md" radius="md">
              <Stack gap="md">
                <ApiErrorAlert error={createDraftMut.error} />
                <TextInput
                  label="Draft name (unique)"
                  value={draftName}
                  onChange={(e) => setDraftName(e.target.value)}
                  autoComplete="off"
                  placeholder="my_alpha_1"
                />
                <AlphaSourceEditor value={code} onChange={setCode} height="36vh" />
                <FinStratConfigForm
                  resetKey="draft-bt-new"
                  config={defaultFinStratConfig}
                  onValidChange={setFinstratDraft}
                  isPending={createDraftMut.isPending}
                  submitLabel="Apply strategy to draft"
                  onSubmit={setFinstratDraft}
                />
                <Button
                  type="button"
                  color="yellow"
                  disabled={createDraftMut.isPending}
                  onClick={() => createDraftMut.mutate()}
                >
                  {createDraftMut.isPending ? 'Creating…' : 'Save draft alpha'}
                </Button>
                {createDraftMut.error instanceof ApiError && createDraftMut.error.status === 409 && (
                  <Text size="sm" c="dimmed">
                    Choose a different name (duplicate).
                  </Text>
                )}
              </Stack>
            </Paper>
          )}
        </Stack>
      )}

      {alphaId ? (
        <BacktestConfigPanel
          alphaId={alphaId}
          formId={formId}
          hideInlineSubmit={false}
        />
      ) : (
        <Text c="dimmed" size="sm">
          Select or create an alpha to configure the backtest.
        </Text>
      )}
    </PageScaffold>
  )
}
