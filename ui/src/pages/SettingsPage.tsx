import {
  Alert,
  Anchor,
  Button,
  Group,
  NumberInput,
  Paper,
  PasswordInput,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { getAppSettings, getTradeAccountConfigurations, patchAppSettings, patchTradeAccountConfigurations } from '../api/endpoints'
import type {
  AlpacaAccountConfigurationsOut,
  AlpacaAccountConfigurationsPatch,
  AppRuntimeTunableKey,
  AppRuntimeTunables,
} from '../api/types'
import { APP_RUNTIME_TUNABLE_KEYS } from '../api/types'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { applyTableDensity, resolveInitialTableDensity, type TableDensity } from '../density'

const FIELD_LABELS: Record<AppRuntimeTunableKey, string> = {
  worker_poll_interval_seconds: 'Worker poll interval (s)',
  max_target_history_points: 'Max target history points',
  max_group_exposure_history_points: 'Max group exposure history points',
  max_exposure_history_points: 'Max exposure history points',
  max_trade_events: 'Max trade events',
  index_ohlcv_backfill_batch_size: 'Index OHLCV backfill batch size',
  market_data_cache_ttl_days: 'Market data cache TTL (days)',
  ollama_timeout_seconds: 'Ollama timeout (s)',
  ollama_model: 'Ollama model id',
}

function buildPatch(baseline: AppRuntimeTunables, current: AppRuntimeTunables): Partial<AppRuntimeTunables> {
  const out: Record<string, string | number> = {}
  for (const k of APP_RUNTIME_TUNABLE_KEYS) {
    if (current[k] !== baseline[k]) {
      out[k] = current[k]
    }
  }
  return out as Partial<AppRuntimeTunables>
}

function cloneRuntime(r: AppRuntimeTunables): AppRuntimeTunables {
  return { ...r }
}

function cloneAlpacaCfg(r: AlpacaAccountConfigurationsOut): AlpacaAccountConfigurationsOut {
  return { ...r }
}

function buildAlpacaPatch(
  baseline: AlpacaAccountConfigurationsOut,
  current: AlpacaAccountConfigurationsOut,
): AlpacaAccountConfigurationsPatch {
  const out: AlpacaAccountConfigurationsPatch = {}
  if (current.dtbp_check !== baseline.dtbp_check) out.dtbp_check = current.dtbp_check
  if (current.pdt_check !== baseline.pdt_check) out.pdt_check = current.pdt_check
  if (current.trade_confirm_email !== baseline.trade_confirm_email)
    out.trade_confirm_email = current.trade_confirm_email
  if (current.max_margin_multiplier !== baseline.max_margin_multiplier)
    out.max_margin_multiplier = current.max_margin_multiplier
  if (current.fractional_trading !== baseline.fractional_trading)
    out.fractional_trading = current.fractional_trading
  if (current.no_shorting !== baseline.no_shorting) out.no_shorting = current.no_shorting
  if (current.suspend_trade !== baseline.suspend_trade) out.suspend_trade = current.suspend_trade
  if (current.ptp_no_exception_entry !== baseline.ptp_no_exception_entry)
    out.ptp_no_exception_entry = current.ptp_no_exception_entry
  if (current.max_options_trading_level !== baseline.max_options_trading_level)
    out.max_options_trading_level = current.max_options_trading_level
  return out
}

export default function SettingsPage() {
  const qc = useQueryClient()
  const q = useQuery({ queryKey: ['appSettings'], queryFn: getAppSettings })
  const [baseline, setBaseline] = useState<AppRuntimeTunables | null>(null)
  const [form, setForm] = useState<AppRuntimeTunables | null>(null)
  const [tradeToken, setTradeToken] = useState('')
  const [alpacaDeskToken, setAlpacaDeskToken] = useState('')
  const [alpacaBaseline, setAlpacaBaseline] = useState<AlpacaAccountConfigurationsOut | null>(null)
  const [alpacaForm, setAlpacaForm] = useState<AlpacaAccountConfigurationsOut | null>(null)
  const [density, setDensity] = useState<TableDensity>(() => resolveInitialTableDensity())

  useEffect(() => {
    if (!q.data) return
    setBaseline(cloneRuntime(q.data.runtime))
    setForm(cloneRuntime(q.data.runtime))
  }, [q.data])

  const mut = useMutation({
    mutationFn: async () => {
      if (!form || !baseline) throw new Error('Settings not loaded')
      const body = buildPatch(baseline, form)
      if (Object.keys(body).length === 0) {
        return getAppSettings()
      }
      return patchAppSettings(body, tradeToken)
    },
    onSuccess: (data) => {
      qc.setQueryData(['appSettings'], data)
      setBaseline(cloneRuntime(data.runtime))
      setForm(cloneRuntime(data.runtime))
    },
  })

  const alpacaLoad = useMutation({
    mutationFn: async () => getTradeAccountConfigurations(alpacaDeskToken.trim()),
    onSuccess: (data) => {
      setAlpacaBaseline(cloneAlpacaCfg(data))
      setAlpacaForm(cloneAlpacaCfg(data))
    },
  })

  const alpacaSave = useMutation({
    mutationFn: async () => {
      if (!alpacaForm || !alpacaBaseline) throw new Error('Load Alpaca configuration first')
      const body = buildAlpacaPatch(alpacaBaseline, alpacaForm)
      if (Object.keys(body).length === 0) {
        return getTradeAccountConfigurations(alpacaDeskToken.trim())
      }
      return patchTradeAccountConfigurations(alpacaDeskToken.trim(), body)
    },
    onSuccess: (data) => {
      setAlpacaBaseline(cloneAlpacaCfg(data))
      setAlpacaForm(cloneAlpacaCfg(data))
    },
  })

  const writeEnabled = q.data?.environment.trade_desk_write_configured ?? false
  const dirty = baseline && form ? Object.keys(buildPatch(baseline, form)).length > 0 : false
  const alpacaEnabled = q.data?.environment.alpaca_enabled ?? false
  const alpacaDirty =
    alpacaBaseline && alpacaForm ? Object.keys(buildAlpacaPatch(alpacaBaseline, alpacaForm)).length > 0 : false

  return (
    <PageScaffold>
      <Title order={2}>Settings</Title>
      <Text size="sm" c="dimmed">
        Server tunables merge <strong>environment</strong> defaults with an optional <strong>database</strong>{' '}
        overlay. Secrets stay in <code>.env</code> / process env only.
      </Text>

      <Paper withBorder p="md" radius="md">
        <Title order={4} mb="xs">
          Keyboard
        </Title>
        <Text size="sm" c="dimmed" mb="xs">
          Global: <strong>⌘K</strong> / <strong>Ctrl+K</strong> command palette · <strong>⇧Space</strong> ticker
          search (avoids macOS Spotlight) · <strong>⇧↑</strong> / <strong>⇧↓</strong> cycle primary nav (skipped in
          modals). Tables: focus the table area, then <strong>⇧↑</strong> / <strong>⇧↓</strong> and{' '}
          <strong>⌘↵</strong> to open the row. Alpha workspace: <strong>⌘↵</strong> runs <strong>Run backtest</strong>{' '}
          when enabled.
        </Text>
        <Text size="xs" c="dimmed">
          Full reference:{' '}
          <Anchor href="https://kaushikdey647.github.io/shunya/ui/keyboard-shortcuts/" target="_blank" rel="noreferrer">
            Keyboard shortcuts (docs)
          </Anchor>
          .
        </Text>
      </Paper>

      <ApiErrorAlert error={q.error} />
      <ApiErrorAlert error={mut.error} />
      <ApiErrorAlert error={alpacaLoad.error} />
      <ApiErrorAlert error={alpacaSave.error} />

      {q.isLoading && <Text size="sm">Loading…</Text>}

      {q.data && (
        <Stack gap="lg">
          <Paper withBorder p="md" radius="md">
            <Title order={4} mb="sm">
              Environment (read-only)
            </Title>
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
              <Text size="sm">
                <strong>Database configured:</strong> {q.data.environment.database_configured ? 'yes' : 'no'}
              </Text>
              <Text size="sm">
                <strong>Alpaca enabled:</strong> {q.data.environment.alpaca_enabled ? 'yes' : 'no'}
              </Text>
              <Text size="sm">
                <strong>Ollama host configured:</strong>{' '}
                {q.data.environment.ollama_host_configured ? 'yes' : 'no'}
              </Text>
              <Text size="sm">
                <strong>Runtime PATCH allowed:</strong>{' '}
                {q.data.environment.trade_desk_write_configured ? 'yes (token required)' : 'no'}
              </Text>
            </SimpleGrid>
            {!q.data.environment.trade_desk_write_configured && (
              <Alert color="yellow" mt="md" title="Writes disabled">
                Set <code>SHUNYA_API_TRADE_DESK_TOKEN</code> on the API and restart. Without it,{' '}
                <code>PATCH /settings/app</code> returns 503.
              </Alert>
            )}
          </Paper>

          <Paper withBorder p="md" radius="md">
            <Title order={4} mb="sm">
              Runtime tunables
            </Title>
            {form && (
              <Stack gap="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
                  {APP_RUNTIME_TUNABLE_KEYS.map((key) => {
                    const src = q.data.sources[key]
                    const suffix = src === 'database' ? ' (from database)' : ' (from environment)'
                    if (key === 'ollama_model') {
                      return (
                        <TextInput
                          key={key}
                          label={`${FIELD_LABELS[key]}${suffix}`}
                          value={form.ollama_model}
                          onChange={(e) =>
                            setForm((prev) =>
                              prev ? { ...prev, ollama_model: e.currentTarget.value } : prev,
                            )
                          }
                        />
                      )
                    }
                    return (
                      <NumberInput
                        key={key}
                        label={`${FIELD_LABELS[key]}${suffix}`}
                        min={key === 'worker_poll_interval_seconds' || key === 'ollama_timeout_seconds' ? 0.01 : 1}
                        step={key === 'worker_poll_interval_seconds' || key === 'ollama_timeout_seconds' ? 0.1 : 1}
                        decimalScale={key === 'worker_poll_interval_seconds' || key === 'ollama_timeout_seconds' ? 2 : 0}
                        value={form[key]}
                        onChange={(v) =>
                          setForm((prev) => {
                            if (!prev) return prev
                            const n = typeof v === 'number' ? v : Number(v)
                            return Number.isFinite(n) ? { ...prev, [key]: n } : prev
                          })
                        }
                      />
                    )
                  })}
                </SimpleGrid>

                <PasswordInput
                  label="X-Shunya-Trade-Desk-Token (for save only)"
                  description="Not stored by this page. Paste the same value as SHUNYA_API_TRADE_DESK_TOKEN on the API."
                  value={tradeToken}
                  onChange={(e) => setTradeToken(e.currentTarget.value)}
                  disabled={!writeEnabled}
                />

                <Group>
                  <Button
                    onClick={() => mut.mutate()}
                    loading={mut.isPending}
                    disabled={!writeEnabled || !form || !baseline || !dirty}
                  >
                    Save runtime changes
                  </Button>
                  <Button
                    variant="default"
                    disabled={!dirty || !baseline}
                    onClick={() => baseline && setForm(cloneRuntime(baseline))}
                  >
                    Reset form
                  </Button>
                </Group>
                {!dirty && writeEnabled && (
                  <Text size="xs" c="dimmed">
                    No pending edits.
                  </Text>
                )}
              </Stack>
            )}
          </Paper>

          <Paper withBorder p="md" radius="md">
            <Title order={4} mb="sm">
              Alpaca
            </Title>
            <Text size="sm" c="dimmed" mb="md">
              Load and update broker account configuration via <code>GET/PATCH /trade/account/configurations</code>.
              Requires <code>SHUNYA_API_TRADE_DESK_TOKEN</code> and Alpaca enabled on the API.
            </Text>
            {!alpacaEnabled && (
              <Alert color="yellow" mb="md" title="Alpaca disabled on API">
                Broker proxy routes return 503 until Alpaca is enabled and keys are configured.
              </Alert>
            )}
            <Stack gap="md">
              <PasswordInput
                label="X-Shunya-Trade-Desk-Token (Alpaca section)"
                description="Used only for Alpaca configuration requests below."
                value={alpacaDeskToken}
                onChange={(e) => setAlpacaDeskToken(e.currentTarget.value)}
                disabled={!alpacaEnabled}
              />
              <Group>
                <Button
                  variant="light"
                  loading={alpacaLoad.isPending}
                  disabled={!alpacaEnabled || !alpacaDeskToken.trim()}
                  onClick={() => alpacaLoad.mutate()}
                >
                  Load configuration
                </Button>
              </Group>
              {alpacaForm && (
                <>
                  <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
                    <Select
                      label="DTBP check"
                      data={[
                        { value: 'both', label: 'both' },
                        { value: 'entry', label: 'entry' },
                        { value: 'exit', label: 'exit' },
                      ]}
                      value={alpacaForm.dtbp_check}
                      onChange={(v) =>
                        v &&
                        setAlpacaForm((prev) => (prev ? { ...prev, dtbp_check: v } : prev))
                      }
                    />
                    <Select
                      label="PDT check"
                      data={[
                        { value: 'both', label: 'both' },
                        { value: 'entry', label: 'entry' },
                        { value: 'exit', label: 'exit' },
                      ]}
                      value={alpacaForm.pdt_check}
                      onChange={(v) =>
                        v &&
                        setAlpacaForm((prev) => (prev ? { ...prev, pdt_check: v } : prev))
                      }
                    />
                    <Select
                      label="Trade confirmation email"
                      data={[
                        { value: 'all', label: 'all' },
                        { value: 'none', label: 'none' },
                      ]}
                      value={alpacaForm.trade_confirm_email}
                      onChange={(v) =>
                        v &&
                        setAlpacaForm((prev) => (prev ? { ...prev, trade_confirm_email: v } : prev))
                      }
                    />
                    <Select
                      label="Max margin multiplier"
                      data={['1', '2', '3', '4'].map((v) => ({ value: v, label: v }))}
                      value={alpacaForm.max_margin_multiplier}
                      onChange={(v) =>
                        v &&
                        setAlpacaForm((prev) => (prev ? { ...prev, max_margin_multiplier: v } : prev))
                      }
                    />
                    <NumberInput
                      label="Max options trading level"
                      description="0–3 per Alpaca; optional."
                      min={0}
                      max={3}
                      value={alpacaForm.max_options_trading_level ?? undefined}
                      onChange={(v) =>
                        setAlpacaForm((prev) => {
                          if (!prev) return prev
                          if (v === '' || v === undefined) {
                            return { ...prev, max_options_trading_level: null }
                          }
                          return { ...prev, max_options_trading_level: Number(v) }
                        })
                      }
                    />
                  </SimpleGrid>
                  <Stack gap="xs">
                    <Switch
                      label="Fractional trading"
                      checked={alpacaForm.fractional_trading}
                      onChange={(e) =>
                        setAlpacaForm((prev) =>
                          prev ? { ...prev, fractional_trading: e.currentTarget.checked } : prev,
                        )
                      }
                    />
                    <Switch
                      label="No shorting"
                      checked={alpacaForm.no_shorting}
                      onChange={(e) =>
                        setAlpacaForm((prev) =>
                          prev ? { ...prev, no_shorting: e.currentTarget.checked } : prev,
                        )
                      }
                    />
                    <Switch
                      label="Suspend trade"
                      checked={alpacaForm.suspend_trade}
                      onChange={(e) =>
                        setAlpacaForm((prev) =>
                          prev ? { ...prev, suspend_trade: e.currentTarget.checked } : prev,
                        )
                      }
                    />
                    <Switch
                      label="PTP no exception entry"
                      checked={alpacaForm.ptp_no_exception_entry}
                      onChange={(e) =>
                        setAlpacaForm((prev) =>
                          prev ? { ...prev, ptp_no_exception_entry: e.currentTarget.checked } : prev,
                        )
                      }
                    />
                  </Stack>
                  <Group>
                    <Button
                      loading={alpacaSave.isPending}
                      disabled={!alpacaDeskToken.trim() || !alpacaDirty}
                      onClick={() => alpacaSave.mutate()}
                    >
                      Save Alpaca configuration
                    </Button>
                    <Button
                      variant="default"
                      disabled={!alpacaDirty || !alpacaBaseline}
                      onClick={() => alpacaBaseline && setAlpacaForm(cloneAlpacaCfg(alpacaBaseline))}
                    >
                      Reset
                    </Button>
                  </Group>
                </>
              )}
            </Stack>
          </Paper>

          <Paper withBorder p="md" radius="md">
            <Text size="sm" c="dimmed" mb="md">
              Stored only in your browser (not sent to the API).
            </Text>
            <Text size="sm" mb="xs">
              Table density
            </Text>
            <SegmentedControl
              value={density}
              onChange={(v) => {
                const next = v as TableDensity
                applyTableDensity(next)
                setDensity(next)
              }}
              data={[
                { label: 'Comfortable', value: 'comfortable' },
                { label: 'Compact', value: 'compact' },
              ]}
            />
          </Paper>

          <Paper withBorder p="md" radius="md">
            <Title order={4} mb="sm">
              Client build-time env (<code>VITE_*</code>)
            </Title>
            <Text size="sm" mb="sm">
              Vite loads <code>.env</code>, <code>.env.local</code>, etc. at dev/build time. Only variables prefixed
              with <code>VITE_</code> are exposed to the bundle (for example <code>VITE_API_BASE</code>). See{' '}
              <code>.env.example</code> in this repo and the README.
            </Text>
            <Table striped highlightOnHover withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Key</Table.Th>
                  <Table.Th>Role</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                <Table.Tr>
                  <Table.Td>
                    <code>VITE_API_BASE</code>
                  </Table.Td>
                  <Table.Td>API origin or <code>/api</code> when same-origin</Table.Td>
                </Table.Tr>
              </Table.Tbody>
            </Table>
          </Paper>
        </Stack>
      )}
    </PageScaffold>
  )
}
