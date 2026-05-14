import {
  Accordion,
  Button,
  Checkbox,
  Code,
  Select,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BACKTEST_SIM_END_EXCLUSIVE,
  BACKTEST_SIM_START,
  BACKTEST_TEST_START,
} from '../api/backtestWindows'
import { enqueueBacktest, listEquityIndices } from '../api/endpoints'
import { defaultFinBtConfig, defaultFinTsRequest } from '../api/defaultConfigs'
import type { BacktestJobOut, FinBtConfig, FinStratConfig, FinTsRequest } from '../api/types'
import FinTsAdvancedSection from '../components/FinTsAdvancedSection'
import {
  applyFinTsAdvanced,
  initialFinTsAdvancedState,
  type FinTsAdvancedState,
} from '../finTs/advancedState'
import ApiErrorAlert from './ApiErrorAlert'

function initialAdvForIndexBacktest(): FinTsAdvancedState {
  const s = initialFinTsAdvancedState()
  s.marketDataProvider = 'timescale'
  s.attachYf = false
  s.attachFund = false
  return s
}

export type BacktestConfigPanelProps = {
  alphaId: string
  formId: string
  hideInlineSubmit?: boolean
  onEnqueueSuccess?: (job: BacktestJobOut) => void
}

export default function BacktestConfigPanel({
  alphaId,
  formId,
  hideInlineSubmit = false,
  onEnqueueSuccess,
}: BacktestConfigPanelProps) {
  const navigate = useNavigate()
  const qc = useQueryClient()

  const [indexCode, setIndexCode] = useState('')
  const [includeTestInResults, setIncludeTestInResults] = useState(false)
  const [omitMembersMissingOhlcv, setOmitMembersMissingOhlcv] = useState(true)

  const [finTsAdv, setFinTsAdv] = useState<FinTsAdvancedState>(initialAdvForIndexBacktest)

  const [finstratOverrideJson, setFinstratOverrideJson] = useState('')
  const [finbtCash, setFinbtCash] = useState(String(defaultFinBtConfig.cash))
  const [finbtCommission, setFinbtCommission] = useState(
    String(defaultFinBtConfig.commission),
  )
  const [finbtSlippage, setFinbtSlippage] = useState(
    String(defaultFinBtConfig.slippage_pct),
  )
  const [finbtSectorGroupCol, setFinbtSectorGroupCol] = useState(
    defaultFinBtConfig.sector_group_column ?? 'Sector',
  )
  const [finbtValidateFinite, setFinbtValidateFinite] = useState(
    defaultFinBtConfig.validate_finite_targets ?? true,
  )

  const indicesQ = useQuery({
    queryKey: ['equity-indices'],
    queryFn: () => listEquityIndices(),
  })

  const indicesWithMembers = useMemo(
    () => (indicesQ.data ?? []).filter((ix) => ix.member_count > 0),
    [indicesQ.data],
  )

  useEffect(() => {
    if (!indexCode || !indicesWithMembers.length) return
    if (!indicesWithMembers.some((x) => x.code === indexCode)) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- invalidate stale index selection when catalog updates
      setIndexCode('')
    }
  }, [indexCode, indicesWithMembers])

  const selectedBenchmark = useMemo(() => {
    if (!indexCode || !indicesQ.data) return null
    const row = indicesQ.data.find((x) => x.code === indexCode)
    return row?.benchmark_ticker ?? null
  }, [indexCode, indicesQ.data])

  const indexSelectData = useMemo(() => {
    const placeholder =
      indicesQ.isLoading
        ? 'Loading…'
        : indicesWithMembers.length
          ? 'Select…'
          : indicesQ.data?.length
            ? 'No index has members yet (sync memberships)'
            : 'No indices (run DB migrations)'
    return [
      { value: '', label: placeholder },
      ...indicesWithMembers.map((ix) => ({
        value: ix.code,
        label: `${ix.display_name} (${ix.member_count} members)`,
      })),
    ]
  }, [indicesQ.isLoading, indicesQ.data?.length, indicesWithMembers])

  const mutation = useMutation({
    mutationFn: () => {
      const aid = alphaId.trim()
      if (!aid) throw new Error('Missing alpha.')
      if (!indexCode.trim()) throw new Error('Select an index.')

      const fin_ts: FinTsRequest = defaultFinTsRequest({
        start_date: BACKTEST_SIM_START,
        end_date: BACKTEST_SIM_END_EXCLUSIVE,
        ticker_list: [],
      })
      applyFinTsAdvanced(fin_ts, finTsAdv)
      fin_ts.market_data_provider = 'timescale'
      fin_ts.attach_yfinance_classifications = false
      fin_ts.attach_fundamentals = false

      let finstrat_override: FinStratConfig | undefined
      const o = finstratOverrideJson.trim()
      if (o !== '') {
        try {
          finstrat_override = JSON.parse(o) as FinStratConfig
        } catch {
          throw new Error('Invalid finstrat_override JSON.')
        }
      }

      const finbt: FinBtConfig = {
        ...defaultFinBtConfig,
        cash: Number(finbtCash),
        commission: Number(finbtCommission),
        slippage_pct: Number(finbtSlippage),
        sector_group_column: finbtSectorGroupCol,
        validate_finite_targets: finbtValidateFinite,
      }

      return enqueueBacktest({
        alpha_id: aid,
        index_code: indexCode.trim(),
        fin_ts,
        finstrat_override,
        finbt,
        include_test_period_in_results: includeTestInResults,
        omit_index_members_missing_ohlcv: omitMembersMissingOhlcv,
      })
    },
    onSuccess: (job) => {
      void qc.invalidateQueries({ queryKey: ['backtests'] })
      if (onEnqueueSuccess) {
        onEnqueueSuccess(job)
      } else {
        void navigate(`/backtests/${job.id}`)
      }
    },
  })

  return (
    <>
      <ApiErrorAlert error={mutation.error} />

      <form
        id={formId}
        onSubmit={(e) => {
          e.preventDefault()
          mutation.mutate()
        }}
      >
        <Stack gap="md">
          <Select
            label="Index universe"
            data={indexSelectData}
            value={indexCode}
            onChange={(v) => setIndexCode(v ?? '')}
            required
            disabled={indicesQ.isLoading || !indicesWithMembers.length}
          />

          {!indicesQ.isLoading &&
            indicesQ.data &&
            indicesQ.data.length > 0 &&
            indicesWithMembers.length === 0 && (
              <Text size="sm" c="dimmed">
                Timescale has index catalog rows but no{' '}
                <Code>symbol_index_membership</Code> data. From the repo:{' '}
                <Code>shunya-timescale sync-index-memberships</Code> (or run{' '}
                <Code>scripts/bootstrap_sp500_ohlcv.py</Code>
                ), then refresh this page.
              </Text>
            )}

          {selectedBenchmark && (
            <Text size="sm" c="dimmed">
              Benchmark index (raw ticker):{' '}
              <Text span ff="monospace">
                {selectedBenchmark}
              </Text>
            </Text>
          )}

          <Stack gap="xs">
            <Text size="sm" c="dimmed">
              <Text span fw={700}>
                Tune
              </Text>{' '}
              window:{' '}
              <Code>{BACKTEST_SIM_START}</Code> – <Code>{BACKTEST_TEST_START}</Code> (end exclusive).{' '}
              <Text span fw={700}>
                Test
              </Text>{' '}
              window: <Code>{BACKTEST_TEST_START}</Code> –{' '}
              <Code>{BACKTEST_SIM_END_EXCLUSIVE}</Code> (end exclusive).
            </Text>
            <Text size="sm" c="dimmed">
              Simulation always uses <strong>daily</strong> bars over <Code>{BACKTEST_SIM_START}</Code>{' '}
              through the last bar before <Code>{BACKTEST_SIM_END_EXCLUSIVE}</Code> (same range the
              server enforces).
            </Text>
          </Stack>

          <Checkbox
            label={`Include test period (${BACKTEST_TEST_START.slice(0, 4)}–${BACKTEST_SIM_END_EXCLUSIVE.slice(0, 4)}) in results`}
            checked={includeTestInResults}
            onChange={(e) => setIncludeTestInResults(e.currentTarget.checked)}
          />

          <Checkbox
            label="Skip index members with no OHLCV in the simulation window (benchmark ticker must still have data in Timescale)"
            checked={omitMembersMissingOhlcv}
            onChange={(e) => setOmitMembersMissingOhlcv(e.currentTarget.checked)}
          />

          <FinTsAdvancedSection
            state={finTsAdv}
            setState={setFinTsAdv}
            timescaleOnly
            hideBarSpec
          />

          <Accordion variant="separated" radius="md">
            <Accordion.Item value="adv-finbt">
              <Accordion.Control>Advanced (overrides, finbt)</Accordion.Control>
              <Accordion.Panel>
                <Stack gap="md" mt="xs">
                  <Textarea
                    label="Finstrat override (JSON, optional)"
                    value={finstratOverrideJson}
                    onChange={(e) => setFinstratOverrideJson(e.target.value)}
                    placeholder="{}"
                    minRows={3}
                    ff="monospace"
                    fz="sm"
                  />
                  <TextInput
                    label="finbt.cash"
                    type="number"
                    value={finbtCash}
                    onChange={(e) => setFinbtCash(e.target.value)}
                  />
                  <TextInput
                    label="finbt.commission"
                    type="number"
                    step="any"
                    value={finbtCommission}
                    onChange={(e) => setFinbtCommission(e.target.value)}
                  />
                  <TextInput
                    label="finbt.slippage_pct"
                    type="number"
                    step="any"
                    value={finbtSlippage}
                    onChange={(e) => setFinbtSlippage(e.target.value)}
                  />
                  <TextInput
                    label="finbt.sector_group_column"
                    value={finbtSectorGroupCol}
                    onChange={(e) => setFinbtSectorGroupCol(e.target.value)}
                  />
                  <Checkbox
                    label="finbt.validate_finite_targets"
                    checked={finbtValidateFinite}
                    onChange={(e) => setFinbtValidateFinite(e.currentTarget.checked)}
                  />
                </Stack>
              </Accordion.Panel>
            </Accordion.Item>
          </Accordion>

          {!hideInlineSubmit && (
            <Button type="submit" color="yellow" disabled={mutation.isPending}>
              {mutation.isPending ? 'Enqueueing…' : 'Run backtest'}
            </Button>
          )}
        </Stack>
      </form>
    </>
  )
}
