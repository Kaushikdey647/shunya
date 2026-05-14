import {
  Alert,
  Button,
  Checkbox,
  Select,
  Stack,
  TextInput,
  Title,
} from '@mantine/core'
import { Controller, useForm } from 'react-hook-form'
import { useEffect, useRef } from 'react'
import { defaultFinStratConfig } from '../api/defaultConfigs'
import type { FinStratConfig } from '../api/types'

export type FinStratFormValues = {
  decay_mode: 'ema' | 'linear'
  decay: string
  decay_window: string
  signal_delay: string
  intraday_session_isolated_lag: boolean
  nan_policy: 'strict' | 'zero_fill'
  temporal_mode: 'bar_step' | 'elapsed_trading_time'
  neutralization: 'none' | 'market' | 'sector' | 'industry'
  truncation: string
  max_single_weight_str: string
  panel_columns_str: string
}

function coalesceNeutralizationForForm(
  n: FinStratConfig['neutralization'] | undefined,
): FinStratFormValues['neutralization'] {
  const raw = n as string | undefined
  if (raw === 'group') return 'sector'
  if (raw === 'none' || raw === 'market' || raw === 'sector' || raw === 'industry') {
    return raw
  }
  return 'market'
}

function toForm(c: FinStratConfig): FinStratFormValues {
  return {
    decay_mode: c.decay_mode ?? 'ema',
    decay: String(c.decay ?? 0),
    decay_window: String(c.decay_window ?? 1),
    signal_delay: String(c.signal_delay ?? 0),
    intraday_session_isolated_lag: c.intraday_session_isolated_lag ?? false,
    nan_policy: c.nan_policy ?? 'strict',
    temporal_mode: c.temporal_mode ?? 'bar_step',
    neutralization: coalesceNeutralizationForForm(c.neutralization),
    truncation: String(c.truncation ?? 0),
    max_single_weight_str:
      c.max_single_weight != null && !Number.isNaN(c.max_single_weight)
        ? String(c.max_single_weight)
        : '',
    panel_columns_str: c.panel_columns?.length ? c.panel_columns.join(', ') : '',
  }
}

function parseFormValues(v: FinStratFormValues):
  | { ok: true; config: FinStratConfig }
  | { ok: false; message: string } {
  const decay = Number(v.decay)
  const decay_window = Number(v.decay_window)
  const signal_delay = Number(v.signal_delay)
  const truncation = Number(v.truncation)
  if (Number.isNaN(decay) || decay < 0 || decay >= 1) {
    return { ok: false, message: 'Decay must be a number in [0, 1).' }
  }
  if (Number.isNaN(decay_window) || !Number.isInteger(decay_window) || decay_window < 1) {
    return { ok: false, message: 'Decay window must be a positive integer.' }
  }
  if (Number.isNaN(signal_delay) || !Number.isInteger(signal_delay) || signal_delay < 0) {
    return { ok: false, message: 'Signal delay must be a non-negative integer.' }
  }
  if (Number.isNaN(truncation) || truncation < 0 || truncation >= 0.5) {
    return { ok: false, message: 'Truncation must be a number in [0, 0.5).' }
  }
  const t = v.max_single_weight_str.trim()
  if (t !== '') {
    const w = Number(t)
    if (Number.isNaN(w) || w <= 0 || w > 1) {
      return { ok: false, message: 'Max single weight: empty or a number in (0, 1].' }
    }
  }
  const max_single_weight = t === '' ? undefined : Number(t)
  const cols = v.panel_columns_str
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean)
  return {
    ok: true,
    config: {
      decay_mode: v.decay_mode,
      decay,
      decay_window,
      signal_delay,
      intraday_session_isolated_lag: v.intraday_session_isolated_lag,
      nan_policy: v.nan_policy,
      temporal_mode: v.temporal_mode,
      neutralization: v.neutralization,
      truncation,
      max_single_weight,
      panel_columns: cols.length ? cols : undefined,
    },
  }
}

type Props = {
  config: FinStratConfig
  resetKey?: string
  onSubmit: (c: FinStratConfig) => void
  onValidChange?: (c: FinStratConfig) => void
  isPending: boolean
  submitLabel: string
  formError?: string | null
}

export default function FinStratConfigForm({
  config,
  resetKey,
  onSubmit,
  onValidChange,
  isPending,
  submitLabel,
  formError,
}: Props) {
  const form = useForm<FinStratFormValues>({
    defaultValues: toForm({ ...defaultFinStratConfig, ...config }),
  })
  const debRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const merged = toForm({ ...defaultFinStratConfig, ...config })
    form.reset(merged)
    if (onValidChange) {
      const p = parseFormValues(merged)
      if (p.ok) onValidChange(p.config)
    }
  }, [resetKey, form, config, onValidChange])

  const decayMode = form.watch('decay_mode')
  const watched = form.watch()

  useEffect(() => {
    if (!onValidChange) return
    if (debRef.current) clearTimeout(debRef.current)
    debRef.current = setTimeout(() => {
      const p = parseFormValues(watched as FinStratFormValues)
      if (p.ok) onValidChange(p.config)
    }, 200)
    return () => {
      if (debRef.current) clearTimeout(debRef.current)
    }
  }, [watched, onValidChange])

  const rootErr = formError ?? form.formState.errors.root?.message

  return (
    <form
      onSubmit={form.handleSubmit((vals) => {
        const p = parseFormValues(vals)
        if (!p.ok) {
          form.setError('root', { type: 'manual', message: p.message })
          return
        }
        onSubmit(p.config)
        form.clearErrors('root')
      })}
    >
      <Stack gap="md">
        <Title order={4}>Strategy config (finstrat)</Title>
        {rootErr && (
          <Alert color="red" variant="light">
            {rootErr}
          </Alert>
        )}
        <Controller
          name="decay_mode"
          control={form.control}
          render={({ field }) => (
            <Select
              label="Decay mode"
              data={[
                { value: 'ema', label: 'ema (EMA coefficient)' },
                { value: 'linear', label: 'linear (fixed window)' },
              ]}
              {...field}
            />
          )}
        />
        {decayMode === 'ema' ? (
          <TextInput
            label="Decay (EMA coefficient in [0, 1))"
            type="number"
            step="any"
            {...form.register('decay')}
          />
        ) : (
          <TextInput
            label="Decay window (bars, integer ≥ 1)"
            type="number"
            step={1}
            {...form.register('decay_window')}
          />
        )}
        <TextInput label="Signal delay" type="number" step={1} {...form.register('signal_delay')} />
        <Controller
          name="intraday_session_isolated_lag"
          control={form.control}
          render={({ field }) => (
            <Checkbox
              label="Intraday session isolated lag"
              checked={field.value}
              onChange={(e) => field.onChange(e.currentTarget.checked)}
            />
          )}
        />
        <Controller
          name="nan_policy"
          control={form.control}
          render={({ field }) => (
            <Select
              label="NaN policy"
              data={[
                { value: 'strict', label: 'strict' },
                { value: 'zero_fill', label: 'zero_fill' },
              ]}
              {...field}
            />
          )}
        />
        <Controller
          name="temporal_mode"
          control={form.control}
          render={({ field }) => (
            <Select
              label="Temporal mode"
              data={[
                { value: 'bar_step', label: 'bar_step' },
                { value: 'elapsed_trading_time', label: 'elapsed_trading_time' },
              ]}
              {...field}
            />
          )}
        />
        <Controller
          name="neutralization"
          control={form.control}
          render={({ field }) => (
            <Select
              label="Neutralization"
              data={[
                { value: 'none', label: 'None' },
                { value: 'market', label: 'Market' },
                { value: 'sector', label: 'Sector' },
                { value: 'industry', label: 'Industry' },
              ]}
              {...field}
            />
          )}
        />
        <TextInput label="Truncation" type="number" step="any" {...form.register('truncation')} />
        <TextInput
          label="Max single weight (0–1, empty = none)"
          {...form.register('max_single_weight_str')}
        />
        <TextInput
          label="Panel columns (comma-separated, optional)"
          {...form.register('panel_columns_str')}
        />
        <Button type="submit" color="yellow" disabled={isPending}>
          {isPending ? 'Saving…' : submitLabel}
        </Button>
      </Stack>
    </form>
  )
}
