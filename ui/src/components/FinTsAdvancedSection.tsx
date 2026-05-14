import {
  Accordion,
  Checkbox,
  NumberInput,
  Select,
  Stack,
  Text,
  TextInput,
} from '@mantine/core'
import type { Dispatch, SetStateAction } from 'react'
import type { BarUnit, FeatureMode, MarketDataProvider, TradingAxisMode } from '../api/types'
import type { FinTsAdvancedState } from '../finTs/advancedState'

const BAR_UNITS: BarUnit[] = [
  'SECONDS',
  'MINUTES',
  'HOURS',
  'DAYS',
  'WEEKS',
  'MONTHS',
  'YEARS',
]

type Props = {
  state: FinTsAdvancedState
  setState: Dispatch<SetStateAction<FinTsAdvancedState>>
  timescaleOnly?: boolean
  hideBarSpec?: boolean
}

export default function FinTsAdvancedSection({
  state,
  setState,
  timescaleOnly = false,
  hideBarSpec = false,
}: Props) {
  const a = state
  const set = setState

  return (
    <Accordion variant="separated" radius="md">
      <Accordion.Item value="fin-ts-adv">
        <Accordion.Control>Advanced (fin_ts)</Accordion.Control>
        <Accordion.Panel>
          <Stack gap="md" mt="xs">
            {!hideBarSpec && (
              <>
                <Checkbox
                  label="Set bar_spec"
                  checked={a.useBarSpec}
                  onChange={(e) => set((s) => ({ ...s, useBarSpec: e.currentTarget.checked }))}
                />
                {a.useBarSpec && (
                  <Stack gap="md">
                    <Select
                      label="Bar unit"
                      data={BAR_UNITS.map((u) => ({ value: u, label: u }))}
                      value={a.barUnit}
                      onChange={(v) =>
                        v && set((s) => ({ ...s, barUnit: v as BarUnit }))
                      }
                    />
                    <NumberInput
                      label="Bar step"
                      min={1}
                      value={a.barStep}
                      onChange={(v) =>
                        set((s) => ({
                          ...s,
                          barStep: typeof v === 'number' ? v : 1,
                        }))
                      }
                    />
                  </Stack>
                )}
              </>
            )}
            {hideBarSpec && (
              <Text size="sm" c="dimmed">
                Bar cadence is fixed to <Text span ff="monospace">DAYS</Text> step{' '}
                <Text span ff="monospace">
                  1
                </Text>{' '}
                for backtests.
              </Text>
            )}
            {timescaleOnly ? (
              <Text size="sm" c="dimmed">
                Market data: Timescale only (index universe). Classifications load from DB when
                available.
              </Text>
            ) : (
              <Select
                label="Market data provider"
                data={[
                  { value: 'auto', label: 'auto' },
                  { value: 'timescale', label: 'timescale' },
                  { value: 'yfinance', label: 'yfinance' },
                ]}
                value={a.marketDataProvider}
                onChange={(v) =>
                  v &&
                  set((s) => ({
                    ...s,
                    marketDataProvider: v as MarketDataProvider,
                  }))
                }
              />
            )}
            <Select
              label="Feature mode"
              data={[
                { value: 'full', label: 'full' },
                { value: 'ohlcv_only', label: 'ohlcv_only' },
              ]}
              value={a.featureMode}
              onChange={(v) =>
                v &&
                set((s) => ({
                  ...s,
                  featureMode: v as FeatureMode,
                }))
              }
            />
            <Select
              label="Trading axis mode"
              data={[
                { value: 'observed', label: 'observed' },
                { value: 'canonical', label: 'canonical' },
              ]}
              value={a.tradingAxisMode}
              onChange={(v) =>
                v &&
                set((s) => ({
                  ...s,
                  tradingAxisMode: v as TradingAxisMode,
                }))
              }
            />
            {!timescaleOnly && (
              <Checkbox
                label="attach_yfinance_classifications"
                checked={a.attachYf}
                onChange={(e) => set((s) => ({ ...s, attachYf: e.currentTarget.checked }))}
              />
            )}
            {!timescaleOnly && (
              <Checkbox
                label="attach_fundamentals"
                checked={a.attachFund}
                onChange={(e) => set((s) => ({ ...s, attachFund: e.currentTarget.checked }))}
              />
            )}
            <Checkbox
              label="strict_trading_grid"
              checked={a.strictTradingGrid}
              onChange={(e) => set((s) => ({ ...s, strictTradingGrid: e.currentTarget.checked }))}
            />
            <Checkbox
              label="strict_provider_universe"
              checked={a.strictProviderUniverse}
              onChange={(e) =>
                set((s) => ({ ...s, strictProviderUniverse: e.currentTarget.checked }))
              }
            />
            <Checkbox
              label="strict_ohlcv"
              checked={a.strictOhlcv}
              onChange={(e) => set((s) => ({ ...s, strictOhlcv: e.currentTarget.checked }))}
            />
            <Checkbox
              label="strict_empty"
              checked={a.strictEmpty}
              onChange={(e) => set((s) => ({ ...s, strictEmpty: e.currentTarget.checked }))}
            />
            <TextInput
              label="require_history_bars (optional)"
              type="number"
              min={1}
              value={a.requireHistoryBars}
              onChange={(e) => set((s) => ({ ...s, requireHistoryBars: e.target.value }))}
              placeholder="empty = omit"
            />
          </Stack>
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  )
}
