import type { CSSProperties } from 'react'
import { useMemo } from 'react'
import { useMantineColorScheme, useMantineTheme } from '@mantine/core'

import { useChartPalette, type ChartAxisTick } from '../../charts/useChartPalette'

export type AxisTick = ChartAxisTick

export type BacktestChartTheme = {
  muted: string
  gridStroke: string
  panelBg: string
  borderColor: string
  textColor: string
  accent: string
  benchStroke: string
  turnoverBar: string
  turnoverBarStroke: string
  turnoverLine: string
  chartAxisStyle: AxisTick
  tooltipStyle: CSSProperties
  ddStroke: string
  ddFill: string
  concBlue: string
  concCyan: string
}

/** Mantine colors + Recharts tick/tooltip styles for backtest result charts. */
export function useBacktestChartTheme(): BacktestChartTheme {
  const base = useChartPalette()
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()

  return useMemo(() => {
    const turnoverBar =
      colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[3]
    const turnoverBarStroke =
      colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[5]
    const turnoverLine =
      colorScheme === 'dark' ? theme.colors.dark[3] : theme.colors.gray[7]
    const ddStroke = theme.colors.red[6]!
    const ddFill =
      colorScheme === 'dark' ? theme.colors.red[9]! : theme.colors.red[1]!
    const benchStroke = theme.colors.cyan[colorScheme === 'dark' ? 5 : 6]!
    const concBlue = theme.colors.blue[6]!
    const concCyan = theme.colors.cyan[6]!

    return {
      ...base,
      benchStroke,
      turnoverBar,
      turnoverBarStroke,
      turnoverLine,
      ddStroke,
      ddFill,
      concBlue,
      concCyan,
    }
  }, [base, colorScheme, theme])
}
