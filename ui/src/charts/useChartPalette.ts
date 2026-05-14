/**
 * Shared chart colors derived from the Mantine theme (semantic roles for Recharts / LWC).
 * Prefer these over ad-hoc hex in new chart code.
 */
import type { CSSProperties } from 'react'
import { useMemo } from 'react'
import { useMantineColorScheme, useMantineTheme } from '@mantine/core'

export type ChartAxisTick = {
  fontSize: number
  fill: string
  fontVariantNumeric: 'tabular-nums'
}

/** Base tokens: grid, axes, tooltips, primary series accent. */
export type ChartPalette = {
  muted: string
  gridStroke: string
  panelBg: string
  borderColor: string
  textColor: string
  accent: string
  chartAxisStyle: ChartAxisTick
  tooltipStyle: CSSProperties
}

export function useChartPalette(): ChartPalette {
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()

  return useMemo(() => {
    const muted =
      colorScheme === 'dark' ? theme.colors.dark[2] : theme.colors.gray[6]
    const gridStroke =
      colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[3]
    const panelBg =
      colorScheme === 'dark' ? theme.other.darkPanelBg : theme.white
    const borderColor =
      colorScheme === 'dark' ? theme.other.darkBorder : theme.colors.gray[4]
    const textColor =
      colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.dark[9]
    const accent =
      theme.colors.yellow[colorScheme === 'dark' ? 5 : 6] ?? theme.colors.yellow[6]
    const chartAxisStyle: ChartAxisTick = {
      fontSize: 11,
      fill: muted,
      fontVariantNumeric: 'tabular-nums',
    }
    const tooltipStyle: CSSProperties = {
      background: panelBg,
      border: `1px solid ${borderColor}`,
      borderRadius: theme.defaultRadius,
      color: textColor,
      fontSize: '0.8125rem',
    }

    return {
      muted,
      gridStroke,
      panelBg,
      borderColor,
      textColor,
      accent,
      chartAxisStyle,
      tooltipStyle,
    }
  }, [colorScheme, theme])
}
