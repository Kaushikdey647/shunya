import { Box, Stack, Text, useMantineColorScheme, useMantineTheme } from '@mantine/core'
import {
  ColorType,
  createChart,
  LineSeries,
  LineType,
  type ISeriesApi,
  type IChartApi,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useEffect, useRef, useState } from 'react'
import { chartColorsFromMantine } from '../../InstrumentChart'
import { buildL1MidSpreadLinePoints, isUsableAlpacaL1Quote } from '../../../live/l1Derived'
import { useLiveL1 } from '../../../live/l1Store'

/** Stepped mid (top) and spread ask−bid (bottom) using `LineType.WithSteps` (lightweight-charts v5). */
export function LiveL1SpreadMidChart() {
  const { state } = useLiveL1()
  const quotes = state.quotes
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()
  const [themeTick, setThemeTick] = useState(0)

  const midPaneRef = useRef<HTMLDivElement>(null)
  const spreadPaneRef = useRef<HTMLDivElement>(null)
  const chartMidRef = useRef<IChartApi | null>(null)
  const chartSpreadRef = useRef<IChartApi | null>(null)
  const seriesMidRef = useRef<ISeriesApi<'Line'> | null>(null)
  const seriesSpreadRef = useRef<ISeriesApi<'Line'> | null>(null)

  useEffect(() => {
    const obs = new MutationObserver(() => setThemeTick((n) => n + 1))
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-mantine-color-scheme'],
    })
    return () => obs.disconnect()
  }, [])

  useEffect(() => {
    const elM = midPaneRef.current
    const elS = spreadPaneRef.current
    if (!elM || !elS) return

    const colors = chartColorsFromMantine(theme, colorScheme)
    const common = {
      layout: {
        background: { type: ColorType.Solid, color: colors.bg },
        textColor: colors.text,
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      timeScale: { borderColor: colors.grid },
      rightPriceScale: { borderColor: colors.grid },
    }

    const chartM = createChart(elM, {
      ...common,
      width: elM.clientWidth,
      height: Math.max(120, elM.clientHeight || 160),
    })
    const chartS = createChart(elS, {
      ...common,
      width: elS.clientWidth,
      height: Math.max(100, elS.clientHeight || 120),
    })

    const sM = chartM.addSeries(LineSeries, {
      color: colors.up,
      lineWidth: 2,
      lineType: LineType.WithSteps,
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
    })
    const sS = chartS.addSeries(LineSeries, {
      color: colors.down,
      lineWidth: 2,
      lineType: LineType.WithSteps,
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
    })

    chartMidRef.current = chartM
    chartSpreadRef.current = chartS
    seriesMidRef.current = sM
    seriesSpreadRef.current = sS

    const roM = new ResizeObserver(() => {
      chartM.applyOptions({
        width: elM.clientWidth,
        height: Math.max(120, elM.clientHeight || 160),
      })
    })
    const roS = new ResizeObserver(() => {
      chartS.applyOptions({
        width: elS.clientWidth,
        height: Math.max(100, elS.clientHeight || 120),
      })
    })
    roM.observe(elM)
    roS.observe(elS)

    return () => {
      roM.disconnect()
      roS.disconnect()
      chartM.remove()
      chartS.remove()
      chartMidRef.current = null
      chartSpreadRef.current = null
      seriesMidRef.current = null
      seriesSpreadRef.current = null
    }
  }, [theme, colorScheme, themeTick])

  useEffect(() => {
    const sM = seriesMidRef.current
    const sS = seriesSpreadRef.current
    const cM = chartMidRef.current
    const cS = chartSpreadRef.current
    if (!sM || !sS || !cM || !cS) return

    const { mid: midData, spread: spreadData } = buildL1MidSpreadLinePoints(quotes)
    const midSeries = midData.map((p) => ({ ...p, time: p.time as UTCTimestamp }))
    const spreadSeries = spreadData.map((p) => ({ ...p, time: p.time as UTCTimestamp }))

    sM.setData(midSeries)
    sS.setData(spreadSeries)
    if (midSeries.length > 0) {
      cM.timeScale().fitContent()
      cS.timeScale().fitContent()
    }
  }, [quotes])

  const last =
    quotes.length > 0
      ? [...quotes].reverse().find((q) => isUsableAlpacaL1Quote(q)) ?? null
      : null
  const lastMid =
    last != null ? ((last.bid_price + last.ask_price) / 2).toFixed(4) : '—'
  const lastSpread =
    last != null ? Math.max(0, last.ask_price - last.bid_price).toFixed(4) : '—'

  return (
    <Stack gap="xs">
      <Text size="sm">
        Last mid <strong>{lastMid}</strong> · spread <strong>{lastSpread}</strong>
      </Text>
      <Text size="xs" c="dimmed">
        Midpoint and spread are piecewise-constant between quote times (<code>LineType.WithSteps</code>
        ).
      </Text>
      <Box ref={midPaneRef} style={{ width: '100%', minWidth: 0, height: 180 }} />
      <Box ref={spreadPaneRef} style={{ width: '100%', minWidth: 0, height: 140 }} />
    </Stack>
  )
}
