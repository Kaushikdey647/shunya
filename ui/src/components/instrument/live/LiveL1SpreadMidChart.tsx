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

/**
 * Top pane: stepped **bid**, **ask**, and **mid** (same price scale).
 * Bottom pane: stepped **spread** (ask − bid).
 */
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
  const seriesBidRef = useRef<ISeriesApi<'Line'> | null>(null)
  const seriesAskRef = useRef<ISeriesApi<'Line'> | null>(null)
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
      height: Math.max(140, elM.clientHeight || 200),
    })
    const chartS = createChart(elS, {
      ...common,
      width: elS.clientWidth,
      height: Math.max(100, elS.clientHeight || 120),
    })

    const sBid = chartM.addSeries(LineSeries, {
      color: colors.up,
      lineWidth: 2,
      lineType: LineType.WithSteps,
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
    })
    const sAsk = chartM.addSeries(LineSeries, {
      color: colors.down,
      lineWidth: 2,
      lineType: LineType.WithSteps,
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 },
    })
    const sMid = chartM.addSeries(LineSeries, {
      color: theme.colors.yellow[colorScheme === 'dark' ? 5 : 6]!,
      lineWidth: 1,
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
    seriesBidRef.current = sBid
    seriesAskRef.current = sAsk
    seriesMidRef.current = sMid
    seriesSpreadRef.current = sS

    const roM = new ResizeObserver(() => {
      chartM.applyOptions({
        width: elM.clientWidth,
        height: Math.max(140, elM.clientHeight || 200),
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
      seriesBidRef.current = null
      seriesAskRef.current = null
      seriesMidRef.current = null
      seriesSpreadRef.current = null
    }
  }, [theme, colorScheme, themeTick])

  useEffect(() => {
    const sBid = seriesBidRef.current
    const sAsk = seriesAskRef.current
    const sMid = seriesMidRef.current
    const sS = seriesSpreadRef.current
    const cM = chartMidRef.current
    const cS = chartSpreadRef.current
    if (!sBid || !sAsk || !sMid || !sS || !cM || !cS) return

    const { mid: midData, spread: spreadData, bid: bidData, ask: askData } = buildL1MidSpreadLinePoints(quotes)
    const toTs = <T extends { time: number; value: number }>(arr: T[]) =>
      arr.map((p) => ({ ...p, time: p.time as UTCTimestamp }))

    sBid.setData(toTs(bidData))
    sAsk.setData(toTs(askData))
    sMid.setData(toTs(midData))
    sS.setData(toTs(spreadData))
    if (midData.length > 0) {
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
      <Text size="sm" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
        Last mid <strong>{lastMid}</strong> · spread <strong>{lastSpread}</strong>
      </Text>
      <Text size="xs" c="dimmed">
        Bid (teal), ask (red), mid (amber), and spread — piecewise-constant between quote times (
        <code>LineType.WithSteps</code>).
      </Text>
      <Box ref={midPaneRef} style={{ width: '100%', minWidth: 0, height: 200 }} />
      <Box ref={spreadPaneRef} style={{ width: '100%', minWidth: 0, height: 140 }} />
    </Stack>
  )
}
