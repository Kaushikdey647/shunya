import { Box, Group, Text, useMantineColorScheme, useMantineTheme } from '@mantine/core'
import { useEffect, useRef } from 'react'
import {
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  LastPriceAnimationMode,
  LineSeries,
  type SeriesMarker,
  type UTCTimestamp,
} from 'lightweight-charts'
import type { PerformanceOverlayRow } from '../../lib/backtestCharts'
import type { BacktestChartTheme } from './useBacktestChartTheme'

function msToUtcTimestamp(ms: number): UTCTimestamp {
  return Math.floor(ms / 1000) as UTCTimestamp
}

type Props = {
  overlay: PerformanceOverlayRow[]
  heightPx: number
  theme: BacktestChartTheme
  benchTicker: string | null
  tradeEvents: Record<string, unknown>[]
}

/**
 * Indexed strategy vs benchmark using lightweight-charts (canvas).
 * More reliable than Recharts for two sparse / uneven line series on one time scale.
 */
export function BacktestPerformanceLcPane({
  overlay,
  heightPx,
  theme,
  benchTicker,
  tradeEvents,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const mantineTheme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()

  useEffect(() => {
    const el = wrapRef.current
    if (!el || overlay.length === 0) return

    const bg = colorScheme === 'dark' ? String(mantineTheme.other.darkPanelBg) : String(mantineTheme.white)
    const grid = theme.gridStroke
    const text = theme.muted

    const upColor = mantineTheme.colors.green[6]!
    const downColor = mantineTheme.colors.red[6]!
    const noBenchColor = theme.accent

    const stratPoints = overlay.map((r) => {
      const time = msToUtcTimestamp(r.t)
      const value = r.equityIdx
      const b = r.benchIdx
      if (b == null || !Number.isFinite(b)) {
        return { time, value, color: noBenchColor }
      }
      const bench = b as number
      return {
        time,
        value,
        color: value >= bench ? upColor : downColor,
      }
    })
    const benchPoints = overlay
      .filter((r) => r.benchIdx != null && Number.isFinite(r.benchIdx))
      .map((r) => ({
        time: msToUtcTimestamp(r.t),
        value: r.benchIdx as number,
      }))

    const tradeMarkers: SeriesMarker<UTCTimestamp>[] = []
    for (const ev of tradeEvents) {
      const rawTs = ev.ts
      if (typeof rawTs !== 'string') continue
      const ms = Date.parse(rawTs)
      if (!Number.isFinite(ms)) continue
      const side = String(ev.side ?? '')
      const isBuy = side === 'buy'
      tradeMarkers.push({
        time: msToUtcTimestamp(ms),
        position: isBuy ? 'belowBar' : 'aboveBar',
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        color: isBuy ? upColor : downColor,
      })
    }

    const w = Math.max(2, el.clientWidth || 640)
    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: bg },
        textColor: text,
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: grid },
        horzLines: { color: grid },
      },
      width: w,
      height: heightPx,
      timeScale: { borderColor: grid },
      rightPriceScale: { borderColor: grid },
      crosshair: {
        mode: CrosshairMode.Magnet,
      },
    })
    const stratSeries = chart.addSeries(LineSeries, {
      color: theme.accent,
      lineWidth: 2,
      lastPriceAnimation: LastPriceAnimationMode.Disabled,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    })
    stratSeries.setData(stratPoints)

    const markersPlugin = createSeriesMarkers(stratSeries, tradeMarkers, { autoScale: true })

    const benchSeries = chart.addSeries(LineSeries, {
      color: theme.benchStroke,
      lineWidth: 2,
      lastPriceAnimation: LastPriceAnimationMode.Disabled,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    })
    if (benchPoints.length > 0) {
      benchSeries.setData(benchPoints)
    }

    chart.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      const cw = wrapRef.current?.clientWidth
      if (cw == null || cw < 2) return
      chart.applyOptions({ width: cw })
    })
    ro.observe(el)

    return () => {
      ro.disconnect()
      markersPlugin.detach()
      chart.remove()
    }
  }, [overlay, heightPx, theme, colorScheme, mantineTheme, tradeEvents])

  const benchLabel = benchTicker ? `Benchmark (${benchTicker})` : 'Benchmark (indexed)'
  const upLegend = mantineTheme.colors.green[6]!
  const downLegend = mantineTheme.colors.red[6]!

  return (
    <Box>
      <Box
        ref={wrapRef}
        data-testid="performance-lc-pane"
        style={{ width: '100%', minWidth: 0, height: heightPx }}
      />
      <Group gap="xl" justify="center" mt="xs" wrap="wrap">
        <Group gap={8} align="center" wrap="nowrap">
          <Group gap={4}>
            <Box style={{ width: 12, height: 3, backgroundColor: upLegend, borderRadius: 1 }} />
            <Box style={{ width: 12, height: 3, backgroundColor: downLegend, borderRadius: 1 }} />
            <Box style={{ width: 12, height: 3, backgroundColor: theme.accent, borderRadius: 1 }} />
          </Group>
          <Text size="xs" c="dimmed">
            Strategy (indexed): green ≥ benchmark, red below, gold no benchmark on that bar
          </Text>
        </Group>
        <Group gap={6}>
          <Box
            style={{
              width: 18,
              height: 3,
              backgroundColor: theme.benchStroke,
              borderRadius: 1,
            }}
          />
          <Text size="xs" c="dimmed">
            {benchLabel}
          </Text>
        </Group>
        {tradeEvents.length > 0 && (
          <Group gap={6}>
            <Box style={{ width: 12, height: 3, backgroundColor: upLegend, borderRadius: 1 }} />
            <Text size="xs" c="dimmed">
              Arrows: buys (up) / sells (down) at execution time
            </Text>
          </Group>
        )}
      </Group>
    </Box>
  )
}
