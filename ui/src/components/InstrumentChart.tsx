import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts'
import { useMantineColorScheme, useMantineTheme } from '@mantine/core'
import { useEffect, useRef, useState } from 'react'
import type { InstrumentTickerNewsItem, OhlcvBar } from '../api/types'
import { barTimesUtcSeconds, snapNewsToBarTime, timeToUnixSeconds } from '../utils/chartBarTimes'

function chartColorsFromMantine(
  theme: ReturnType<typeof useMantineTheme>,
  colorScheme: string,
) {
  const isDark = colorScheme === 'dark'
  return {
    bg: isDark ? String(theme.other.darkPanelBg) : String(theme.white),
    text: isDark ? theme.colors.dark[2]! : theme.colors.gray[6]!,
    grid: isDark ? theme.colors.dark[5]! : theme.colors.gray[3]!,
    up: theme.colors.teal[6]!,
    down: theme.colors.red[6]!,
    news: theme.colors.yellow[colorScheme === 'dark' ? 5 : 6]!,
  }
}

const EMPTY_NEWS: InstrumentTickerNewsItem[] = []

function markerIdForBarTime(barUnix: number): string {
  return `nw-${barUnix}`
}

function extractMarkerId(param: MouseEventParams<Time>): string | undefined {
  const hi = param.hoveredInfo
  if (hi?.objectId != null && (hi.objectKind === 'series-marker' || hi.type === 'marker')) {
    return String(hi.objectId)
  }
  if (param.hoveredObjectId != null) {
    return String(param.hoveredObjectId)
  }
  return undefined
}

function safeHttpUrl(raw: string | null | undefined): string | undefined {
  if (!raw || !raw.trim()) return undefined
  const u = raw.trim()
  if (/^https?:\/\//i.test(u)) return u
  return undefined
}

function formatNewsDateTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(d)
}

function appendDetailRow(
  parent: HTMLElement,
  label: string,
  value: string,
  className = 'instrument-chart-tooltip-detail',
) {
  const row = document.createElement('div')
  row.className = className
  const lab = document.createElement('span')
  lab.className = 'instrument-chart-tooltip-detail-label'
  lab.textContent = label
  const val = document.createElement('span')
  val.className = 'instrument-chart-tooltip-detail-value'
  val.textContent = value
  row.appendChild(lab)
  row.appendChild(val)
  parent.appendChild(row)
}

function renderNewsTooltip(root: HTMLElement, items: InstrumentTickerNewsItem[]) {
  root.replaceChildren()
  const inner = document.createElement('div')
  inner.className = 'instrument-chart-tooltip-inner'

  items.forEach((item, idx) => {
    if (idx > 0) {
      const sep = document.createElement('hr')
      sep.className = 'instrument-chart-tooltip-sep'
      inner.appendChild(sep)
    }

    const card = document.createElement('article')
    card.className = 'instrument-chart-tooltip-card'

    if (item.published_at) {
      const dt = document.createElement('time')
      dt.className = 'instrument-chart-tooltip-datetime'
      dt.dateTime = item.published_at
      dt.textContent = formatNewsDateTime(item.published_at)
      card.appendChild(dt)
    }

    if (item.publisher?.trim()) {
      const co = document.createElement('div')
      co.className = 'instrument-chart-tooltip-company'
      const lab = document.createElement('span')
      lab.className = 'instrument-chart-tooltip-company-label'
      lab.textContent = 'Source'
      const name = document.createElement('span')
      name.className = 'instrument-chart-tooltip-company-name'
      name.textContent = item.publisher.trim()
      co.appendChild(lab)
      co.appendChild(name)
      card.appendChild(co)
    }

    const title = document.createElement('h3')
    title.className = 'instrument-chart-tooltip-title'
    title.textContent = item.title
    card.appendChild(title)

    const sum = item.summary?.trim()
    if (sum) {
      const p = document.createElement('p')
      p.className = 'instrument-chart-tooltip-summary'
      p.textContent = sum
      card.appendChild(p)
    }

    const desc = item.description?.trim()
    if (desc && desc !== sum) {
      const p = document.createElement('p')
      p.className = 'instrument-chart-tooltip-desc'
      p.textContent = desc
      card.appendChild(p)
    }

    const thumb = safeHttpUrl(item.thumbnail_url ?? undefined)
    if (thumb) {
      const fig = document.createElement('div')
      fig.className = 'instrument-chart-tooltip-thumb-wrap'
      const img = document.createElement('img')
      img.className = 'instrument-chart-tooltip-thumb'
      img.src = thumb
      img.alt = ''
      img.loading = 'lazy'
      img.decoding = 'async'
      fig.appendChild(img)
      card.appendChild(fig)
    }

    const details = document.createElement('div')
    details.className = 'instrument-chart-tooltip-details'
    if (item.content_type) {
      appendDetailRow(details, 'Type', item.content_type)
    }
    if (item.story_id) {
      const sid =
        item.story_id.length > 36 ? `${item.story_id.slice(0, 8)}…` : item.story_id
      appendDetailRow(details, 'Story ID', sid)
    }
    const localeBits = [item.canonical_site, item.canonical_region, item.canonical_lang].filter(
      Boolean,
    ) as string[]
    if (localeBits.length) {
      appendDetailRow(details, 'Edition', localeBits.join(' · '))
    }
    if (item.provider_source_id) {
      appendDetailRow(details, 'Feed', item.provider_source_id)
    }
    if (item.is_hosted === true) {
      appendDetailRow(details, 'Hosted', 'Yes')
    }
    if (details.childNodes.length) {
      card.appendChild(details)
    }

    const badges = document.createElement('div')
    badges.className = 'instrument-chart-tooltip-badges'
    let hasBadge = false
    if (item.editors_pick) {
      const b = document.createElement('span')
      b.className = 'instrument-chart-tooltip-badge'
      b.textContent = "Editor's pick"
      badges.appendChild(b)
      hasBadge = true
    }
    if (item.is_premium_news) {
      const b = document.createElement('span')
      b.className = 'instrument-chart-tooltip-badge'
      b.textContent = 'Premium'
      badges.appendChild(b)
      hasBadge = true
    }
    if (item.is_premium_free_news) {
      const b = document.createElement('span')
      b.className = 'instrument-chart-tooltip-badge'
      b.textContent = 'Premium free'
      badges.appendChild(b)
      hasBadge = true
    }
    if (hasBadge) {
      card.appendChild(badges)
    }

    const articleHref = safeHttpUrl(item.link ?? undefined)
    const providerHref = safeHttpUrl(item.provider_url ?? undefined)
    if (articleHref || providerHref) {
      const actions = document.createElement('div')
      actions.className = 'instrument-chart-tooltip-actions'
      if (articleHref) {
        const a = document.createElement('a')
        a.href = articleHref
        a.target = '_blank'
        a.rel = 'noopener noreferrer'
        a.className = 'instrument-chart-tooltip-link'
        a.textContent = 'Open article'
        actions.appendChild(a)
      }
      if (providerHref && providerHref !== articleHref) {
        const a = document.createElement('a')
        a.href = providerHref
        a.target = '_blank'
        a.rel = 'noopener noreferrer'
        a.className = 'instrument-chart-tooltip-link instrument-chart-tooltip-link-muted'
        a.textContent = 'Publisher site'
        actions.appendChild(a)
      }
      card.appendChild(actions)
    }

    inner.appendChild(card)
  })

  root.appendChild(inner)
}

function renderNewsTooltipCompact(root: HTMLElement, items: InstrumentTickerNewsItem[]) {
  root.replaceChildren()
  const inner = document.createElement('div')
  inner.className = 'instrument-chart-tooltip-inner'
  const p = document.createElement('p')
  p.style.margin = '0'
  p.style.fontSize = '0.75rem'
  p.style.color = 'var(--text-muted)'
  p.textContent =
    items.length === 1 ? items[0]!.title : `${items.length} articles — Context feed`
  inner.appendChild(p)
  root.appendChild(inner)
}

function placeTooltip(
  shell: HTMLElement,
  tooltip: HTMLElement,
  x: number,
  y: number,
) {
  const pad = 10
  tooltip.style.display = 'block'
  const tw = tooltip.offsetWidth
  const th = tooltip.offsetHeight
  let left = x + pad
  let top = y + pad
  if (left + tw > shell.clientWidth - pad) {
    left = Math.max(pad, shell.clientWidth - tw - pad)
  }
  if (top + th > shell.clientHeight - pad) {
    top = Math.max(pad, y - th - pad)
  }
  if (left < pad) left = pad
  if (top < pad) top = pad
  tooltip.style.left = `${left}px`
  tooltip.style.top = `${top}px`
}

function planNewsMarkers(
  bars: OhlcvBar[],
  news: InstrumentTickerNewsItem[],
  newsColor: string,
): {
  markers: SeriesMarker<UTCTimestamp>[]
  tooltipByMarkerId: Map<string, InstrumentTickerNewsItem[]>
} {
  if (bars.length === 0 || news.length === 0) {
    return { markers: [], tooltipByMarkerId: new Map() }
  }
  const barTimes = barTimesUtcSeconds(bars)
  const byBar = new Map<number, InstrumentTickerNewsItem[]>()

  for (const n of news) {
    if (!n.published_at) continue
    const ts = Math.floor(new Date(n.published_at).getTime() / 1000)
    if (!Number.isFinite(ts)) continue
    const barT = snapNewsToBarTime(ts, barTimes)
    const list = byBar.get(barT) ?? []
    list.push(n)
    byBar.set(barT, list)
  }

  const tooltipByMarkerId = new Map<string, InstrumentTickerNewsItem[]>()
  const markers: SeriesMarker<UTCTimestamp>[] = []

  for (const [time, items] of byBar) {
    const id = markerIdForBarTime(time)
    tooltipByMarkerId.set(id, items)
    markers.push({
      time: time as UTCTimestamp,
      position: 'inBar',
      color: newsColor,
      shape: 'circle',
      id,
      size: 1,
    })
  }

  return { markers, tooltipByMarkerId }
}

type Props = {
  bars: OhlcvBar[]
  news?: InstrumentTickerNewsItem[] | undefined
  /** Fires with logical crosshair time (unix seconds) when pointer moves; null when cleared. */
  onCrosshairBarUtc?: (unixSec: number | null) => void
  /** Shrink marker hover tooltip when using side Context feed. */
  compactNewsTooltip?: boolean
}

export default function InstrumentChart({
  bars,
  news,
  onCrosshairBarUtc,
  compactNewsTooltip = false,
}: Props) {
  const theme = useMantineTheme()
  const { colorScheme } = useMantineColorScheme()
  const shellRef = useRef<HTMLDivElement>(null)
  const chartPaneRef = useRef<HTMLDivElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const crosshairCbRef = useRef(onCrosshairBarUtc)
  const compactTipRef = useRef(compactNewsTooltip)
  useEffect(() => {
    crosshairCbRef.current = onCrosshairBarUtc
  }, [onCrosshairBarUtc])
  useEffect(() => {
    compactTipRef.current = compactNewsTooltip
  }, [compactNewsTooltip])
  const [themeTick, setThemeTick] = useState(0)
  const newsList = news ?? EMPTY_NEWS

  useEffect(() => {
    const obs = new MutationObserver(() => setThemeTick((n) => n + 1))
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-mantine-color-scheme'],
    })
    return () => obs.disconnect()
  }, [])

  useEffect(() => {
    const pane = chartPaneRef.current
    const shell = shellRef.current
    const tooltipEl = tooltipRef.current
    if (!pane || !shell || !tooltipEl) return

    const colors = chartColorsFromMantine(theme, colorScheme)
    const chart = createChart(pane, {
      layout: {
        background: { type: ColorType.Solid, color: colors.bg },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.grid },
        horzLines: { color: colors.grid },
      },
      width: pane.clientWidth,
      height: pane.clientHeight,
      timeScale: { borderColor: colors.grid },
      rightPriceScale: { borderColor: colors.grid },
    })

    const series = chart.addSeries(CandlestickSeries, {
      upColor: colors.up,
      downColor: colors.down,
      borderVisible: false,
      wickUpColor: colors.up,
      wickDownColor: colors.down,
    })

    const data = bars.map((b) => ({
      time: Math.floor(new Date(b.time).getTime() / 1000) as UTCTimestamp,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
    series.setData(data)
    if (data.length > 0) {
      chart.timeScale().fitContent()
    }

    const { markers: newsMarkers, tooltipByMarkerId } = planNewsMarkers(
      bars,
      newsList,
      colors.news,
    )
    const seriesMarkers = createSeriesMarkers(series, newsMarkers, {
      autoScale: true,
    })

    const hideTooltip = () => {
      tooltipEl.style.display = 'none'
      tooltipEl.replaceChildren()
    }

    const onCrosshairMove = (param: MouseEventParams<Time>) => {
      const utc = param.point ? timeToUnixSeconds(param.time) : null
      crosshairCbRef.current?.(utc)

      if (tooltipByMarkerId.size === 0) {
        hideTooltip()
        return
      }
      if (!param.point) {
        hideTooltip()
        return
      }
      const markerId = extractMarkerId(param)
      if (!markerId) {
        hideTooltip()
        return
      }
      const items = tooltipByMarkerId.get(markerId)
      if (!items?.length) {
        hideTooltip()
        return
      }
      if (compactTipRef.current) {
        renderNewsTooltipCompact(tooltipEl, items)
      } else {
        renderNewsTooltip(tooltipEl, items)
      }
      requestAnimationFrame(() => {
        placeTooltip(shell, tooltipEl, param.point!.x, param.point!.y)
      })
    }

    chart.subscribeCrosshairMove(onCrosshairMove)

    const onShellLeave = () => {
      hideTooltip()
      crosshairCbRef.current?.(null)
    }
    shell.addEventListener('mouseleave', onShellLeave)

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: pane.clientWidth, height: pane.clientHeight })
    })
    ro.observe(pane)

    return () => {
      shell.removeEventListener('mouseleave', onShellLeave)
      chart.unsubscribeCrosshairMove(onCrosshairMove)
      seriesMarkers.detach()
      ro.disconnect()
      chart.remove()
      hideTooltip()
    }
  }, [bars, newsList, theme, colorScheme, themeTick])

  return (
    <div ref={shellRef} className="instrument-chart-shell">
      <div ref={chartPaneRef} className="instrument-chart-pane" />
      <div
        ref={tooltipRef}
        className="instrument-chart-tooltip"
        aria-hidden="true"
      />
    </div>
  )
}
