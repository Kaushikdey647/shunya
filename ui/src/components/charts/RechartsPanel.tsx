import { cloneElement, useLayoutEffect, useRef, useState, type ReactElement } from 'react'

const isDev = import.meta.env.DEV

export type RechartsPanelProps = {
  /** Fixed pixel height passed to Recharts (required for stable layout in grid/flex). */
  heightPx: number
  /**
   * Width used until the wrapper reports a real width ≥ 2px (hidden tabs, flex reflow, etc.).
   * Recharts ``ResponsiveContainer`` overwrites its ``initialDimension`` on mount from
   * ``getBoundingClientRect``; a 0px read there leaves charts at zero width, so we measure
   * ourselves and ignore transient zeros.
   */
  initialWidthPx?: number
  /**
   * When provided and zero, skip mounting the chart (saves work; use empty state outside).
   * When omitted, chart mounts whenever heightPx > 0.
   */
  dataLength?: number
  /** Single Recharts root (LineChart, AreaChart, …). */
  children: ReactElement<{ width?: number; height?: number }>
  className?: string
  /** For tests / automation. */
  'data-testid'?: string
}

/**
 * Stable shell for Recharts: measures the wrapper and passes explicit ``width`` / ``height`` to
 * the chart. Avoids ``ResponsiveContainer`` clobbering size with a 0px layout read (tabs,
 * ``min-width: auto`` in grids, first paint).
 */
export function RechartsPanel({
  heightPx,
  initialWidthPx = 640,
  dataLength,
  children,
  className,
  'data-testid': dataTestId,
}: RechartsPanelProps) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [chartWidth, setChartWidth] = useState(initialWidthPx)

  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return

    const apply = () => {
      const w = Math.floor(el.getBoundingClientRect().width)
      if (w >= 2) setChartWidth(w)
    }

    apply()
    const ro = new ResizeObserver(apply)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useLayoutEffect(() => {
    if (!isDev || !wrapRef.current) return
    const el = wrapRef.current
    const id = requestAnimationFrame(() => {
      const r = el.getBoundingClientRect()
      if (r.width < 2 || r.height < 2) {
        console.warn('[RechartsPanel] container has near-zero size', {
          width: r.width,
          height: r.height,
        })
      }
    })
    return () => cancelAnimationFrame(id)
  })

  const skip =
    heightPx <= 0 || (dataLength !== undefined && dataLength === 0)

  const chart =
    !skip && chartWidth >= 2
      ? cloneElement(children, {
          width: chartWidth,
          height: heightPx,
        })
      : null

  return (
    <div
      ref={wrapRef}
      className={className}
      data-testid={dataTestId ?? 'recharts-panel'}
      style={{
        width: '100%',
        minWidth: 0,
        height: heightPx,
        minHeight: heightPx,
      }}
    >
      {chart}
    </div>
  )
}
