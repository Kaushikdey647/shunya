import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { marketClockStreamWebSocketUrl } from '../api/wsUrl'
import type { MarketClockResponse } from '../api/types'

export type MarketClockContextValue = {
  /** Latest payload from the market-clock WebSocket (server source of truth). */
  data: MarketClockResponse | undefined
  isLoading: boolean
  isError: boolean
  /** ``true`` / ``false`` from API; ``null`` while the first tick is pending. */
  alpacaL1UsEquitiesStreamAllowed: boolean | null
}

const MarketClockContext = createContext<MarketClockContextValue | null>(null)

const INITIAL_BACKOFF_MS = 1000
const MAX_BACKOFF_MS = 30_000

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function parseTick(raw: unknown): MarketClockResponse | null {
  if (!isRecord(raw)) return null
  if (raw.type !== 'tick' || raw.schema !== 1) return null
  const utc_iso = typeof raw.utc_iso === 'string' ? raw.utc_iso : null
  const us_line = typeof raw.us_line === 'string' ? raw.us_line : null
  const in_line = typeof raw.in_line === 'string' ? raw.in_line : null
  const us_listed_rth_open = raw.us_listed_rth_open
  const alpaca_l1_us_equities_stream_allowed = raw.alpaca_l1_us_equities_stream_allowed
  if (
    !utc_iso ||
    !us_line ||
    !in_line ||
    typeof us_listed_rth_open !== 'boolean' ||
    typeof alpaca_l1_us_equities_stream_allowed !== 'boolean'
  ) {
    return null
  }
  return {
    utc_iso,
    us_line,
    in_line,
    us_listed_rth_open,
    alpaca_l1_us_equities_stream_allowed,
  }
}

export function MarketClockProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<MarketClockResponse | undefined>(undefined)
  const [isLoading, setIsLoading] = useState(true)
  const [isError, setIsError] = useState(false)

  const url = useMemo(() => marketClockStreamWebSocketUrl(), [])

  useEffect(() => {
    let cancelled = false
    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined
    let backoffMs = INITIAL_BACKOFF_MS

    const clearTimer = () => {
      if (reconnectTimer !== undefined) {
        clearTimeout(reconnectTimer)
        reconnectTimer = undefined
      }
    }

    const connect = () => {
      if (cancelled) return
      clearTimer()
      setIsError(false)
      try {
        ws = new WebSocket(url)
      } catch {
        setIsError(true)
        setIsLoading(false)
        scheduleReconnect()
        return
      }

      ws.onopen = () => {
        backoffMs = INITIAL_BACKOFF_MS
      }

      ws.onmessage = (ev) => {
        let raw: unknown
        try {
          raw = JSON.parse(String(ev.data)) as unknown
        } catch {
          return
        }
        if (!isRecord(raw)) return
        if (raw.type === 'hello') return
        const tick = parseTick(raw)
        if (!tick) return
        setData(tick)
        setIsLoading(false)
        setIsError(false)
      }

      ws.onerror = () => {
        setIsError(true)
        setIsLoading(false)
      }

      ws.onclose = () => {
        ws = null
        if (cancelled) return
        setIsError(true)
        setIsLoading(false)
        scheduleReconnect()
      }
    }

    const scheduleReconnect = () => {
      if (cancelled) return
      clearTimer()
      reconnectTimer = setTimeout(() => {
        reconnectTimer = undefined
        backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS)
        connect()
      }, backoffMs)
    }

    connect()

    return () => {
      cancelled = true
      clearTimer()
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close()
      }
      if (ws && ws.readyState === WebSocket.CONNECTING) {
        ws.close()
      }
      ws = null
    }
  }, [url])

  const value = useMemo((): MarketClockContextValue => {
    let alpacaL1UsEquitiesStreamAllowed: boolean | null
    if (isError) {
      alpacaL1UsEquitiesStreamAllowed = false
    } else {
      const allowed = data?.alpaca_l1_us_equities_stream_allowed
      alpacaL1UsEquitiesStreamAllowed =
        typeof allowed === 'boolean' ? allowed : isLoading ? null : false
    }
    return {
      data,
      isLoading,
      isError,
      alpacaL1UsEquitiesStreamAllowed,
    }
  }, [data, isError, isLoading])

  return <MarketClockContext.Provider value={value}>{children}</MarketClockContext.Provider>
}

export function useMarketClock(): MarketClockContextValue {
  const v = useContext(MarketClockContext)
  if (!v) {
    throw new Error('useMarketClock must be used within MarketClockProvider')
  }
  return v
}
