import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { notifications } from '@mantine/notifications'
import { notificationsStreamWebSocketUrl } from '../api/wsUrl'
import { setClientNotificationSink } from './clientSink'
import { NotificationStreamContext } from './notificationStreamContext'
import type { ClientNotificationInput, NotificationLevel, StoredNotification } from './types'

const MAX_ITEMS = 200
const TOAST_MS = 10_000
const INITIAL_BACKOFF_MS = 1000
const MAX_BACKOFF_MS = 30_000

function levelColor(level: NotificationLevel): string {
  switch (level) {
    case 'error':
      return 'red'
    case 'warning':
      return 'yellow'
    case 'info':
    default:
      return 'blue'
  }
}

function showAppToast(item: StoredNotification): void {
  const title =
    item.title ??
    item.code ??
    (item.source === 'http' ? 'Request failed' : item.level === 'error' ? 'Error' : 'Notice')

  notifications.show({
    id: item.id,
    title,
    message: item.message,
    color: levelColor(item.level),
    autoClose: TOAST_MS,
    allowClose: true,
  })
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function parseWsNotification(raw: unknown): StoredNotification | null {
  if (!isRecord(raw)) return null
  if (raw.type !== 'notification') return null
  const id = typeof raw.id === 'string' ? raw.id : null
  const ts = typeof raw.ts === 'string' ? raw.ts : null
  const level = raw.level
  const message = typeof raw.message === 'string' ? raw.message : null
  if (!id || !ts || !message) return null
  if (level !== 'error' && level !== 'warning' && level !== 'info') return null

  const code = typeof raw.code === 'string' ? raw.code : undefined
  const title = typeof raw.title === 'string' ? raw.title : undefined
  let context: Record<string, unknown> | undefined
  if (isRecord(raw.context)) {
    context = { ...raw.context }
  }

  return { id, ts, level, message, code, title, context, source: 'websocket' }
}

export function NotificationStreamProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<StoredNotification[]>([])

  const append = useCallback((item: StoredNotification) => {
    setItems((prev) => [item, ...prev].slice(0, MAX_ITEMS))
    showAppToast(item)
  }, [])

  const onClientSink = useCallback(
    (n: ClientNotificationInput) => {
      append({
        id: crypto.randomUUID(),
        ts: new Date().toISOString(),
        level: n.level,
        message: n.message,
        code: n.code,
        title: n.title,
        context: n.context,
        source: n.source ?? 'http',
      })
    },
    [append],
  )

  useEffect(() => {
    setClientNotificationSink(onClientSink)
    return () => setClientNotificationSink(null)
  }, [onClientSink])

  const url = useMemo(() => notificationsStreamWebSocketUrl(), [])
  const appendRef = useRef(append)

  useEffect(() => {
    appendRef.current = append
  }, [append])

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
      try {
        ws = new WebSocket(url)
      } catch {
        scheduleReconnect()
        return
      }

      ws.onopen = () => {
        backoffMs = INITIAL_BACKOFF_MS
      }

      ws.onmessage = (ev) => {
        let data: unknown
        try {
          data = JSON.parse(String(ev.data)) as unknown
        } catch {
          return
        }
        if (!isRecord(data)) return
        if (data.type === 'hello') return
        const parsed = parseWsNotification(data)
        if (parsed) appendRef.current(parsed)
      }

      ws.onerror = () => {
        /* onclose will handle reconnect */
      }

      ws.onclose = () => {
        ws = null
        if (cancelled) return
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

  const value = useMemo(() => ({ items }), [items])

  return <NotificationStreamContext.Provider value={value}>{children}</NotificationStreamContext.Provider>
}
