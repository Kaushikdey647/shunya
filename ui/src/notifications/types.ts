export type NotificationLevel = 'error' | 'warning' | 'info'

export type NotificationSource = 'websocket' | 'http'

export type StoredNotification = {
  id: string
  ts: string
  level: NotificationLevel
  message: string
  code?: string
  title?: string
  context?: Record<string, unknown>
  source?: NotificationSource
}

export type ClientNotificationInput = {
  level: NotificationLevel
  message: string
  code?: string
  title?: string
  context?: Record<string, unknown>
  source?: NotificationSource
}
