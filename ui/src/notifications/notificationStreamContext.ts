import { createContext } from 'react'
import type { StoredNotification } from './types'

export type NotificationStreamContextValue = {
  items: StoredNotification[]
}

export const NotificationStreamContext = createContext<NotificationStreamContextValue | null>(null)
