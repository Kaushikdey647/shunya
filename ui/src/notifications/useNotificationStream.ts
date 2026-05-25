import { useContext } from 'react'
import { NotificationStreamContext, type NotificationStreamContextValue } from './notificationStreamContext'

export function useNotificationStream(): NotificationStreamContextValue {
  const ctx = useContext(NotificationStreamContext)
  if (!ctx) {
    throw new Error('useNotificationStream must be used within NotificationStreamProvider')
  }
  return ctx
}
