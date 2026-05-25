import type { ClientNotificationInput } from './types'

type Sink = (n: ClientNotificationInput) => void

let sink: Sink | null = null

export function setClientNotificationSink(fn: Sink | null): void {
  sink = fn
}

export function emitClientNotification(n: ClientNotificationInput): void {
  try {
    sink?.(n)
  } catch {
    /* avoid breaking fetch path */
  }
}
