import { Alert, type AlertProps } from '@mantine/core'
import { ApiError } from '../api/client'
import { titleForErrorCode } from '../api/errorCatalog'

type Props = {
  error: unknown
  /** Softer slab in dark mode (e.g. dashboard). */
  variant?: AlertProps['variant']
  /** Tighter padding and smaller message text. */
  compact?: boolean
}

export default function ApiErrorAlert({ error, variant = 'light', compact = false }: Props) {
  if (!error) return null
  const message =
    error instanceof ApiError
      ? error.message
      : error instanceof Error
        ? error.message
        : String(error)
  const code = error instanceof ApiError ? error.code : undefined
  const title = titleForErrorCode(code) ?? (code ? code.replace(/_/g, ' ') : undefined)

  return (
    <Alert
      color="red"
      variant={variant}
      role="alert"
      title={title}
      p={compact ? 'xs' : undefined}
      styles={
        compact
          ? {
              message: { fontSize: 'var(--mantine-font-size-sm)' },
            }
          : undefined
      }
    >
      {message}
    </Alert>
  )
}
