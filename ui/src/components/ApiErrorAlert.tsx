import { Alert } from '@mantine/core'
import { ApiError } from '../api/client'
import { titleForErrorCode } from '../api/errorCatalog'

export default function ApiErrorAlert({ error }: { error: unknown }) {
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
    <Alert color="red" variant="light" role="alert" title={title}>
      {message}
    </Alert>
  )
}
