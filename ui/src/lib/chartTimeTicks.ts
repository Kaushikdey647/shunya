/** Shared X-axis tick formatter for Recharts when `dataKey` is epoch milliseconds. */
export function tickDate(ms: number): string {
  try {
    return new Date(ms).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: '2-digit',
    })
  } catch {
    return ''
  }
}
