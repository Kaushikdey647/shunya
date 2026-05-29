import { Text } from '@mantine/core'

/** Positive / negative % with teal/red; flat zero or missing uses dimmed. */
export function SignedPctText({
  v,
  digits = 2,
}: {
  v: number | null | undefined
  digits?: number
}) {
  if (v == null || !Number.isFinite(v)) {
    return (
      <Text span c="dimmed" fz="inherit">
        —
      </Text>
    )
  }
  const c = v > 0 ? 'teal' : v < 0 ? 'red' : 'dimmed'
  return (
    <Text span ff="monospace" fz="inherit" c={c} style={{ fontVariantNumeric: 'tabular-nums' }}>
      {v >= 0 ? '+' : ''}
      {v.toFixed(digits)}%
    </Text>
  )
}
