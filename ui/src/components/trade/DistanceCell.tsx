import { Box, Group, Text, useComputedColorScheme } from '@mantine/core'

type Props = {
  /** target - current - pending (remaining to work vs PM after children). */
  dist: number
  /** Scale bar to max |distance| in the table (at least 1). */
  cap: number
}

/**
 * Execution lag: green when still short vs target (dist &gt; 0), red when past target (dist &lt; 0).
 * Micro-bar width ∝ |dist| / cap.
 */
export default function DistanceCell({ dist, cap }: Props) {
  const scheme = useComputedColorScheme('light', { getInitialValueInEffect: false })
  const safeCap = Math.max(cap, 1)
  const abs = Math.abs(dist)
  const fill = Math.min(1, abs / safeCap)
  const isDark = scheme === 'dark'

  let fg: string
  let bar: string
  if (dist > 0) {
    fg = isDark ? '#00FF00' : 'var(--mantine-color-teal-7)'
    bar = isDark ? '#00FF00' : 'var(--mantine-color-teal-5)'
  } else if (dist < 0) {
    fg = isDark ? '#FF0000' : 'var(--mantine-color-red-7)'
    bar = isDark ? '#FF0000' : 'var(--mantine-color-red-5)'
  } else {
    fg = 'var(--mantine-color-dimmed)'
    bar = 'var(--mantine-color-dark-3)'
  }

  return (
    <Group gap={8} justify="flex-end" wrap="nowrap" align="center">
      <Box style={{ flex: 1, maxWidth: 72, minWidth: 36 }}>
        <Box
          h={5}
          style={{
            borderRadius: 1,
            background: isDark ? '#222' : 'var(--mantine-color-gray-3)',
            overflow: 'hidden',
            display: 'flex',
            justifyContent: dist < 0 ? 'flex-end' : 'flex-start',
          }}
        >
          <Box h="100%" w={`${fill * 100}%`} style={{ background: bar }} />
        </Box>
      </Box>
      <Text
        ta="right"
        ff="monospace"
        fz="sm"
        fw={600}
        style={{ fontVariantNumeric: 'tabular-nums', color: fg, minWidth: '3.5rem' }}
      >
        {dist > 0 ? `+${dist}` : dist}
      </Text>
    </Group>
  )
}
