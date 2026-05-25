import { Stack, Text } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { useMarketClock } from '../time/MarketClockContext'
import { MEDIA_MIN_MD } from '../theme/breakpoints'

/** Header clocks: US + India wall time from ``/settings/market-clock/stream``. Hidden on narrow viewports. */
export default function MarketClockStrip() {
  const wideEnough = useMediaQuery(MEDIA_MIN_MD)
  const { data, isLoading, isError } = useMarketClock()

  if (!wideEnough) {
    return null
  }

  if (isError) {
    return (
      <Text size="xs" c="red" ff="monospace">
        Clock unavailable
      </Text>
    )
  }

  if (isLoading && !data) {
    return (
      <Text size="xs" c="dimmed" ff="monospace">
        Clock…
      </Text>
    )
  }

  if (!data) {
    return null
  }

  return (
    <Stack gap={0} justify="center" style={{ flexShrink: 0, lineHeight: 1.12 }}>
      <Text size="xs" ff="monospace" lh={1.1} c="dimmed">
        {data.us_line}
      </Text>
      <Text size="xs" ff="monospace" lh={1.1} c="dimmed">
        {data.in_line}
      </Text>
    </Stack>
  )
}
