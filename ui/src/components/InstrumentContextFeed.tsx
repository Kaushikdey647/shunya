import { Anchor, Paper, ScrollArea, Stack, Text, Title, useMantineTheme } from '@mantine/core'
import { useEffect, useMemo, useRef } from 'react'
import type { InstrumentTickerNewsItem } from '../api/types'
import { snapNewsToBarTime } from '../utils/chartBarTimes'

function formatNewsWhen(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(d)
}

function snappedBarUtc(item: InstrumentTickerNewsItem, barTimes: number[]): number | null {
  if (!item.published_at || barTimes.length === 0) return null
  const ts = Math.floor(new Date(item.published_at).getTime() / 1000)
  if (!Number.isFinite(ts)) return null
  return snapNewsToBarTime(ts, barTimes)
}

type Props = {
  news: InstrumentTickerNewsItem[]
  barTimes: number[]
  focusBarUnix: number | null
}

export default function InstrumentContextFeed({ news, barTimes, focusBarUnix }: Props) {
  const theme = useMantineTheme()
  const scrollRef = useRef<HTMLDivElement>(null)

  const sorted = useMemo(() => {
    return [...news].sort((a, b) => {
      const ta = a.published_at ? Date.parse(a.published_at) : 0
      const tb = b.published_at ? Date.parse(b.published_at) : 0
      return ta - tb
    })
  }, [news])

  useEffect(() => {
    if (focusBarUnix == null || !scrollRef.current) return
    const el = scrollRef.current.querySelector(`[data-snapped-bar="${focusBarUnix}"]`)
    el?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [focusBarUnix])

  return (
    <Paper
      withBorder
      p="md"
      radius="md"
      component="aside"
      aria-label="Context feed"
      style={{ minHeight: 200 }}
    >
      <Title order={2} size="h5" mb="xs">
        Context
      </Title>
      <Text c="dimmed" size="xs" mb="md">
        News on the price timeline. Move the crosshair on the chart to scroll matching items.
      </Text>
      <ScrollArea h={420} viewportRef={scrollRef} type="auto">
        <Stack gap="sm" pr="xs">
          {sorted.length === 0 ? (
            <Text c="dimmed" size="sm">
              No news for this symbol.
            </Text>
          ) : (
            sorted.map((item, i) => {
              const barU = snappedBarUtc(item, barTimes)
              const focused = barU != null && focusBarUnix != null && barU === focusBarUnix
              const href = item.link?.trim() ? item.link.trim() : undefined
              return (
                <Paper
                  key={`${item.title}-${String(item.story_id ?? i)}`}
                  p="sm"
                  radius="sm"
                  withBorder
                  data-snapped-bar={barU ?? undefined}
                  style={{
                    borderColor: focused ? theme.colors.yellow[5] : undefined,
                    background: focused
                      ? `color-mix(in srgb, ${theme.colors.yellow[4]} 12%, transparent)`
                      : undefined,
                  }}
                >
                  <Stack gap={6}>
                    {item.published_at && (
                      <Text
                        component="time"
                        dateTime={item.published_at}
                        size="xs"
                        c="dimmed"
                        style={{ fontVariantNumeric: 'tabular-nums' }}
                      >
                        {formatNewsWhen(item.published_at)}
                      </Text>
                    )}
                    {href ? (
                      <Anchor href={href} target="_blank" rel="noopener noreferrer" size="sm" fw={600}>
                        {item.title}
                      </Anchor>
                    ) : (
                      <Text size="sm" fw={600}>
                        {item.title}
                      </Text>
                    )}
                    {item.publisher?.trim() && (
                      <Text c="dimmed" size="xs">
                        {item.publisher.trim()}
                      </Text>
                    )}
                    {item.summary?.trim() && (
                      <Text c="dimmed" size="xs" lineClamp={4}>
                        {item.summary.trim()}
                      </Text>
                    )}
                  </Stack>
                </Paper>
              )
            })
          )}
        </Stack>
      </ScrollArea>
    </Paper>
  )
}
