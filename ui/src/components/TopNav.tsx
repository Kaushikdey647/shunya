import { ActionIcon, Anchor, Box, Burger, Group, Indicator, Menu, ScrollArea, Text } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import type { RefObject } from 'react'
import { Link } from 'react-router-dom'
import type { TickerSearchHandle } from './TickerSearch'
import DensityToggle from './DensityToggle'
import HealthIndicator from './HealthIndicator'
import MarketClockStrip from './MarketClockStrip'
import ThemeToggle from './ThemeToggle'
import TickerSearch from './TickerSearch'
import { useNotificationStream } from '../notifications/useNotificationStream'
import { MEDIA_BELOW_SM } from '../theme/breakpoints'

function BellIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  )
}

type Props = {
  mobileNavOpened: boolean
  onMobileNavToggle: () => void
  tickerSearchRef: RefObject<TickerSearchHandle | null>
}

export default function TopNav({ mobileNavOpened, onMobileNavToggle, tickerSearchRef }: Props) {
  const isMobile = useMediaQuery(MEDIA_BELOW_SM)
  const { items } = useNotificationStream()

  return (
    <Group
      h="100%"
      px="md"
      justify="space-between"
      wrap="nowrap"
      gap="sm"
      style={{ flex: 1, minWidth: 0 }}
    >
      <Group gap="sm" wrap="nowrap" style={{ flexShrink: 0 }}>
        {isMobile && (
          <Burger
            opened={mobileNavOpened}
            onClick={onMobileNavToggle}
            size="sm"
            aria-label={mobileNavOpened ? 'Close navigation' : 'Open navigation'}
          />
        )}
        <Anchor component={Link} to="/dashboard" fw={700} underline="never" c="var(--mantine-color-text)">
          Shunya
        </Anchor>
        <MarketClockStrip />
      </Group>
      <Box style={{ flex: 1, minWidth: 0, maxWidth: isMobile ? undefined : 448 }}>
        <TickerSearch ref={tickerSearchRef} />
      </Box>
      <Group gap="xs" wrap="nowrap" style={{ flexShrink: 0 }}>
        <Menu shadow="md" width={360} position="bottom-end">
          <Menu.Target>
            <Indicator inline disabled={items.length === 0} label={items.length} size={16}>
              <ActionIcon
                type="button"
                variant="default"
                size="input-sm"
                aria-label={`Notifications, ${items.length} stored this session`}
              >
                <BellIcon />
              </ActionIcon>
            </Indicator>
          </Menu.Target>
          <Menu.Dropdown>
            <Menu.Label>Notifications (this session)</Menu.Label>
            <ScrollArea.Autosize mah={320} type="auto">
              {items.length === 0 ? (
                <Text size="sm" c="dimmed" px="xs" py="sm">
                  No notifications yet.
                </Text>
              ) : (
                items.map((n) => (
                  <Menu.Item key={n.id} closeMenuOnClick={false} style={{ whiteSpace: 'normal', alignItems: 'flex-start' }}>
                    <Box>
                      <Text size="xs" c="dimmed">
                        {n.ts}
                        {n.source ? ` · ${n.source}` : ''}
                        {n.level ? ` · ${n.level}` : ''}
                        {n.code ? ` · ${n.code}` : ''}
                      </Text>
                      {n.title ? (
                        <>
                          <Text size="sm" fw={600}>
                            {n.title}
                          </Text>
                          <Text size="sm">{n.message}</Text>
                        </>
                      ) : (
                        <Text size="sm">{n.message}</Text>
                      )}
                    </Box>
                  </Menu.Item>
                ))
              )}
            </ScrollArea.Autosize>
          </Menu.Dropdown>
        </Menu>
        <HealthIndicator />
        <DensityToggle />
        <ThemeToggle />
      </Group>
    </Group>
  )
}
