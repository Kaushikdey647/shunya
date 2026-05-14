import { Anchor, Box, Burger, Group } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { Link } from 'react-router-dom'
import DensityToggle from './DensityToggle'
import HealthIndicator from './HealthIndicator'
import ThemeToggle from './ThemeToggle'
import TickerSearch from './TickerSearch'

type Props = {
  mobileNavOpened: boolean
  onMobileNavToggle: () => void
}

export default function TopNav({ mobileNavOpened, onMobileNavToggle }: Props) {
  const isMobile = useMediaQuery('(max-width: 47.99em)')

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
        <Anchor component={Link} to="/" fw={700} underline="never" c="var(--mantine-color-text)">
          Shunya
        </Anchor>
      </Group>
      <Box style={{ flex: 1, minWidth: 0, maxWidth: isMobile ? undefined : 448 }}>
        <TickerSearch />
      </Box>
      <Group gap="xs" wrap="nowrap" style={{ flexShrink: 0 }}>
        <HealthIndicator />
        <DensityToggle />
        <ThemeToggle />
      </Group>
    </Group>
  )
}
