import { Container, SimpleGrid, Stack, Text, Title } from '@mantine/core'
import MacroStrip from '../components/home/MacroStrip'
import HealthMiniCard from '../components/home/HealthMiniCard'
import MarketHeadlines from '../components/home/MarketHeadlines'
import MoversPanel from '../components/home/MoversPanel'
import RecentBacktestsFeed from '../components/home/RecentBacktestsFeed'
import WatchlistCard from '../components/home/WatchlistCard'

export default function HomePage() {
  return (
    <Container size="xl" px={{ base: 'sm', sm: 'md' }}>
      <Stack gap="lg">
        <Stack gap={4}>
          <Title order={1}>Dashboard</Title>
          <Text c="dimmed" size="sm">
            Macro context, movers, headlines, and your latest backtests.
          </Text>
        </Stack>

        <MacroStrip />

        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
          <Stack gap="lg">
            <MoversPanel />
            <MarketHeadlines />
          </Stack>
          <Stack gap="lg">
            <RecentBacktestsFeed />
            <WatchlistCard />
            <HealthMiniCard />
          </Stack>
        </SimpleGrid>
      </Stack>
    </Container>
  )
}
