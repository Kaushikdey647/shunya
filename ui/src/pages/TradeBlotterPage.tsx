import { Alert, Button, Grid, Group, PasswordInput, Stack, Tabs, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getAppSettings, getOmsPositions } from '../api/endpoints'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import FillsTape from '../components/trade/FillsTape'
import OrderEntryForm from '../components/trade/OrderEntryForm'
import OrdersBlotter from '../components/trade/OrdersBlotter'
import PositionsBlotter from '../components/trade/PositionsBlotter'
import { useTradeDeskToken } from '../hooks/useTradeDeskToken'
import { OmsStreamProvider, useOmsStream } from '../live/omsStore'

function BlotterBody({ tradeDeskToken }: { tradeDeskToken: string }) {
  const { state } = useOmsStream()

  const positionsQ = useQuery({
    queryKey: ['oms', 'positions', tradeDeskToken],
    queryFn: () => getOmsPositions(tradeDeskToken),
    enabled: tradeDeskToken.length > 0,
    refetchInterval: 10_000,
    staleTime: 5_000,
  })

  const paper = state.paper ?? positionsQ.data?.paper ?? null

  return (
    <Stack gap="md">
      <Group gap="xs">
        <Text size="xs" c="dimmed">
          Order stream:
        </Text>
        <Text size="xs" fw={600} c={state.phase === 'live' ? 'teal' : state.phase === 'error' ? 'red' : 'dimmed'}>
          {state.phase}
        </Text>
        {state.lastError && (
          <Text size="xs" c="red">
            {state.lastError}
          </Text>
        )}
      </Group>

      <Grid gap="md" align="stretch">
        <Grid.Col span={{ base: 12, md: 4 }}>
          <OrderEntryForm tradeDeskToken={tradeDeskToken} paper={paper} />
        </Grid.Col>
        <Grid.Col span={{ base: 12, md: 8 }}>
          <FillsTape />
        </Grid.Col>
      </Grid>

      <Tabs defaultValue="orders">
        <Tabs.List>
          <Tabs.Tab value="orders">Orders</Tabs.Tab>
          <Tabs.Tab value="positions">Positions</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="orders" pt="md">
          <OrdersBlotter tradeDeskToken={tradeDeskToken} />
        </Tabs.Panel>
        <Tabs.Panel value="positions" pt="md">
          <PositionsBlotter tradeDeskToken={tradeDeskToken} />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}

export default function TradeBlotterPage() {
  const { draftToken, setDraftToken, appliedToken, applyToken } = useTradeDeskToken()
  const settingsQ = useQuery({ queryKey: ['appSettings'], queryFn: getAppSettings })
  const alpacaEnabled = settingsQ.data?.environment.alpaca_enabled ?? false
  const streamActive = alpacaEnabled && appliedToken.length > 0

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Title order={1}>Blotter</Title>
      <Text size="sm" c="dimmed">
        Manual paper order entry and a live order / position / fill blotter against Alpaca.
      </Text>

      <ApiErrorAlert error={settingsQ.error} />

      {!alpacaEnabled && (
        <Alert color="yellow" title="Alpaca disabled">
          The API reports Alpaca as disabled (<code>SHUNYA_API_ALPACA_ENABLED</code>). Order entry and the
          live blotter are unavailable until it is enabled and keys are configured.
        </Alert>
      )}

      <Group align="flex-end" gap="sm" mt="md" mb="md">
        <PasswordInput
          label="X-Shunya-Trade-Desk-Token"
          description="Same value as SHUNYA_API_TRADE_DESK_TOKEN on the API."
          value={draftToken}
          onChange={(e) => setDraftToken(e.currentTarget.value)}
          disabled={!alpacaEnabled}
          style={{ flex: 1, maxWidth: 420 }}
        />
        <Button
          variant="filled"
          disabled={!alpacaEnabled || draftToken.trim().length === 0}
          onClick={applyToken}
        >
          Apply token
        </Button>
      </Group>

      <OmsStreamProvider token={appliedToken} streamActive={streamActive}>
        <BlotterBody tradeDeskToken={appliedToken} />
      </OmsStreamProvider>
    </PageScaffold>
  )
}
