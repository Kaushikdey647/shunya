import { Alert, Button, Group, SimpleGrid, Stack, Text, Title } from '@mantine/core'
import { useCallback, useEffect, useState } from 'react'
import type { InstrumentKind } from '../../api/types'
import { LiveL1Provider, useLiveL1 } from '../../live/l1Store'
import { LiveL1BidAskBubbleChart } from './live/LiveL1BidAskBubbleChart'
import { LiveL1OfiHistogram } from './live/LiveL1OfiHistogram'
import { LiveL1SpreadMidChart } from './live/LiveL1SpreadMidChart'
import { LiveL1TradeTape } from './live/LiveL1TradeTape'

type Props = {
  symbol: string
  instrumentKind: InstrumentKind | undefined
  alpacaEnabled: boolean
  /** When false (e.g. user left the Live Data tab), disconnect. */
  enabled: boolean
}

function LivePanelBody({
  instrumentKind,
  kindOk,
  wantStream,
  onConnect,
  onDisconnect,
}: {
  instrumentKind: InstrumentKind | undefined
  kindOk: boolean
  wantStream: boolean
  onConnect: () => void
  onDisconnect: () => void
}) {
  const { state } = useLiveL1()
  const phase = state.phase
  const feedLabel = state.feed
  const channels = state.channels

  return (
    <Stack gap="md">
      <Group justify="space-between" align="center" wrap="wrap">
        <div>
          <Title order={4}>IEX L1 (BBO + trades)</Title>
          <Text size="xs" c="dimmed">
            Unified Alpaca stream: quotes (BBO) and prints over the IEX feed. Channels:{' '}
            {channels?.join(', ') ?? '—'}. Feed: {feedLabel ?? '—'}.
          </Text>
        </div>
        <Group gap="xs">
          {!wantStream || phase === 'error' ? (
            <Button size="compact-sm" color="yellow" onClick={onConnect} disabled={!kindOk}>
              Connect
            </Button>
          ) : (
            <Button size="compact-sm" variant="default" onClick={onDisconnect}>
              Disconnect
            </Button>
          )}
          <Text size="xs" c="dimmed">
            {phase === 'idle' && 'Disconnected'}
            {phase === 'connecting' && 'Connecting…'}
            {phase === 'live' && 'Streaming'}
            {phase === 'error' && 'Error'}
          </Text>
        </Group>
      </Group>

      {instrumentKind === undefined && (
        <Text size="xs" c="dimmed">
          Loading instrument type…
        </Text>
      )}

      {state.lastError && (
        <Alert color="red" variant="light">
          {state.lastError}
        </Alert>
      )}

      <LiveL1SpreadMidChart />

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        <LiveL1BidAskBubbleChart />
        <LiveL1OfiHistogram />
      </SimpleGrid>

      <LiveL1TradeTape />
    </Stack>
  )
}

export default function InstrumentAlpacaLivePanel({
  symbol,
  instrumentKind,
  alpacaEnabled,
  enabled,
}: Props) {
  const [wantStream, setWantStream] = useState(false)
  const kindOk = instrumentKind === 'equity' || instrumentKind === 'etf'
  const streamActive = Boolean(symbol) && alpacaEnabled && enabled && kindOk && wantStream

  const disconnect = useCallback(() => {
    setWantStream(false)
  }, [])

  const connect = useCallback(() => {
    if (!alpacaEnabled || !kindOk) return
    setWantStream(true)
  }, [alpacaEnabled, kindOk])

  useEffect(() => {
    if (!enabled) {
      disconnect()
    }
  }, [enabled, disconnect])

  if (!alpacaEnabled) {
    return (
      <Stack gap="xs">
        <Title order={4}>Alpaca live stream</Title>
        <Text size="sm" c="dimmed">
          Enable Alpaca on the API (<code>SHUNYA_API_ALPACA_ENABLED</code> and APCA keys) for IEX L1 streaming here.
        </Text>
      </Stack>
    )
  }

  if (instrumentKind !== undefined && !kindOk) {
    return (
      <Stack gap="xs">
        <Title order={4}>Alpaca live stream</Title>
        <Text size="sm" c="dimmed">
          Realtime streaming is available for stocks and ETFs only (this symbol is <code>{instrumentKind}</code>).
        </Text>
      </Stack>
    )
  }

  return (
    <LiveL1Provider symbol={symbol} streamActive={streamActive}>
      <LivePanelBody
        instrumentKind={instrumentKind}
        kindOk={kindOk}
        wantStream={wantStream}
        onConnect={connect}
        onDisconnect={disconnect}
      />
    </LiveL1Provider>
  )
}
