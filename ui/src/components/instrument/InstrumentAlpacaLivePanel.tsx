import { Alert, Button, Grid, Group, Stack, Text, Title } from '@mantine/core'
import { useCallback, useEffect, useState } from 'react'
import type { InstrumentKind } from '../../api/types'
import { LiveL1Provider, useLiveL1 } from '../../live/l1Store'
import { useMarketClock } from '../../time/MarketClockContext'
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
  usL1StreamGateError,
}: {
  instrumentKind: InstrumentKind | undefined
  kindOk: boolean
  wantStream: boolean
  onConnect: () => void
  onDisconnect: () => void
  usL1StreamGateError: boolean
}) {
  const { state } = useLiveL1()
  const phase = state.phase
  const feedLabel = state.feed
  const channels = state.channels
  const [staleNoData, setStaleNoData] = useState(false)

  useEffect(() => {
    if (!wantStream) {
      setStaleNoData(false)
      return
    }
    if (phase !== 'live') {
      setStaleNoData(false)
      return
    }
    if (state.quotes.length > 0 || state.trades.length > 0) {
      setStaleNoData(false)
      return
    }
    const ms = 20_000
    const id = window.setTimeout(() => setStaleNoData(true), ms)
    return () => window.clearTimeout(id)
  }, [wantStream, phase, state.quotes.length, state.trades.length])

  const showInlineLastError =
    state.lastError != null && state.lastErrorCode !== 'us_rth_closed'

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

      {usL1StreamGateError && (
        <Alert color="red" variant="light" title="Market hours unavailable">
          Could not open the market clock stream (<code>/settings/market-clock/stream</code>). You can still
          try Connect; the API will
          reject IEX L1 outside US regular hours and surface that in notifications.
        </Alert>
      )}

      {showInlineLastError && (
        <Alert color="red" variant="light">
          {state.lastError}
        </Alert>
      )}

      {staleNoData && phase === 'live' && (
        <Alert color="yellow" variant="light" title="No BBO or trades yet">
          Still no quotes or trades after 20 seconds. You may be on a very quiet tape, or the upstream
          feed may be blocked — check the API log for <code>alpaca_upstream</code> / subscription lines,
          or run <code>uv run python scripts/diag_alpaca_l1_ws.py …</code>.
        </Alert>
      )}

      <LiveL1SpreadMidChart />

      <Grid gap="md">
        <Grid.Col span={{ base: 12, lg: 7 }}>
          <LiveL1BidAskBubbleChart />
        </Grid.Col>
        <Grid.Col span={{ base: 12, lg: 5 }}>
          <LiveL1OfiHistogram />
        </Grid.Col>
      </Grid>

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
  const { isError: clockError } = useMarketClock()

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
          Enable Alpaca on the API (<code>SHUNYA_API_ALPACA_ENABLED</code> and APCA keys) for IEX L1
          streaming here.
        </Text>
      </Stack>
    )
  }

  if (instrumentKind !== undefined && !kindOk) {
    return (
      <Stack gap="xs">
        <Title order={4}>Alpaca live stream</Title>
        <Text size="sm" c="dimmed">
          Realtime streaming is available for stocks and ETFs only (this symbol is{' '}
          <code>{instrumentKind}</code>).
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
        usL1StreamGateError={clockError}
      />
    </LiveL1Provider>
  )
}
