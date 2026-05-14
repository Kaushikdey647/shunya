import {
  Badge,
  Box,
  Button,
  Group,
  Paper,
  ScrollArea,
  Stack,
  Table,
  Text,
  Title,
} from '@mantine/core'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import DistanceCell from '../components/trade/DistanceCell'
import OrderStreamLine from '../components/trade/OrderStreamLine'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { setSentinel } from '../lib/tradeDeskStore'
import { useTradeDesk } from '../hooks/useTradeDesk'

type PosRow = {
  sym: string
  current: number
  target: number
  pending: number
}

function mockPositions(tick: number): PosRow[] {
  const base: PosRow[] = [
    { sym: 'AAPL', current: 1200, target: 1250, pending: 40 },
    { sym: 'MSFT', current: -400, target: -380, pending: -12 },
    { sym: 'NVDA', current: 220, target: 180, pending: 15 },
    { sym: 'SPY', current: 50, target: 50, pending: 0 },
  ]
  const wobble = Math.sin(tick / 4) * 3
  return base.map((r) => ({
    ...r,
    current: Math.round(r.current + wobble),
    pending: Math.round(r.pending + (tick % 3 === 0 ? 1 : 0)),
  }))
}

const STATUSES = ['PENDING', 'WORKING', 'PARTIAL', 'FILLED', 'CANCELLED'] as const

export default function LiveCockpitPage() {
  const desk = useTradeDesk()
  const density = useMantineTableDensity()
  const [tick, setTick] = useState(0)
  const [liveDd, setLiveDd] = useState(() => desk.sentinel.drawdownFromHwm)
  const [lines, setLines] = useState<string[]>(() => [
    '[09:30:01] parent-7a2 · PENDING → WORKING (slice 1/12)',
    '[09:30:04] parent-7a2 · WORKING · child alp-aa11 @ 182.40',
  ])

  useEffect(() => {
    setLiveDd(desk.sentinel.drawdownFromHwm)
  }, [desk.sentinel.drawdownFromHwm])

  useEffect(() => {
    if (desk.sentinel.killed) return undefined
    const id = window.setInterval(() => {
      setTick((x) => x + 1)
      setLiveDd((d) => Math.max(0, Math.min(0.25, d + (Math.random() - 0.52) * 0.0008)))
    }, 2000)
    return () => window.clearInterval(id)
  }, [desk.sentinel.killed])

  useEffect(() => {
    const id = window.setInterval(() => {
      setLines((prev) => {
        const t = new Date().toLocaleTimeString()
        const st = STATUSES[Math.floor(Math.random() * 3)]
        const next = `${t} · parent-${(Math.random() * 1e9).toString(36).slice(0, 4)} · ${st} (OMS sim)`
        return [...prev.slice(-80), next]
      })
    }, 4200)
    return () => window.clearInterval(id)
  }, [])

  const positions = useMemo(() => mockPositions(tick), [tick])
  const distCap = useMemo(() => {
    const ds = positions.map((r) => Math.abs(r.target - r.current - r.pending))
    return Math.max(1, ...ds)
  }, [positions])
  const ddPct = liveDd * 100

  return (
    <PageScaffold size="fluid" px={{ base: 0, sm: 'md' }} gap={0}>
      <Paper
        withBorder
        radius={0}
        px="md"
        py="sm"
        mb="md"
        style={{
          borderTop: 0,
          borderLeft: 0,
          borderRight: 0,
          position: 'sticky',
          top: 0,
          zIndex: 20,
          background: 'var(--mantine-color-body)',
        }}
      >
        <Group justify="space-between" align="center" wrap="wrap">
          <Stack gap={2}>
            <Text size="xs" c="dimmed" tt="uppercase" fw={700}>
              Sentinel · drawdown from high water mark
            </Text>
            <Group gap="md" align="baseline">
              <Text fw={800} fz="xl" ff="monospace" c={ddPct > 8 ? 'orange' : undefined} style={{ fontVariantNumeric: 'tabular-nums' }}>
                {ddPct.toFixed(2)}%
              </Text>
              {desk.sentinel.killed ? (
                <Badge color="red" variant="filled">
                  KILL ACTIVE
                </Badge>
              ) : (
                <Badge color="teal" variant="light">
                  Systems nominal
                </Badge>
              )}
            </Group>
          </Stack>
          <Group gap="sm">
            <Button
              color="red"
              variant="filled"
              disabled={desk.sentinel.killed}
              onClick={() => {
                if (window.confirm('Activate Sentinel kill-switch? EMS child waves will be cancelled (sim).')) {
                  setSentinel({ killed: true })
                }
              }}
            >
              Kill switch
            </Button>
            {desk.sentinel.killed && (
              <Button variant="light" color="gray" onClick={() => setSentinel({ killed: false })}>
                Reset (sim)
              </Button>
            )}
            <Button component={Link} to="/execution/demo-parent-vwap-1" variant="default" size="compact-sm">
              Open execution tracer
            </Button>
          </Group>
        </Group>
      </Paper>

      <Box px={{ base: 'sm', md: 'md' }}>
        <Stack gap="lg">
          <Title order={2} size="h3">
            Live positions (delta view)
          </Title>
          <Text size="xs" c="dimmed">
            Distance = target − current − pending (EMS). Positive: still short of PM target; negative: past
            target. Bar ∝ |distance| vs row max.
          </Text>
          <Table.ScrollContainer minWidth={720}>
            <Table {...density} striped withTableBorder>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Symbol</Table.Th>
                  <Table.Th ta="right">Current</Table.Th>
                  <Table.Th ta="right">Target (PM)</Table.Th>
                  <Table.Th ta="right">Pending (EMS)</Table.Th>
                  <Table.Th ta="right">Distance</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {positions.map((r) => {
                  const dist = r.target - r.current - r.pending
                  return (
                    <Table.Tr key={r.sym}>
                      <Table.Td ff="monospace" fw={600}>
                        {r.sym}
                      </Table.Td>
                      <Table.Td ta="right" ff="monospace">
                        {r.current}
                      </Table.Td>
                      <Table.Td ta="right" ff="monospace">
                        {r.target}
                      </Table.Td>
                      <Table.Td ta="right" ff="monospace">
                        {r.pending}
                      </Table.Td>
                      <Table.Td>
                        <DistanceCell dist={dist} cap={distCap} />
                      </Table.Td>
                    </Table.Tr>
                  )
                })}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>

          <Title order={2} size="h3">
            Order stream
          </Title>
          <Paper withBorder p="sm" radius="md" className="order-stream-panel">
            <ScrollArea h={280}>
              <Stack gap={0}>
                {lines.map((ln, i) => (
                  <OrderStreamLine key={`${i}-${ln.slice(0, 24)}`} line={ln} />
                ))}
              </Stack>
            </ScrollArea>
          </Paper>
        </Stack>
      </Box>
    </PageScaffold>
  )
}
