import {
  Badge,
  Button,
  Group,
  Modal,
  NumberInput,
  Paper,
  SegmentedControl,
  Slider,
  Stack,
  Switch,
  Table,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useQueries } from '@tanstack/react-query'
import { Link, useParams } from 'react-router-dom'
import PageScaffold from '../components/PageScaffold'
import CorrelationHeatmap from '../components/trade/CorrelationHeatmap'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import {
  deployAlphaToPortfolio,
  removeSlotFromPortfolio,
  setLastPortfolioUniverseSnapshot,
  setSlotWeight,
  updatePortfolio,
} from '../lib/tradeDeskStore'
import { useTradeDesk } from '../hooks/useTradeDesk'
import { useEffect, useMemo, useState } from 'react'
import { getAlpha, getUniverseTickers } from '../api/endpoints'

function ledgerRows(alphaIds: string[], goLive: boolean) {
  const syms = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']
  return syms.flatMap((sym) =>
    alphaIds.map((aid, idx) => {
      const base = ((aid.charCodeAt(0) ?? 0) + sym.charCodeAt(0)) % 40
      const theoretical = base - 20 + idx * 3
      const slip = goLive ? ((idx * 7) % 5) - 2 : 0
      const actual = theoretical + slip
      return {
        sym,
        alpha: aid.slice(0, 8),
        theoretical,
        actual,
        delta: actual - theoretical,
      }
    }),
  )
}

export default function PortfolioWorkspacePage() {
  const { id } = useParams<{ id: string }>()
  const desk = useTradeDesk()
  const density = useMantineTableDensity()
  const [pasteId, setPasteId] = useState('')
  const [uniModal, setUniModal] = useState(false)

  const portfolio = useMemo(() => desk.portfolios.find((p) => p.id === id), [desk.portfolios, id])

  const alphaQueries = useQueries({
    queries: (portfolio?.slots ?? []).map((s) => ({
      queryKey: ['alpha', s.alphaId],
      queryFn: () => getAlpha(s.alphaId),
      enabled: Boolean(portfolio && id),
    })),
  })

  const universeIds = useMemo(() => {
    const u = new Set<string>()
    for (const q of alphaQueries) {
      const du = q.data?.default_universe_id
      if (du) u.add(du)
    }
    return [...u]
  }, [alphaQueries.map((q) => q.data?.default_universe_id ?? '').join('|')])

  const tickerQueries = useQueries({
    queries: universeIds.map((uid) => ({
      queryKey: ['universe-tickers', uid],
      queryFn: () => getUniverseTickers(uid),
      enabled: Boolean(portfolio && uid),
    })),
  })

  const unionTickers = useMemo(() => {
    const s = new Set<string>()
    for (const q of tickerQueries) {
      for (const t of q.data?.tickers ?? []) s.add(t)
    }
    return [...s].sort()
  }, [tickerQueries.map((q) => (q.data?.tickers ?? []).join(',')).join(';')])

  useEffect(() => {
    if (!portfolio) return
    setLastPortfolioUniverseSnapshot(
      unionTickers,
      `portfolio:${portfolio.name} (${portfolio.id.slice(0, 8)}…)`,
    )
  }, [portfolio?.id, portfolio?.name, unionTickers.join(',')])

  if (!id) {
    return (
      <PageScaffold>
        <Text c="dimmed">Missing portfolio id.</Text>
      </PageScaffold>
    )
  }

  if (!portfolio) {
    return (
      <PageScaffold>
        <Button component={Link} to="/portfolios" variant="default">
          ← Portfolios
        </Button>
        <Text c="dimmed">Portfolio not found (it may have been deleted).</Text>
      </PageScaffold>
    )
  }

  const heatLabels = portfolio.slots.map((s) => s.alphaName ?? s.alphaId)
  const rows = ledgerRows(
    portfolio.slots.map((s) => s.alphaId),
    portfolio.goLive,
  )

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Group justify="space-between" align="flex-start" wrap="wrap">
        <div>
          <Button component={Link} to="/portfolios" variant="default" size="compact-sm" mb="xs">
            ← Registry
          </Button>
          <Title order={1}>{portfolio.name}</Title>
          <Group gap="xs" mt={4}>
            <Badge variant="light" color="gray" ff="monospace" size="sm">
              {portfolio.id}
            </Badge>
            <Badge variant="outline" color="yellow" size="sm">
              StrategySpec v{portfolio.strategySpecVersion}
            </Badge>
            <Badge variant="light" color="gray" size="sm">
              Union universe: {unionTickers.length} names
            </Badge>
            <Button variant="light" size="compact-xs" onClick={() => setUniModal(true)}>
              View tickers
            </Button>
          </Group>
          <Modal opened={uniModal} onClose={() => setUniModal(false)} title="Portfolio union tickers" size="lg">
            <Text size="xs" c="dimmed" mb="sm">
              Union of default universes from each slot alpha (alphas without a default universe contribute nothing).
            </Text>
            <Text component="pre" fz="xs" ff="monospace" style={{ whiteSpace: 'pre-wrap', maxHeight: 360, overflow: 'auto' }}>
              {unionTickers.length ? unionTickers.join(', ') : '—'}
            </Text>
            <Button
              mt="md"
              variant="default"
              disabled={!unionTickers.length}
              onClick={() => {
                void navigator.clipboard.writeText(unionTickers.join(','))
              }}
            >
              Copy CSV
            </Button>
          </Modal>
        </div>
        <Stack gap="xs" align="flex-end">
          <Group gap="xs" wrap="wrap" justify="flex-end">
            <Button component={Link} to="/live" size="compact-sm" variant="subtle" color="gray">
              Live cockpit
            </Button>
            <Button component={Link} to="/risk" size="compact-sm" variant="subtle" color="gray">
              Risk center
            </Button>
          </Group>
          <Switch
            label="Go live"
            description="Prefer streaming event feed over backtest / snapshot for this book."
            checked={portfolio.goLive}
            onChange={(e) => updatePortfolio(portfolio.id, { goLive: e.currentTarget.checked })}
            color="gray"
            size="sm"
          />
          <Button component={Link} to="/studio" variant="light" color="yellow" size="compact-sm">
            Open Studio to attach alphas
          </Button>
        </Stack>
      </Group>

      <Paper withBorder p="md" radius="md">
        <Text fw={600} size="sm" mb="xs">
          Quick attach (alpha id)
        </Text>
        <Text size="xs" c="dimmed" mb="sm">
          Temporary bridge until portfolio API lists alphas in-product. Prefer <Text span fw={600}>Add to portfolio</Text>{' '}
          from an alpha workspace.
        </Text>
        <Group align="flex-end" wrap="wrap">
          <TextInput
            label="Alpha id"
            placeholder="uuid from Studio"
            value={pasteId}
            onChange={(e) => setPasteId(e.currentTarget.value)}
            maw={480}
            w="100%"
            miw={220}
            ff="monospace"
          />
          <Button
            color="yellow"
            disabled={!pasteId.trim()}
            onClick={() => {
              deployAlphaToPortfolio({
                portfolioId: portfolio.id,
                alphaId: pasteId.trim(),
                weight: 1,
              })
              setPasteId('')
            }}
          >
            Add sleeve
          </Button>
        </Group>
      </Paper>

      <Paper withBorder p="md" radius="md">
        <Text fw={600} size="sm" mb="md">
          Blending config
        </Text>
        <SegmentedControl
          size="sm"
          maw={440}
          value={portfolio.blendMode}
          onChange={(v) =>
            updatePortfolio(portfolio.id, { blendMode: v as 'alpha_blend' | 'target_blend' })
          }
          data={[
            { label: 'Alpha blend (early aggregation)', value: 'alpha_blend' },
            { label: 'Target blend (late aggregation)', value: 'target_blend' },
          ]}
        />
        <Text size="xs" c="dimmed" mt="sm">
          Early: combine signals before risk / execution. Late: aggregate target weights after per-alpha constraints.
        </Text>
      </Paper>

      <Title order={3} size="h4">
        Strategy slots
      </Title>
      {portfolio.slots.length === 0 ? (
        <Text c="dimmed" size="sm">
          No alphas yet.
        </Text>
      ) : (
        <Stack gap="lg">
          {portfolio.slots.map((slot) => (
            <Paper key={slot.alphaId} withBorder p="md" radius="md">
              <Group justify="space-between" align="flex-start" wrap="wrap" mb="sm">
                <div>
                  <Text fw={600}>{slot.alphaName ?? 'Alpha'}</Text>
                  <Text size="xs" ff="monospace" c="dimmed">
                    {slot.alphaId}
                  </Text>
                </div>
                <Button
                  component={Link}
                  size="compact-xs"
                  variant="light"
                  to={`/studio/${encodeURIComponent(slot.alphaId)}`}
                >
                  Open in Studio
                </Button>
              </Group>
              <Text size="xs" fw={600} mb={6}>
                Weight
              </Text>
              <Slider
                min={0.05}
                max={3}
                step={0.05}
                value={slot.weight}
                onChange={(w) => setSlotWeight(portfolio.id, slot.alphaId, w)}
                marks={[
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                ]}
              />
              <Group mt="md" grow align="flex-end">
                <NumberInput
                  label="Conviction z (optional)"
                  description="Reserved for z-scored sizing in PortfolioManager."
                  value={slot.convictionZ ?? 0}
                  onChange={(v) => {
                    const z = typeof v === 'number' ? v : 0
                    const next = portfolio.slots.map((s) =>
                      s.alphaId === slot.alphaId ? { ...s, convictionZ: z === 0 ? undefined : z } : s,
                    )
                    updatePortfolio(portfolio.id, { slots: next })
                  }}
                  decimalScale={2}
                  step={0.1}
                />
                <Button
                  color="red"
                  variant="subtle"
                  size="xs"
                  onClick={() => removeSlotFromPortfolio(portfolio.id, slot.alphaId)}
                >
                  Remove slot
                </Button>
              </Group>
            </Paper>
          ))}
        </Stack>
      )}

      <CorrelationHeatmap labels={heatLabels} />

      <Title order={3} size="h4" mt="md">
        Virtual ledger
      </Title>
      <Text size="xs" c="dimmed" mb="xs">
        Theoretical vs actual per sub-alpha (mock grid — replace with OMS reconciliation feed).
      </Text>
      <Table.ScrollContainer minWidth={560}>
        <Table {...density} verticalSpacing="xs" horizontalSpacing="xs" fz="xs" striped withTableBorder>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Symbol</Table.Th>
              <Table.Th>Alpha</Table.Th>
              <Table.Th ta="right">Theoretical sh.</Table.Th>
              <Table.Th ta="right">Actual sh.</Table.Th>
              <Table.Th ta="right">Δ</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {rows.slice(0, 24).map((r, i) => (
              <Table.Tr key={`${r.sym}-${r.alpha}-${i}`}>
                <Table.Td ff="monospace">{r.sym}</Table.Td>
                <Table.Td ff="monospace">{r.alpha}…</Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {r.theoretical}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {r.actual}
                </Table.Td>
                <Table.Td ta="right" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }} c={r.delta === 0 ? undefined : 'yellow'}>
                  {r.delta > 0 ? `+${r.delta}` : r.delta}
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </PageScaffold>
  )
}
