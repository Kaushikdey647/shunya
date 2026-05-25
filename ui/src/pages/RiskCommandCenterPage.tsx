import {
  Button,
  NumberInput,
  Paper,
  SimpleGrid,
  Table,
  Text,
  Title,
  useComputedColorScheme,
} from '@mantine/core'
import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
} from 'recharts'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'
import { setAdvCaps, setRiskSettings } from '../lib/tradeDeskStore'
import type { AdvCapRow, RiskSettings } from '../lib/tradeDeskStore'
import { useTradeDesk } from '../hooks/useTradeDesk'

function factorRadar(risk: RiskSettings) {
  const skew = risk.maxGrossLeverage + risk.maxSingleNamePct / 20
  const bench = 50
  return [
    { factor: 'Growth', a: Math.min(100, 40 + skew * 10), b: bench },
    { factor: 'Value', a: Math.min(100, 55 - skew * 5), b: bench },
    { factor: 'Momentum', a: Math.min(100, 48 + risk.turnoverBudgetAnnual * 4), b: bench },
    { factor: 'Quality', a: Math.min(100, 62 - risk.maxDrawdownStopPct * 0.8), b: bench },
    { factor: 'Size', a: Math.min(100, 35 + risk.maxSectorPct * 0.6), b: bench },
  ]
}

export default function RiskCommandCenterPage() {
  const desk = useTradeDesk()
  const density = useMantineTableDensity()
  const scheme = useComputedColorScheme('light', { getInitialValueInEffect: false })
  const radar = useMemo(() => factorRadar(desk.risk), [desk.risk])

  const gridStroke = scheme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : '#c4c4c4'
  const gridStrokeW = scheme === 'dark' ? 1.25 : 1
  const benchStroke = scheme === 'dark' ? '#888888' : '#6b7280'
  const portFillOp = scheme === 'dark' ? 0.52 : 0.35

  const patchRisk = (patch: Partial<RiskSettings>) => {
    setRiskSettings(patch)
  }

  const onAdvCell = (idx: number, field: keyof AdvCapRow, value: number) => {
    const next = desk.advCaps.map((row, i) => (i === idx ? { ...row, [field]: value } : row))
    setAdvCaps(next as AdvCapRow[])
  }

  return (
    <PageScaffold size="fluid" px={{ base: 'sm', md: 'md' }}>
      <Title order={1}>Risk command center</Title>
      <Text c="dimmed" size="sm">
        Global hard limits for <Text span ff="monospace">cvxpy</Text> / RiskEngine — persisted locally until config API ships.
      </Text>

      <Paper withBorder p="md" radius="md">
        <Title order={3} size="h4" mb="md">
          Global constraints
        </Title>
        <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="md">
          <NumberInput
            maw={220}
            label="Max gross leverage (×)"
            value={desk.risk.maxGrossLeverage}
            onChange={(v) => typeof v === 'number' && patchRisk({ maxGrossLeverage: v })}
            min={0.5}
            max={4}
            step={0.05}
            decimalScale={2}
            size="sm"
          />
          <NumberInput
            maw={220}
            label="Max single name (%)"
            value={desk.risk.maxSingleNamePct}
            onChange={(v) => typeof v === 'number' && patchRisk({ maxSingleNamePct: v })}
            min={1}
            max={50}
            step={0.5}
            decimalScale={1}
            size="sm"
          />
          <NumberInput
            maw={220}
            label="Max sector (%)"
            value={desk.risk.maxSectorPct}
            onChange={(v) => typeof v === 'number' && patchRisk({ maxSectorPct: v })}
            min={5}
            max={80}
            step={1}
            size="sm"
          />
          <NumberInput
            maw={220}
            label="Max drawdown stop (%)"
            value={desk.risk.maxDrawdownStopPct}
            onChange={(v) => typeof v === 'number' && patchRisk({ maxDrawdownStopPct: v })}
            min={2}
            max={40}
            step={0.5}
            size="sm"
          />
          <NumberInput
            maw={220}
            label="Turnover budget (annual ×)"
            value={desk.risk.turnoverBudgetAnnual}
            onChange={(v) => typeof v === 'number' && patchRisk({ turnoverBudgetAnnual: v })}
            min={0.5}
            max={20}
            step={0.1}
            decimalScale={2}
            size="sm"
          />
        </SimpleGrid>
      </Paper>

      <Title order={3} size="h4">
        ADV cap registry
      </Title>
      <Text size="xs" c="dimmed" mb="xs">
        Market impact budget vs utilization (editable mock rows).
      </Text>
      <Table.ScrollContainer minWidth={520}>
        <Table {...density} striped withTableBorder>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Symbol</Table.Th>
              <Table.Th ta="right">ADV cap</Table.Th>
              <Table.Th ta="right">Used</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {desk.advCaps.map((row, idx) => (
              <Table.Tr key={row.symbol}>
                <Table.Td ff="monospace">{row.symbol}</Table.Td>
                <Table.Td ta="right">
                  <NumberInput
                    hideControls
                    size="xs"
                    maw={120}
                    value={row.advPct}
                    onChange={(v) => typeof v === 'number' && onAdvCell(idx, 'advPct', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    decimalScale={2}
                  />
                </Table.Td>
                <Table.Td ta="right">
                  <NumberInput
                    hideControls
                    size="xs"
                    maw={120}
                    value={row.usedPct}
                    onChange={(v) => typeof v === 'number' && onAdvCell(idx, 'usedPct', v)}
                    min={0}
                    max={1}
                    step={0.01}
                    decimalScale={2}
                  />
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>

      <Title order={3} size="h4" mt="md">
        Factor exposure (proxy)
      </Title>
      <Text size="xs" c="dimmed" mb="xs">
        Portfolio vs benchmark (50 neutral MVP) — replace with Barra / internal factor loadings.
      </Text>
      <Paper withBorder p="md" radius="md" h={380}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="72%" data={radar}>
            <PolarGrid stroke={gridStroke} strokeWidth={gridStrokeW} />
            <PolarAngleAxis dataKey="factor" tick={{ fontSize: 11, fill: scheme === 'dark' ? '#aaa' : '#555' }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: scheme === 'dark' ? '#888' : '#666' }} />
            <Radar
              name="Benchmark"
              dataKey="b"
              stroke={benchStroke}
              strokeWidth={2}
              strokeDasharray="5 4"
              fill={benchStroke}
              fillOpacity={scheme === 'dark' ? 0.12 : 0.08}
              isAnimationActive={false}
            />
            <Radar
              name="Portfolio"
              dataKey="a"
              stroke="var(--mantine-color-yellow-5)"
              strokeWidth={2.5}
              fill="var(--mantine-color-yellow-5)"
              fillOpacity={portFillOp}
              isAnimationActive={false}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <RTooltip />
          </RadarChart>
        </ResponsiveContainer>
      </Paper>

      <Button component={Link} to="/live" variant="subtle" size="compact-sm" mt="md">
        ← Live cockpit
      </Button>
    </PageScaffold>
  )
}
