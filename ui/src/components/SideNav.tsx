import { NavLink as MantineNavLink, Stack, Text } from '@mantine/core'
import { Link, matchPath, useLocation } from 'react-router-dom'

type NavItem = { to: string; label: string; end?: boolean }

const RESEARCH: NavItem[] = [
  { to: '/', label: 'Home', end: true },
  { to: '/search', label: 'Search', end: true },
  { to: '/data', label: 'Data summary', end: true },
  { to: '/settings', label: 'Settings', end: true },
]

const STUDIO: NavItem[] = [
  { to: '/studio', label: 'Studio', end: false },
  { to: '/universe', label: 'Universes', end: false },
  { to: '/backtests', label: 'Backtests', end: false },
]

const TRADE: NavItem[] = [
  { to: '/portfolios', label: 'Portfolios', end: false },
  { to: '/live', label: 'Live', end: true },
  { to: '/trade/account', label: 'Account', end: true },
  { to: '/execution', label: 'Execution', end: false },
  { to: '/risk', label: 'Risk', end: true },
]

function Section({ title, items, onNavigate }: { title: string; items: NavItem[]; onNavigate?: () => void }) {
  const location = useLocation()
  return (
    <Stack gap={4}>
      <Text size="xs" tt="uppercase" fw={700} c="dimmed" px="xs">
        {title}
      </Text>
      {items.map((item) => {
        const active = Boolean(matchPath({ path: item.to, end: item.end ?? false }, location.pathname))
        return (
          <MantineNavLink
            key={item.to}
            component={Link}
            to={item.to}
            label={item.label}
            active={active}
            onClick={() => onNavigate?.()}
          />
        )
      })}
    </Stack>
  )
}

type Props = {
  onNavigate?: () => void
}

export default function SideNav({ onNavigate }: Props) {
  return (
    <Stack component="nav" gap="lg" p="xs" aria-label="Primary">
      <Section title="Research" items={RESEARCH} onNavigate={onNavigate} />
      <Section title="Studio" items={STUDIO} onNavigate={onNavigate} />
      <Section title="Trade" items={TRADE} onNavigate={onNavigate} />
    </Stack>
  )
}
