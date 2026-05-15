import { NavLink as MantineNavLink, Stack, Text } from '@mantine/core'
import { Link, useLocation } from 'react-router-dom'
import { navItemMatchesPathname, PRIMARY_NAV_SECTIONS, type PrimaryNavItem } from '../nav/primaryNav'

function Section({
  title,
  items,
  onNavigate,
}: {
  title: string
  items: PrimaryNavItem[]
  onNavigate?: () => void
}) {
  const location = useLocation()
  return (
    <Stack gap={4}>
      <Text size="xs" tt="uppercase" fw={700} c="dimmed" px="xs">
        {title}
      </Text>
      {items.map((item) => {
        const active = navItemMatchesPathname(item, location.pathname)
        return (
          <MantineNavLink
            key={item.to}
            component={Link}
            to={item.to}
            label={item.label}
            active={active}
            aria-current={active ? 'page' : undefined}
            onClick={() => onNavigate?.()}
            styles={{
              label: {
                color: active ? 'var(--mantine-color-yellow-filled)' : undefined,
                fontWeight: active ? 600 : undefined,
              },
            }}
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
      {PRIMARY_NAV_SECTIONS.map((section) => (
        <Section key={section.title} title={section.title} items={section.items} onNavigate={onNavigate} />
      ))}
    </Stack>
  )
}
