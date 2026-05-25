import { AppShell as MantineAppShell, Box, useComputedColorScheme } from '@mantine/core'
import { useDisclosure } from '@mantine/hooks'
import { useRef, useState } from 'react'
import { Outlet } from 'react-router-dom'
import CommandPalette from './CommandPalette'
import SideNav from './SideNav'
import TopNav from './TopNav'
import type { TickerSearchHandle } from './TickerSearch'
import { useGlobalAppShortcuts } from '../hooks/useGlobalAppShortcuts'

export default function AppShell() {
  const colorScheme = useComputedColorScheme('light', { getInitialValueInEffect: false })
  const [cmdOpen, setCmdOpen] = useState(false)
  const [mobileNavOpened, { toggle: toggleMobileNav, close: closeMobileNav }] = useDisclosure()
  const tickerSearchRef = useRef<TickerSearchHandle | null>(null)

  useGlobalAppShortcuts({
    commandPaletteOpen: cmdOpen,
    setCommandPaletteOpen: setCmdOpen,
    tickerSearchRef,
  })

  return (
    <MantineAppShell
      header={{ height: 44 }}
      navbar={{
        width: 200,
        breakpoint: 'sm',
        collapsed: { mobile: !mobileNavOpened },
      }}
      padding={0}
      transitionDuration={280}
      transitionTimingFunction="cubic-bezier(0.4, 0, 0.2, 1)"
    >
      <MantineAppShell.Header>
        <TopNav
          mobileNavOpened={mobileNavOpened}
          onMobileNavToggle={toggleMobileNav}
          tickerSearchRef={tickerSearchRef}
        />
      </MantineAppShell.Header>
      <MantineAppShell.Navbar p={0} bg={colorScheme === 'dark' ? '#161b22' : undefined}>
        <SideNav onNavigate={closeMobileNav} />
      </MantineAppShell.Navbar>
      {/* Do not set p/pt/pb/ps/pe on Main — they override shell offsets for header, navbar, aside, footer. */}
      <MantineAppShell.Main>
        <Box pb={{ base: 'sm', md: 'md' }}>
          <Outlet />
        </Box>
      </MantineAppShell.Main>
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
    </MantineAppShell>
  )
}
