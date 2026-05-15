import { matchPath } from 'react-router-dom'

export type PrimaryNavItem = { to: string; label: string; end?: boolean }

export type PrimaryNavSection = { title: string; items: PrimaryNavItem[] }

export const PRIMARY_NAV_RESEARCH: PrimaryNavItem[] = [
  { to: '/dashboard', label: 'Dashboard', end: true },
  { to: '/search', label: 'Search', end: true },
  { to: '/data', label: 'Data summary', end: true },
  { to: '/settings', label: 'Settings', end: true },
]

export const PRIMARY_NAV_STUDIO: PrimaryNavItem[] = [
  { to: '/studio', label: 'Studio', end: false },
  { to: '/universe', label: 'Universes', end: false },
  { to: '/backtests', label: 'Backtests', end: false },
]

export const PRIMARY_NAV_TRADE: PrimaryNavItem[] = [
  { to: '/portfolios', label: 'Portfolios', end: false },
  { to: '/live', label: 'Live', end: true },
  { to: '/trade/account', label: 'Account', end: true },
  { to: '/execution', label: 'Execution', end: false },
  { to: '/risk', label: 'Risk', end: true },
]

export const PRIMARY_NAV_SECTIONS: PrimaryNavSection[] = [
  { title: 'Research', items: PRIMARY_NAV_RESEARCH },
  { title: 'Studio', items: PRIMARY_NAV_STUDIO },
  { title: 'Trade', items: PRIMARY_NAV_TRADE },
]

/** Flat route order as shown top-to-bottom in the sidebar (Research → Studio → Trade). */
export const FLAT_PRIMARY_NAV: PrimaryNavItem[] = PRIMARY_NAV_SECTIONS.flatMap((s) => s.items)

export function navItemMatchesPathname(item: PrimaryNavItem, pathname: string): boolean {
  return Boolean(matchPath({ path: item.to, end: item.end ?? false }, pathname))
}

/** Index into `FLAT_PRIMARY_NAV` for keyboard cycling; falls back to `0` if nothing matches. */
export function findPrimaryNavIndex(pathname: string): number {
  const idx = FLAT_PRIMARY_NAV.findIndex((item) => navItemMatchesPathname(item, pathname))
  return idx >= 0 ? idx : 0
}
