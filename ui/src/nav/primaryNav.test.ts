import { describe, expect, it } from 'vitest'
import { FLAT_PRIMARY_NAV, findPrimaryNavIndex, navItemMatchesPathname } from './primaryNav'

describe('findPrimaryNavIndex', () => {
  it('resolves dashboard', () => {
    expect(findPrimaryNavIndex('/dashboard')).toBe(0)
  })

  it('resolves studio child routes to Studio nav item', () => {
    const studioIdx = FLAT_PRIMARY_NAV.findIndex((i) => i.to === '/studio' && !i.end)
    expect(studioIdx).toBeGreaterThanOrEqual(0)
    expect(findPrimaryNavIndex('/studio/alpha-uuid-here')).toBe(studioIdx)
  })

  it('falls back to 0 for unknown paths', () => {
    expect(findPrimaryNavIndex('/no-such-path')).toBe(0)
  })
})

describe('navItemMatchesPathname', () => {
  it('matches exact dashboard', () => {
    const dashboard = FLAT_PRIMARY_NAV[0]
    expect(navItemMatchesPathname(dashboard, '/dashboard')).toBe(true)
    expect(navItemMatchesPathname(dashboard, '/search')).toBe(false)
  })
})
