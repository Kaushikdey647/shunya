import { describe, expect, it } from 'vitest'
import { APP_TICKER_SEARCH_ROOT, MONACO_ROOT_ATTR, isInsideAppTickerSearch, isInsideMonaco } from './domGuards'

describe('isInsideMonaco', () => {
  it('returns true for descendants of Monaco root', () => {
    const root = document.createElement('div')
    root.setAttribute(MONACO_ROOT_ATTR, '')
    const inner = document.createElement('div')
    root.appendChild(inner)
    document.body.appendChild(root)
    expect(isInsideMonaco(inner)).toBe(true)
    root.remove()
  })

  it('returns false outside root', () => {
    const el = document.createElement('span')
    document.body.appendChild(el)
    expect(isInsideMonaco(el)).toBe(false)
    el.remove()
  })
})

describe('isInsideAppTickerSearch', () => {
  it('returns true inside ticker root', () => {
    const root = document.createElement('div')
    root.setAttribute(APP_TICKER_SEARCH_ROOT, '')
    const input = document.createElement('input')
    root.appendChild(input)
    document.body.appendChild(root)
    expect(isInsideAppTickerSearch(input)).toBe(true)
    root.remove()
  })
})
