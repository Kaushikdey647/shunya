const STORAGE_KEY = 'shunya_watchlist_v1'

const SYMBOL_PATTERN = /^[-A-Z0-9^.]{1,32}$/i

export function readWatchlist(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((x): x is string => typeof x === 'string')
      .map((s) => s.trim().toUpperCase())
      .filter((s) => SYMBOL_PATTERN.test(s))
  } catch {
    return []
  }
}

export function writeWatchlist(symbols: string[]): void {
  const norm = [...new Set(symbols.map((s) => s.trim().toUpperCase()).filter((s) => SYMBOL_PATTERN.test(s)))]
  localStorage.setItem(STORAGE_KEY, JSON.stringify(norm))
}

export function addWatchlistSymbol(symbol: string): string[] {
  const s = symbol.trim().toUpperCase()
  if (!SYMBOL_PATTERN.test(s)) return readWatchlist()
  const cur = readWatchlist()
  if (cur.includes(s)) return cur
  const next = [...cur, s]
  writeWatchlist(next)
  return next
}

export function removeWatchlistSymbol(symbol: string): string[] {
  const s = symbol.trim().toUpperCase()
  const next = readWatchlist().filter((x) => x !== s)
  writeWatchlist(next)
  return next
}
