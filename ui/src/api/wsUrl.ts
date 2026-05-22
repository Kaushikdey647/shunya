/** Same origin rules as HTTP {@link apiFetch} in `./client`. */
const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export function buildWebSocketUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  const base = API_BASE.endsWith('/') ? API_BASE.slice(0, -1) : API_BASE

  if (base.startsWith('http://') || base.startsWith('https://')) {
    const httpUrl = `${base}${p}`
    const u = new URL(httpUrl)
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:'
    return u.toString()
  }

  const wsProto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = typeof window !== 'undefined' ? window.location.host : 'localhost'
  return `${wsProto}//${host}${base}${p}`
}

/** IEX L1: BBO quotes + trades (unified stream). */
export function instrumentAlpacaL1WebSocketUrl(symbol: string): string {
  const enc = encodeURIComponent(symbol)
  return buildWebSocketUrl(`/instruments/${enc}/stream/alpaca-l1`)
}

/** @deprecated Use {@link instrumentAlpacaL1WebSocketUrl}; `alpaca-bars` is removed server-side. */
export function instrumentAlpacaLiveWebSocketUrl(symbol: string): string {
  return instrumentAlpacaL1WebSocketUrl(symbol)
}
