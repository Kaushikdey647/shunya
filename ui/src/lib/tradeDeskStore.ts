/** Client-side command & control state until OMS/EMS APIs are wired. */

export type BlendMode = 'alpha_blend' | 'target_blend'

export type PortfolioSlot = {
  alphaId: string
  alphaName?: string
  weight: number
  /** Optional z-score style conviction for late / risk-aware sizing. */
  convictionZ?: number
}

export type PortfolioRecord = {
  id: string
  name: string
  blendMode: BlendMode
  slots: PortfolioSlot[]
  /** When true, UI prefers streaming / live feeds over backtest snapshots. */
  goLive: boolean
  /** Bump when StrategySpec / blend config changes (for audit trail in UI). */
  strategySpecVersion: number
  createdAt: string
  updatedAt: string
}

export type AdvCapRow = {
  symbol: string
  advPct: number
  usedPct: number
  notes?: string
}

export type RiskSettings = {
  maxGrossLeverage: number
  maxSingleNamePct: number
  maxSectorPct: number
  maxDrawdownStopPct: number
  turnoverBudgetAnnual: number
}

export type SentinelState = {
  killed: boolean
  /** Drawdown from HWM as a positive fraction, e.g. 0.032 for 3.2% */
  drawdownFromHwm: number
}

export type TradeDeskState = {
  portfolios: PortfolioRecord[]
  risk: RiskSettings
  advCaps: AdvCapRow[]
  sentinel: SentinelState
  /** Last EMS parent ids the user opened (for /execution hub). */
  recentParentIds: string[]
  /** Last union of default-universe tickers for the active portfolio (for EMS / desk context). */
  lastPortfolioUniverseTickers: string[]
  lastPortfolioUniverseNote: string | null
}

const STORAGE_KEY = 'shunya_trade_desk_v1'

const defaultRisk: RiskSettings = {
  maxGrossLeverage: 1.5,
  maxSingleNamePct: 10,
  maxSectorPct: 25,
  maxDrawdownStopPct: 12,
  turnoverBudgetAnnual: 4.5,
}

const defaultAdv: AdvCapRow[] = [
  { symbol: 'AAPL', advPct: 0.35, usedPct: 0.12 },
  { symbol: 'MSFT', advPct: 0.4, usedPct: 0.08 },
  { symbol: 'NVDA', advPct: 0.5, usedPct: 0.31 },
  { symbol: 'SPY', advPct: 0.15, usedPct: 0.02 },
]

const defaultState = (): TradeDeskState => ({
  portfolios: [],
  risk: defaultRisk,
  advCaps: defaultAdv,
  sentinel: { killed: false, drawdownFromHwm: 0.018 },
  recentParentIds: ['demo-parent-vwap-1'],
  lastPortfolioUniverseTickers: [],
  lastPortfolioUniverseNote: null,
})

type Listener = () => void
const listeners = new Set<Listener>()

function nowIso() {
  return new Date().toISOString()
}

function load(): TradeDeskState {
  if (typeof window === 'undefined') return defaultState()
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return defaultState()
    const parsed = JSON.parse(raw) as Partial<TradeDeskState>
    return {
      ...defaultState(),
      ...parsed,
      portfolios: Array.isArray(parsed.portfolios) ? parsed.portfolios : [],
      risk: { ...defaultRisk, ...parsed.risk },
      advCaps: Array.isArray(parsed.advCaps) && parsed.advCaps.length ? parsed.advCaps : defaultAdv,
      sentinel: { ...defaultState().sentinel, ...parsed.sentinel },
      recentParentIds:
        Array.isArray(parsed.recentParentIds) && parsed.recentParentIds.length > 0
          ? parsed.recentParentIds
          : ['demo-parent-vwap-1'],
      lastPortfolioUniverseTickers: Array.isArray(parsed.lastPortfolioUniverseTickers)
        ? parsed.lastPortfolioUniverseTickers.map((t) => String(t).toUpperCase())
        : [],
      lastPortfolioUniverseNote:
        typeof parsed.lastPortfolioUniverseNote === 'string'
          ? parsed.lastPortfolioUniverseNote
          : null,
    }
  } catch {
    return defaultState()
  }
}

function save(s: TradeDeskState) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s))
  } catch {
    /* ignore quota */
  }
}

let cache: TradeDeskState = defaultState()
if (typeof window !== 'undefined') {
  cache = load()
}

export function getTradeDeskState(): TradeDeskState {
  return typeof window === 'undefined' ? defaultState() : cache
}

export function subscribeTradeDesk(listener: Listener) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

function emit() {
  for (const l of listeners) l()
}

function mutate(updater: (prev: TradeDeskState) => TradeDeskState) {
  const next = updater(cache)
  save(next)
  cache = next
  emit()
}

export function createPortfolio(name: string): PortfolioRecord {
  const id =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `pf_${Math.random().toString(36).slice(2, 12)}`
  const t = nowIso()
  const row: PortfolioRecord = {
    id,
    name: name.trim() || 'Untitled portfolio',
    blendMode: 'alpha_blend',
    slots: [],
    goLive: false,
    strategySpecVersion: 1,
    createdAt: t,
    updatedAt: t,
  }
  mutate((s) => ({ ...s, portfolios: [...s.portfolios, row] }))
  return row
}

export function updatePortfolio(id: string, patch: Partial<Omit<PortfolioRecord, 'id' | 'createdAt'>>) {
  mutate((s) => ({
    ...s,
    portfolios: s.portfolios.map((p) =>
      p.id === id
        ? {
            ...p,
            ...patch,
            updatedAt: nowIso(),
            strategySpecVersion:
              patch.slots != null || patch.blendMode != null || patch.goLive != null
                ? p.strategySpecVersion + 1
                : p.strategySpecVersion,
          }
        : p,
    ),
  }))
}

export function deletePortfolio(id: string) {
  mutate((s) => ({ ...s, portfolios: s.portfolios.filter((p) => p.id !== id) }))
}

export function addSlotToPortfolio(
  portfolioId: string,
  slot: Omit<PortfolioSlot, 'weight'> & { weight?: number },
) {
  mutate((s) => ({
    ...s,
    portfolios: s.portfolios.map((p) => {
      if (p.id !== portfolioId) return p
      const others = p.slots.filter((x) => x.alphaId !== slot.alphaId)
      const nextSlot: PortfolioSlot = {
        alphaId: slot.alphaId,
        alphaName: slot.alphaName,
        weight: slot.weight ?? (others.length === 0 ? 1 : 1 / (others.length + 1)),
      }
      return {
        ...p,
        slots: [...others, nextSlot],
        updatedAt: nowIso(),
        strategySpecVersion: p.strategySpecVersion + 1,
      }
    }),
  }))
}

export function removeSlotFromPortfolio(portfolioId: string, alphaId: string) {
  mutate((s) => ({
    ...s,
    portfolios: s.portfolios.map((p) =>
      p.id === portfolioId
        ? {
            ...p,
            slots: p.slots.filter((x) => x.alphaId !== alphaId),
            updatedAt: nowIso(),
            strategySpecVersion: p.strategySpecVersion + 1,
          }
        : p,
    ),
  }))
}

export function setSlotWeight(portfolioId: string, alphaId: string, weight: number) {
  mutate((s) => ({
    ...s,
    portfolios: s.portfolios.map((p) =>
      p.id === portfolioId
        ? {
            ...p,
            slots: p.slots.map((x) => (x.alphaId === alphaId ? { ...x, weight } : x)),
            updatedAt: nowIso(),
            strategySpecVersion: p.strategySpecVersion + 1,
          }
        : p,
    ),
  }))
}

export function setRiskSettings(patch: Partial<RiskSettings>) {
  mutate((s) => ({ ...s, risk: { ...s.risk, ...patch } }))
}

export function setAdvCaps(rows: AdvCapRow[]) {
  mutate((s) => ({ ...s, advCaps: rows }))
}

export function setSentinel(patch: Partial<SentinelState>) {
  mutate((s) => ({ ...s, sentinel: { ...s.sentinel, ...patch } }))
}

export function touchRecentParent(parentId: string) {
  mutate((s) => {
    const rest = s.recentParentIds.filter((x) => x !== parentId)
    return { ...s, recentParentIds: [parentId, ...rest].slice(0, 20) }
  })
}

export function setLastPortfolioUniverseSnapshot(tickers: string[], note?: string | null) {
  const seen = new Set<string>()
  const norm: string[] = []
  for (const t of tickers) {
    const u = String(t).trim().toUpperCase()
    if (!u || seen.has(u)) continue
    seen.add(u)
    norm.push(u)
  }
  norm.sort()
  const noteNorm = note ?? null
  mutate((s) => {
    const sameLen = s.lastPortfolioUniverseTickers.length === norm.length
    const sameTickers =
      sameLen && s.lastPortfolioUniverseTickers.every((t, i) => t === norm[i])
    const sameNote = (s.lastPortfolioUniverseNote ?? '') === (noteNorm ?? '')
    if (sameTickers && sameNote) {
      return s
    }
    return {
      ...s,
      lastPortfolioUniverseTickers: norm,
      lastPortfolioUniverseNote: noteNorm,
    }
  })
}

export function deployAlphaToPortfolio(opts: {
  portfolioId: string
  alphaId: string
  alphaName?: string | null
  weight?: number
  sourceJobId?: string
}) {
  const { portfolioId, alphaId, alphaName, weight } = opts
  addSlotToPortfolio(portfolioId, {
    alphaId,
    alphaName: alphaName ?? undefined,
    weight: weight ?? 1,
  })
}
