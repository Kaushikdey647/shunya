import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type ReactNode,
} from 'react'
import type { AlpacaL1WsMessage, AlpacaL1WsQuote, AlpacaL1WsTrade } from '../api/types'
import { isUsableAlpacaL1Quote, isUsableAlpacaL1Trade } from './l1Derived'
import { instrumentAlpacaL1WebSocketUrl } from '../api/wsUrl'

const QUOTE_RING_MAX = 2000
const TRADE_RING_MAX = 400
const TAPE_MAX = 60

export type LiveL1Phase = 'idle' | 'connecting' | 'live' | 'error'

export type LiveL1TapeRow =
  | { kind: 'trade'; t: AlpacaL1WsTrade }
  | { kind: 'correction'; summary: string; time: string }
  | { kind: 'cancel'; summary: string; time: string }

export type LiveL1State = {
  quotes: AlpacaL1WsQuote[]
  trades: AlpacaL1WsTrade[]
  tape: LiveL1TapeRow[]
  phase: LiveL1Phase
  lastError: string | null
  feed: string | null
  channels: string[] | null
}

const initialState = (): LiveL1State => ({
  quotes: [],
  trades: [],
  tape: [],
  phase: 'idle',
  lastError: null,
  feed: null,
  channels: null,
})

type Action =
  | { type: 'RESET' }
  | { type: 'SET_PHASE'; phase: LiveL1Phase }
  | { type: 'SET_ERROR'; message: string }
  | { type: 'HELLO'; feed: string; channels: string[] }
  | { type: 'QUOTE'; q: AlpacaL1WsQuote }
  | { type: 'TRADE'; t: AlpacaL1WsTrade }
  | { type: 'CORRECTION'; summary: string; time: string }
  | { type: 'CANCEL'; summary: string; time: string }
  | { type: 'WS_CLOSED' }

function pushRing<T>(arr: T[], item: T, max: number): T[] {
  const next = [...arr, item]
  if (next.length <= max) return next
  return next.slice(next.length - max)
}

function reducer(state: LiveL1State, action: Action): LiveL1State {
  switch (action.type) {
    case 'RESET':
      return initialState()
    case 'SET_PHASE':
      return { ...state, phase: action.phase }
    case 'SET_ERROR':
      return { ...state, phase: 'error', lastError: action.message }
    case 'HELLO':
      return {
        ...state,
        phase: 'live',
        feed: action.feed,
        channels: action.channels,
        lastError: null,
      }
    case 'QUOTE':
      return { ...state, quotes: pushRing(state.quotes, action.q, QUOTE_RING_MAX) }
    case 'TRADE': {
      const trades = pushRing(state.trades, action.t, TRADE_RING_MAX)
      const tape = pushRing(
        state.tape,
        { kind: 'trade' as const, t: action.t },
        TAPE_MAX,
      )
      return { ...state, trades, tape }
    }
    case 'CORRECTION': {
      const tape = pushRing(
        state.tape,
        { kind: 'correction' as const, summary: action.summary, time: action.time },
        TAPE_MAX,
      )
      return { ...state, tape }
    }
    case 'CANCEL': {
      const tape = pushRing(
        state.tape,
        { kind: 'cancel' as const, summary: action.summary, time: action.time },
        TAPE_MAX,
      )
      return { ...state, tape }
    }
    case 'WS_CLOSED':
      if (state.phase === 'error') return state
      return { ...state, phase: 'idle', feed: null, channels: null }
    default:
      return state
  }
}

function parseAlpacaL1Message(raw: string): AlpacaL1WsMessage | null {
  try {
    const msg = JSON.parse(raw) as Record<string, unknown>
    const t = msg.type
    if (
      t === 'hello' ||
      t === 'quote' ||
      t === 'trade' ||
      t === 'trade_correction' ||
      t === 'trade_cancel' ||
      t === 'error'
    ) {
      return msg as AlpacaL1WsMessage
    }
    return null
  } catch {
    return null
  }
}

type LiveL1Ctx = {
  symbol: string
  state: LiveL1State
  dispatch: React.Dispatch<Action>
}

const LiveL1Context = createContext<LiveL1Ctx | null>(null)

export function useLiveL1(): LiveL1Ctx {
  const v = useContext(LiveL1Context)
  if (!v) throw new Error('useLiveL1 must be used within LiveL1Provider')
  return v
}

type ProviderProps = {
  symbol: string
  /** When true, open WebSocket and ingest; when false, clear state and disconnect. */
  streamActive: boolean
  children: ReactNode
}

export function LiveL1Provider({ symbol, streamActive, children }: ProviderProps) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)

  useEffect(() => {
    dispatch({ type: 'RESET' })
  }, [symbol])

  useEffect(() => {
    if (!streamActive) {
      dispatch({ type: 'RESET' })
      return
    }

    dispatch({ type: 'SET_PHASE', phase: 'connecting' })
    const url = instrumentAlpacaL1WebSocketUrl(symbol)
    const ws = new WebSocket(url)

    ws.onmessage = (ev) => {
      const msg = parseAlpacaL1Message(String(ev.data))
      if (!msg) return
      if (msg.type === 'hello') {
        dispatch({ type: 'HELLO', feed: msg.feed, channels: msg.channels })
        return
      }
      if (msg.type === 'error') {
        dispatch({ type: 'SET_ERROR', message: msg.message })
        ws.close()
        return
      }
      if (msg.type === 'quote') {
        if (isUsableAlpacaL1Quote(msg as AlpacaL1WsQuote)) {
          dispatch({ type: 'QUOTE', q: msg as AlpacaL1WsQuote })
        }
        return
      }
      if (msg.type === 'trade') {
        if (isUsableAlpacaL1Trade(msg as AlpacaL1WsTrade)) {
          dispatch({ type: 'TRADE', t: msg as AlpacaL1WsTrade })
        }
        return
      }
      if (msg.type === 'trade_correction') {
        const s = `${msg.symbol} corr ${msg.original_price}→${msg.corrected_price} sz ${msg.original_size}→${msg.corrected_size}`
        dispatch({ type: 'CORRECTION', summary: s, time: msg.time })
        return
      }
      if (msg.type === 'trade_cancel') {
        const s = `${msg.symbol} cancel px ${msg.price} sz ${msg.size} id ${msg.id ?? '—'}`
        dispatch({ type: 'CANCEL', summary: s, time: msg.time })
      }
    }

    ws.onerror = () => {
      dispatch({ type: 'SET_ERROR', message: 'WebSocket connection error.' })
    }

    ws.onclose = () => {
      dispatch({ type: 'WS_CLOSED' })
    }

    return () => {
      ws.close()
    }
  }, [streamActive, symbol])

  const value = useMemo(() => ({ symbol, state, dispatch }), [symbol, state])

  return <LiveL1Context.Provider value={value}>{children}</LiveL1Context.Provider>
}
