import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type ReactNode,
} from 'react'
import { useQueryClient } from '@tanstack/react-query'
import type { OmsFillOut, OmsParentOrderOut, OmsWsMessage } from '../api/types'
import { omsOrdersStreamWebSocketUrl } from '../api/wsUrl'

const FILL_RING_MAX = 500
const INVALIDATE_DEBOUNCE_MS = 500

export type OmsStreamPhase = 'idle' | 'connecting' | 'live' | 'error'

export type OmsState = {
  orders: Record<string, OmsParentOrderOut>
  fills: OmsFillOut[]
  phase: OmsStreamPhase
  lastError: string | null
  paper: boolean | null
}

const initialState = (): OmsState => ({
  orders: {},
  fills: [],
  phase: 'idle',
  lastError: null,
  paper: null,
})

type Action =
  | { type: 'RESET' }
  | { type: 'SET_PHASE'; phase: OmsStreamPhase }
  | { type: 'SET_ERROR'; message: string }
  | { type: 'HELLO'; paper: boolean }
  | { type: 'ORDER'; order: OmsParentOrderOut }
  | { type: 'FILL'; fill: OmsFillOut }
  | { type: 'WS_CLOSED' }

function pushRing<T>(arr: T[], item: T, max: number): T[] {
  const next = [...arr, item]
  if (next.length <= max) return next
  return next.slice(next.length - max)
}

function reducer(state: OmsState, action: Action): OmsState {
  switch (action.type) {
    case 'RESET':
      return initialState()
    case 'SET_PHASE':
      return { ...state, phase: action.phase }
    case 'SET_ERROR':
      return { ...state, phase: 'error', lastError: action.message }
    case 'HELLO':
      return { ...state, phase: 'live', paper: action.paper, lastError: null }
    case 'ORDER':
      return { ...state, orders: { ...state.orders, [action.order.parent_id]: action.order } }
    case 'FILL':
      return { ...state, fills: pushRing(state.fills, action.fill, FILL_RING_MAX) }
    case 'WS_CLOSED':
      if (state.phase === 'error') return state
      if (state.phase === 'live' || state.phase === 'connecting') {
        return {
          ...state,
          phase: 'error',
          lastError: state.lastError ?? 'Order stream closed unexpectedly. Reconnect to resume live updates.',
        }
      }
      return { ...state, phase: 'idle' }
    default:
      return state
  }
}

function parseOmsMessage(raw: string): OmsWsMessage | null {
  try {
    const msg = JSON.parse(raw) as Record<string, unknown>
    const t = msg.type
    if (t === 'hello' || t === 'order' || t === 'fill' || t === 'error') {
      return msg as OmsWsMessage
    }
    return null
  } catch {
    return null
  }
}

type OmsCtx = {
  state: OmsState
  dispatch: React.Dispatch<Action>
}

const OmsContext = createContext<OmsCtx | null>(null)

export function useOmsStream(): OmsCtx {
  const v = useContext(OmsContext)
  if (!v) throw new Error('useOmsStream must be used within OmsStreamProvider')
  return v
}

type ProviderProps = {
  token: string
  /** When true, open WebSocket and ingest; when false, clear state and disconnect. */
  streamActive: boolean
  children: ReactNode
}

export function OmsStreamProvider({ token, streamActive, children }: ProviderProps) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState)
  const closingFromUiRef = useRef(false)
  const queryClient = useQueryClient()
  const invalidateTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const scheduleInvalidatePositions = () => {
    if (invalidateTimerRef.current != null) return
    invalidateTimerRef.current = setTimeout(() => {
      invalidateTimerRef.current = null
      void queryClient.invalidateQueries({ queryKey: ['oms', 'positions'] })
    }, INVALIDATE_DEBOUNCE_MS)
  }

  useEffect(() => {
    return () => {
      if (invalidateTimerRef.current != null) clearTimeout(invalidateTimerRef.current)
    }
  }, [])

  useEffect(() => {
    if (!streamActive || !token) {
      dispatch({ type: 'RESET' })
      return
    }

    dispatch({ type: 'SET_PHASE', phase: 'connecting' })
    const url = omsOrdersStreamWebSocketUrl(token)
    const ws = new WebSocket(url)

    ws.onmessage = (ev) => {
      const msg = parseOmsMessage(String(ev.data))
      if (!msg) return
      if (msg.type === 'hello') {
        dispatch({ type: 'HELLO', paper: msg.paper })
        return
      }
      if (msg.type === 'error') {
        dispatch({ type: 'SET_ERROR', message: msg.message })
        ws.close()
        return
      }
      if (msg.type === 'order') {
        dispatch({ type: 'ORDER', order: msg.order })
        void queryClient.invalidateQueries({ queryKey: ['oms', 'orders'] })
        return
      }
      if (msg.type === 'fill') {
        dispatch({ type: 'FILL', fill: msg.fill })
        scheduleInvalidatePositions()
      }
    }

    ws.onerror = () => {
      dispatch({ type: 'SET_ERROR', message: 'WebSocket connection error.' })
    }

    ws.onclose = () => {
      if (closingFromUiRef.current) {
        closingFromUiRef.current = false
        return
      }
      dispatch({ type: 'WS_CLOSED' })
    }

    return () => {
      closingFromUiRef.current = true
      ws.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- queryClient is stable
  }, [streamActive, token])

  const value = useMemo(() => ({ state, dispatch }), [state])

  return <OmsContext.Provider value={value}>{children}</OmsContext.Provider>
}
