import { useSyncExternalStore } from 'react'
import { getTradeDeskState, subscribeTradeDesk } from '../lib/tradeDeskStore'

export function useTradeDesk() {
  return useSyncExternalStore(subscribeTradeDesk, getTradeDeskState, getTradeDeskState)
}
