import { useState } from 'react'

const SESSION_KEY = 'shunya_trade_desk_token_v1'

function readSession(): string {
  if (typeof window === 'undefined') return ''
  try {
    return window.sessionStorage.getItem(SESSION_KEY) ?? ''
  } catch {
    return ''
  }
}

function writeSession(token: string): void {
  if (typeof window === 'undefined') return
  try {
    if (token) window.sessionStorage.setItem(SESSION_KEY, token)
    else window.sessionStorage.removeItem(SESSION_KEY)
  } catch {
    /* ignore quota / privacy-mode errors */
  }
}

/**
 * Draft/applied `X-Shunya-Trade-Desk-Token` pattern shared by trade-desk pages.
 *
 * Requests only fire against `appliedToken`, set via {@link applyToken}, so the token
 * isn't sent on every keystroke. Persisted in `sessionStorage` (cleared when the tab closes).
 */
export function useTradeDeskToken() {
  const [draftToken, setDraftToken] = useState(() => readSession())
  const [appliedToken, setAppliedToken] = useState(() => readSession())

  const applyToken = () => {
    const t = draftToken.trim()
    setAppliedToken(t)
    writeSession(t)
  }

  return { draftToken, setDraftToken, appliedToken, applyToken }
}
