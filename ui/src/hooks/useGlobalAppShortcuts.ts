import type { Dispatch, RefObject, SetStateAction } from 'react'
import { useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import type { TickerSearchHandle } from '../components/TickerSearch'
import {
  isEditableTarget,
  isInsideAppTickerSearch,
  isInsideAriaModal,
  isInsideMonaco,
} from '../keyboard/domGuards'
import { FLAT_PRIMARY_NAV, findPrimaryNavIndex } from '../nav/primaryNav'

type Options = {
  commandPaletteOpen: boolean
  setCommandPaletteOpen: Dispatch<SetStateAction<boolean>>
  tickerSearchRef: RefObject<TickerSearchHandle | null>
}

/**
 * Global shortcuts: **⌘/Ctrl+K** palette; **Shift+Space** ticker (avoids macOS Spotlight ⌘Space);
 * **Shift+↑/↓** primary nav. Plain arrows stay free for Monaco and fields.
 */
export function useGlobalAppShortcuts({
  commandPaletteOpen,
  setCommandPaletteOpen,
  tickerSearchRef,
}: Options) {
  const navigate = useNavigate()
  const { pathname } = useLocation()

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const t = e.target

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCommandPaletteOpen((o) => !o)
        return
      }

      if (
        e.shiftKey &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey &&
        (e.key === ' ' || e.code === 'Space')
      ) {
        if (commandPaletteOpen) return
        if (isInsideAriaModal(t)) return
        if (isInsideAppTickerSearch(t)) return
        if (isEditableTarget(t)) return
        if (isInsideMonaco(t)) return
        e.preventDefault()
        tickerSearchRef.current?.focusAndOpen()
        return
      }

      if (
        e.shiftKey &&
        !e.metaKey &&
        !e.ctrlKey &&
        !e.altKey &&
        (e.key === 'ArrowUp' || e.key === 'ArrowDown')
      ) {
        if (commandPaletteOpen) return
        if (isInsideAriaModal(t)) return
        if (isInsideAppTickerSearch(t)) return
        if (isEditableTarget(t)) return
        if (isInsideMonaco(t)) return
        e.preventDefault()
        const n = FLAT_PRIMARY_NAV.length
        const idx = findPrimaryNavIndex(pathname)
        const delta = e.key === 'ArrowDown' ? 1 : -1
        const next = (idx + delta + n) % n
        navigate(FLAT_PRIMARY_NAV[next].to)
      }
    }

    document.addEventListener('keydown', onKeyDown, { capture: true })
    return () => document.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [commandPaletteOpen, navigate, pathname, setCommandPaletteOpen, tickerSearchRef])
}
