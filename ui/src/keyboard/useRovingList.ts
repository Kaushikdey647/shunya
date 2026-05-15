import { useCallback, useEffect, useRef, useState } from 'react'
import { isEditableTarget, isInsideMonaco } from './domGuards'

export type RovingTableKeyboardOptions = {
  rowCount: number
  onActivate: (index: number) => void
}

/**
 * Keyboard list for data tables: focus the scroll container, then **Shift+↑/↓** (and Home/End)
 * move the active row; ⌘/Ctrl+Enter opens the row primary action.
 * Does not register window listeners — only `onKeyDownCapture` on the host.
 */
export function useRovingTableKeyboard({ rowCount, onActivate }: RovingTableKeyboardOptions) {
  const [activeIndex, setActiveIndex] = useState(0)
  const activeRef = useRef(0)

  useEffect(() => {
    activeRef.current = activeIndex
  }, [activeIndex])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clamp when page size changes
    setActiveIndex((i) => Math.min(i, Math.max(0, Math.max(0, rowCount - 1))))
  }, [rowCount])

  const onKeyDownCapture = useCallback(
    (e: React.KeyboardEvent) => {
      if (rowCount <= 0) return
      const t = e.target
      if (!(t instanceof HTMLElement)) return
      if (isInsideMonaco(t)) return
      if (isEditableTarget(t)) return

      const max = rowCount - 1
      if (e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey && e.key === 'ArrowDown') {
        e.preventDefault()
        setActiveIndex((i) => Math.min(i + 1, max))
        return
      }
      if (e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey && e.key === 'ArrowUp') {
        e.preventDefault()
        setActiveIndex((i) => Math.max(i - 1, 0))
        return
      }
      if (e.key === 'Home') {
        e.preventDefault()
        setActiveIndex(0)
        return
      }
      if (e.key === 'End') {
        e.preventDefault()
        setActiveIndex(max)
        return
      }
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key === 'Enter') {
        e.preventDefault()
        onActivate(activeRef.current)
      }
    },
    [rowCount, onActivate],
  )

  const rowProps = useCallback(
    (index: number) => ({
      'data-kbd-active': activeIndex === index ? 'true' : undefined,
      style:
        activeIndex === index
          ? ({
              outline: '2px solid var(--mantine-color-yellow-filled)',
              outlineOffset: -2,
            } as const)
          : undefined,
      onClick: () => setActiveIndex(index),
    }),
    [activeIndex],
  )

  return {
    activeIndex,
    setActiveIndex,
    scrollContainerProps: {
      tabIndex: 0,
      onKeyDownCapture,
    },
    rowProps,
  }
}
