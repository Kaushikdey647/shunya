/** Wrapper set on Monaco host in `AlphaSourceEditor`. */
export const MONACO_ROOT_ATTR = 'data-app-monaco-root'

export function isInsideMonaco(target: EventTarget | null): boolean {
  if (!target || !(target instanceof Element)) return false
  return Boolean(target.closest(`[${MONACO_ROOT_ATTR}]`))
}

export function isEditableTarget(target: EventTarget | null): boolean {
  if (!target || !(target instanceof Element)) return false
  const el = target as HTMLElement
  const tag = el.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
  if (el.isContentEditable) return true
  if (el.closest('[contenteditable="true"]')) return true
  return Boolean(el.closest('input, textarea, select'))
}

/** Mantine Modal sets `role="dialog"` with `aria-modal="true"`. */
export function isInsideAriaModal(target: EventTarget | null): boolean {
  if (!target || !(target instanceof Element)) return false
  return Boolean(target.closest('[role="dialog"][aria-modal="true"]'))
}

/** Header ticker search host — global Shift+Space / Shift+arrows must not steal keys here. */
export const APP_TICKER_SEARCH_ROOT = 'data-app-ticker-root'

export function isInsideAppTickerSearch(target: EventTarget | null): boolean {
  if (!target || !(target instanceof Element)) return false
  return Boolean(target.closest(`[${APP_TICKER_SEARCH_ROOT}]`))
}
