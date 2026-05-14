/** Shared with Mantine `localStorageColorSchemeManager` for one source of truth. */
export const SHUNYA_THEME_STORAGE_KEY = 'shunya-theme'

export type Theme = 'light' | 'dark'

function systemPrefersDark(): boolean {
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

export function getStoredTheme(): Theme | null {
  try {
    const v = localStorage.getItem(SHUNYA_THEME_STORAGE_KEY)
    if (v === 'light' || v === 'dark') return v
  } catch {
    /* ignore */
  }
  return null
}

export function resolveInitialTheme(): Theme {
  return getStoredTheme() ?? (systemPrefersDark() ? 'dark' : 'light')
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem(SHUNYA_THEME_STORAGE_KEY, theme)
  } catch {
    /* ignore */
  }
}

export function initTheme(): void {
  applyTheme(resolveInitialTheme())
}

export function toggleTheme(current: Theme): Theme {
  const next: Theme = current === 'dark' ? 'light' : 'dark'
  applyTheme(next)
  return next
}
