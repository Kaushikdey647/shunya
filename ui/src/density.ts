const STORAGE_KEY = 'shunya-table-density'

export type TableDensity = 'comfortable' | 'compact'

export function getStoredTableDensity(): TableDensity | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'comfortable' || v === 'compact') return v
  } catch {
    /* ignore */
  }
  return null
}

export function resolveInitialTableDensity(): TableDensity {
  return getStoredTableDensity() ?? 'comfortable'
}

export function applyTableDensity(density: TableDensity): void {
  document.documentElement.dataset.tableDensity = density
  try {
    localStorage.setItem(STORAGE_KEY, density)
  } catch {
    /* ignore */
  }
}

export function initTableDensity(): void {
  applyTableDensity(resolveInitialTableDensity())
}

export function toggleTableDensity(current: TableDensity): TableDensity {
  const next: TableDensity = current === 'comfortable' ? 'compact' : 'comfortable'
  applyTableDensity(next)
  return next
}
