/**
 * Compact display for large financial / market numbers (reduces raw digit fatigue).
 */

export function formatCompactNumber(
  n: number | null | undefined,
  opts?: { maximumFractionDigits?: number },
): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: opts?.maximumFractionDigits ?? 2,
  }).format(n)
}

/** USD with compact notation for large magnitudes (e.g. $39.32B). */
export function formatUsdCompact(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(n)
}

/** Share / float counts (no currency symbol). */
export function formatShareCount(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return new Intl.NumberFormat(undefined, {
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(n)
}

/**
 * Statement table cells: use compact currency when very large, else grouped decimals.
 */
export function formatInstrumentFinancialValue(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  if (Math.abs(n) >= 1_000_000) {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency: 'USD',
      notation: 'compact',
      maximumFractionDigits: 2,
    }).format(n)
  }
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
}
