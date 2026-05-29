/** Deterministic mock metrics for registry rows until live PM stats are wired. */

function hashStr(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = Math.imul(31, h) + s.charCodeAt(i) | 0
  }
  return h
}

export function mockPortfolioRegistryMetrics(portfolioId: string, slotCount: number) {
  const h = hashStr(portfolioId)
  const base = 0.35 + (Math.abs(h) % 180) / 200
  const liveSharpe = slotCount === 0 ? null : base + slotCount * 0.04
  const netExposure = slotCount === 0 ? 0 : (((h >> 7) % 160) - 80) / 100
  return { liveSharpe, netExposure }
}
