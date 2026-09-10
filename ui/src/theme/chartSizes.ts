/** Shared recharts/lightweight-charts pane heights (px), so panel density is consistent across pages. */
export const CHART_HEIGHT = {
  sparkline: 52, // inline trend, e.g. MacroStrip cards
  compact: 140, // secondary/paired sub-pane (e.g. spread under mid price)
  compactPrimary: 200, // primary pane in a paired-chart layout (e.g. L1 mid price)
  standard: 280, // default single analytical chart
  tall: 520, // hero chart (e.g. backtest equity curve)
} as const

export const CHART_MIN_HEIGHT = {
  /** Matches `.instrument-chart-shell` in layout.css — canonical value for viewport-relative hero panes. */
  heroViewport: 'min(520px, 55vh)',
} as const
