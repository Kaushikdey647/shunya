/**
 * Tabular-friendly monospace stack. JetBrains Mono optional via `index.html` Google Fonts.
 * Keep in sync with `--font-mono` in `theme.css`.
 * Used for Mantine `fontFamilyMonospace`, Monaco, and canvas `ctx.font` (via `shunyaCanvasFont`).
 */
export const SHUNYA_FONT_MONO =
  "'JetBrains Mono', 'SF Mono', 'Roboto Mono', ui-monospace, monospace"

/** Canvas 2D `font` string; CSS variables are not available on canvas. */
export function shunyaCanvasFont(px: number): string {
  return `${px}px ${SHUNYA_FONT_MONO}`
}
