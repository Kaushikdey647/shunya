/**
 * Mantine default breakpoints (matches `ui/postcss.config.cjs`).
 * Use these for `useMediaQuery` / docs instead of ad-hoc pixel widths.
 */
export const SHUNYA_BREAKPOINT_EM = {
  xs: '36em',
  sm: '48em',
  md: '62em',
  lg: '75em',
  xl: '88em',
} as const

/** Viewport strictly below Mantine `sm` (e.g. burger nav, compact header). */
export const MEDIA_BELOW_SM = `(max-width: 47.99em)`

/** Viewport at or above Mantine `md` (e.g. header clock strip). */
export const MEDIA_MIN_MD = `(min-width: ${SHUNYA_BREAKPOINT_EM.md})`
