import { defaultCssVariablesResolver, type MantineTheme } from '@mantine/core'

/**
 * Dark mode: IDE-style cool neutrals (GitHub / VS Code–adjacent), high legibility for long sessions.
 * Aligned with `theme.css` legacy tokens. Light mode: Mantine defaults.
 */
const DARK_BODY = '#0d1117'
const DARK_PANEL = '#161b22'
const DARK_PANEL_HOVER = '#21262d'
const DARK_BORDER = 'rgba(255, 255, 255, 0.12)'

export function shunyaCssVariablesResolver(theme: MantineTheme) {
  const base = defaultCssVariablesResolver(theme)
  return {
    ...base,
    dark: {
      ...base.dark,
      '--mantine-color-body': DARK_BODY,
      '--mantine-color-default': DARK_PANEL,
      '--mantine-color-default-hover': DARK_PANEL_HOVER,
      '--mantine-color-default-border': DARK_BORDER,
    },
  }
}
