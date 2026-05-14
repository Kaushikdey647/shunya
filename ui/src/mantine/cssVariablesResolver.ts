import { defaultCssVariablesResolver, type MantineTheme } from '@mantine/core'

/**
 * Dark mode: Bloomberg-style terminal surfaces (pure black body, #121212 panels, #333 borders).
 * Light mode: unchanged from Mantine defaults (merged from base.light).
 */
export function shunyaCssVariablesResolver(theme: MantineTheme) {
  const base = defaultCssVariablesResolver(theme)
  return {
    ...base,
    dark: {
      ...base.dark,
      '--mantine-color-body': '#000000',
      '--mantine-color-default': '#121212',
      '--mantine-color-default-hover': '#1a1a1a',
      '--mantine-color-default-border': '#333333',
    },
  }
}
