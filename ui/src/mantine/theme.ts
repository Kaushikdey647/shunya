import { createTheme, type MantineColorsTuple } from '@mantine/core'

/**
 * Primary accent: Bloomberg-leaning amber (anchor ~#FCB000 at mid shades).
 * Charts still reference `yellow` in useChartPalette — same scale, higher contrast on dark.
 */
const yellowBrand: MantineColorsTuple = [
  '#fff8e1',
  '#ffecb3',
  '#ffe082',
  '#ffd54f',
  '#ffca28',
  '#FCB000',
  '#ffb300',
  '#ffa000',
  '#e65100',
  '#bf360c',
]

/**
 * Warm neutral dark scale (no blue-gray): Mantine indexes 0 = lightest … 9 = darkest.
 * Used for surfaces and borders in dark mode.
 */
const warmDark: MantineColorsTuple = [
  '#f4f2ef',
  '#ddd9d4',
  '#c4bfba',
  '#a9a39d',
  '#8f8983',
  '#6f6a65',
  '#4a4744',
  '#222120',
  '#0c0c0b',
  '#050504',
]

export const shunyaMantineTheme = createTheme({
  primaryColor: 'yellow',
  colors: {
    yellow: yellowBrand,
    dark: warmDark,
  },
  primaryShade: { light: 6, dark: 5 },
  defaultRadius: 'sm',
  fontFamily:
    "system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', Helvetica, Arial, sans-serif",
  fontFamilyMonospace:
    "'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, 'Liberation Mono', monospace",
  defaultGradient: {
    from: 'yellow.4',
    to: 'yellow.7',
    deg: 120,
  },
  other: {
    darkPageBg: '#000000',
    darkPanelBg: '#121212',
    darkBorder: '#333333',
    terminalGreen: '#00FF00',
    terminalRed: '#FF0000',
    terminalCyan: '#00FFFF',
  },
  components: {
    Button: {
      defaultProps: {
        variant: 'filled',
      },
    },
    NavLink: {
      defaultProps: {
        variant: 'subtle',
      },
      styles: {
        root: {
          'body[data-mantine-color-scheme="dark"] &[data-active]': {
            backgroundColor: '#FCB000',
            color: '#000000',
          },
        },
      },
    },
    Card: {
      defaultProps: {
        withBorder: false,
        padding: 'md',
        radius: 'md',
      },
    },
    SegmentedControl: {
      defaultProps: {
        withItemsBorders: false,
        color: 'dark.7',
      },
    },
    Table: {
      defaultProps: {
        striped: true,
        highlightOnHover: true,
        verticalSpacing: 'xs',
        horizontalSpacing: 'sm',
      },
      styles: {
        th: {
          'body[data-mantine-color-scheme="dark"] &': {
            fontSize: 11,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: '#888888',
          },
        },
        td: {
          'body[data-mantine-color-scheme="dark"] &': {
            borderBottom: '1px solid #333333',
          },
        },
      },
    },
    Anchor: {
      defaultProps: {
        c: 'dimmed',
        underline: 'hover',
      },
    },
  },
})
