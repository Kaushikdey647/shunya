import { createTheme, type MantineColorsTuple } from '@mantine/core'
import { SHUNYA_FONT_MONO } from '../theme/typography'

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
 * Cool slate neutral scale: Mantine indexes 0 = lightest … 9 = darkest.
 * Used for surfaces and borders in dark mode (higher contrast than warm brown-gray).
 */
const coolDark: MantineColorsTuple = [
  '#f8f9fa',
  '#f1f3f5',
  '#e9ecef',
  '#dee2e6',
  '#ced4da',
  '#adb5bd',
  '#868e96',
  '#495057',
  '#343a40',
  '#21262d',
]

export const shunyaMantineTheme = createTheme({
  primaryColor: 'yellow',
  colors: {
    yellow: yellowBrand,
    dark: coolDark,
  },
  primaryShade: { light: 6, dark: 5 },
  defaultRadius: '6px',
  fontFamily:
    "system-ui, 'Segoe UI', Roboto, 'Helvetica Neue', Helvetica, Arial, sans-serif",
  fontFamilyMonospace: SHUNYA_FONT_MONO,
  defaultGradient: {
    from: 'yellow.4',
    to: 'yellow.7',
    deg: 120,
  },
  other: {
    darkPageBg: '#0d1117',
    darkPanelBg: '#161b22',
    darkBorder: 'rgba(255, 255, 255, 0.12)',
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
        radius: '6px',
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
            color: '#8b949e',
          },
        },
        td: {
          'body[data-mantine-color-scheme="dark"] &': {
            borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
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
    Paper: {
      defaultProps: {
        radius: '6px',
        withBorder: false,
        shadow: 'none',
      },
    },
    Title: {
      styles: {
        root: {
          fontWeight: 700,
        },
      },
    },
    Tabs: {
      defaultProps: {
        variant: 'outline',
        radius: 'sm',
      },
    },
    TextInput: {
      defaultProps: {
        size: 'sm',
      },
    },
    Select: {
      defaultProps: {
        size: 'sm',
      },
    },
    Code: {
      defaultProps: {
        fz: 'xs',
      },
      styles: {
        root: {
          '&[data-block]': {
            padding: 'var(--mantine-spacing-xs) var(--mantine-spacing-sm)',
          },
        },
      },
    },
  },
})
