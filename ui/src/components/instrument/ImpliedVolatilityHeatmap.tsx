import {
  Box,
  ScrollArea,
  Stack,
  Text,
  Tooltip,
  useComputedColorScheme,
  useMantineTheme,
} from '@mantine/core'
import { Fragment, useMemo } from 'react'

function formatExpiryShort(iso: string): string {
  try {
    const d = new Date(`${iso}T12:00:00Z`)
    if (Number.isNaN(d.getTime())) return iso
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

function formatStrike(s: number): string {
  if (Number.isInteger(s) || Math.abs(s - Math.round(s)) < 1e-6) return String(Math.round(s))
  return s.toFixed(2)
}

function minMaxIv(matrix: (number | null)[][]): { min: number; max: number } | null {
  let min = Infinity
  let max = -Infinity
  for (const row of matrix) {
    for (const v of row) {
      if (v != null && Number.isFinite(v)) {
        min = Math.min(min, v)
        max = Math.max(max, v)
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null
  return { min, max }
}

function ivCellColor(
  iv: number | null | undefined,
  range: { min: number; max: number } | null,
  palette: readonly string[],
  emptyBg: string,
): string {
  if (iv == null || !Number.isFinite(iv)) return emptyBg
  if (!range || range.max < range.min) return palette[Math.floor(palette.length / 2)]!
  if (range.max === range.min) return palette[Math.floor(palette.length / 2)]!
  const t = (iv - range.min) / (range.max - range.min)
  const idx = Math.round(t * (palette.length - 1))
  return palette[Math.min(Math.max(idx, 0), palette.length - 1)]!
}

type Props = {
  title: string
  variant: 'calls' | 'puts'
  expirations: string[]
  strikes: number[]
  ivMatrix: (number | null)[][]
}

export default function ImpliedVolatilityHeatmap({
  title,
  variant,
  expirations,
  strikes,
  ivMatrix,
}: Props) {
  const theme = useMantineTheme()
  const colorScheme = useComputedColorScheme('light')
  const range = useMemo(() => minMaxIv(ivMatrix), [ivMatrix])
  const palette = variant === 'calls' ? theme.colors.blue : theme.colors.grape
  const emptyBg = colorScheme === 'dark' ? theme.colors.dark[5] : theme.colors.gray[2]
  const border = colorScheme === 'dark' ? theme.colors.dark[4] : theme.colors.gray[4]
  const headerBg = colorScheme === 'dark' ? theme.colors.dark[7] : theme.colors.gray[0]
  const ivText =
    colorScheme === 'dark' ? theme.colors.dark[0] : theme.colors.gray[9]

  const colTemplate = `minmax(56px,auto) repeat(${expirations.length}, minmax(40px, 1fr))`

  return (
    <Stack gap="xs">
      <Text fw={600} size="sm">
        {title}
      </Text>
      {range && (
        <Text size="xs" c="dimmed">
          Scale {(range.min * 100).toFixed(1)}% – {(range.max * 100).toFixed(1)}% IV (this panel)
        </Text>
      )}
      <ScrollArea type="scroll" offsetScrollbars>
        <Box
          style={{
            display: 'grid',
            gridTemplateColumns: colTemplate,
            gap: 1,
            minWidth: 56 + expirations.length * 40,
            border: `1px solid ${border}`,
            borderRadius: theme.radius.sm,
            overflow: 'hidden',
          }}
        >
          <Box
            style={{
              background: headerBg,
              padding: '6px 8px',
              borderBottom: `1px solid ${border}`,
              borderRight: `1px solid ${border}`,
            }}
          />
          {expirations.map((e) => (
            <Box
              key={e}
              style={{
                background: headerBg,
                padding: '6px 4px',
                borderBottom: `1px solid ${border}`,
                textAlign: 'center',
              }}
            >
              <Text size="xs" fw={500} style={{ lineHeight: 1.25 }}>
                {formatExpiryShort(e)}
              </Text>
            </Box>
          ))}

          {strikes.map((strike, i) => (
            <Fragment key={`${strike}-${i}`}>
              <Box
                style={{
                  background: headerBg,
                  padding: '4px 8px',
                  borderRight: `1px solid ${border}`,
                  borderBottom: i < strikes.length - 1 ? `1px solid ${border}` : undefined,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'flex-end',
                }}
              >
                <Text size="xs" ff="monospace" fw={500}>
                  {formatStrike(strike)}
                </Text>
              </Box>
              {expirations.map((e, j) => {
                const iv = ivMatrix[i]?.[j]
                const bg = ivCellColor(iv, range, palette, emptyBg)
                const label = (
                  <Stack gap={2}>
                    <Text size="xs">Strike {formatStrike(strike)}</Text>
                    <Text size="xs">Expiry {e}</Text>
                    <Text size="xs" fw={600}>
                      {iv != null && Number.isFinite(iv) ? `${(iv * 100).toFixed(1)}% IV` : 'No quote'}
                    </Text>
                  </Stack>
                )
                return (
                  <Tooltip key={`${strike}-${e}`} label={label} withArrow position="top" openDelay={120}>
                    <Box
                      style={{
                        background: bg,
                        minHeight: 32,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        borderBottom: i < strikes.length - 1 ? `1px solid ${border}` : undefined,
                        cursor: 'default',
                      }}
                      title={
                        iv != null && Number.isFinite(iv)
                          ? `${formatStrike(strike)} · ${e}: ${(iv * 100).toFixed(1)}%`
                          : undefined
                      }
                    >
                      {iv != null && Number.isFinite(iv) ? (
                        <Text size="10px" ff="monospace" c={ivText} style={{ opacity: 0.95 }}>
                          {(iv * 100).toFixed(0)}
                        </Text>
                      ) : (
                        <Text size="10px" c="dimmed">
                          —
                        </Text>
                      )}
                    </Box>
                  </Tooltip>
                )
              })}
            </Fragment>
          ))}
        </Box>
      </ScrollArea>
    </Stack>
  )
}
