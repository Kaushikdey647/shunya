import { Box } from '@mantine/core'
import type { ReactNode } from 'react'

function toneFor(kw: string): string {
  switch (kw) {
    case 'PENDING':
      return '#888888'
    case 'WORKING':
    case 'PARTIAL':
      return '#FCB000'
    case 'FILLED':
      return '#00FF00'
    case 'CANCELLED':
      return '#FF0000'
    default:
      return 'inherit'
  }
}

/** Single OMS log line with status keyword highlighting (terminal-style). */
export default function OrderStreamLine({ line }: { line: string }) {
  const nodes: ReactNode[] = []
  let last = 0
  const re = /\b(PENDING|WORKING|PARTIAL|FILLED|CANCELLED)\b/g
  let m: RegExpExecArray | null
  // eslint-disable-next-line no-cond-assign -- exec loop
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) {
      nodes.push(
        <span key={`t-${last}`} style={{ color: 'var(--mantine-color-text)' }}>
          {line.slice(last, m.index)}
        </span>,
      )
    }
    const word = m[1]!
    nodes.push(
      <span key={`k-${m.index}`} style={{ color: toneFor(word), fontWeight: 700 }}>
        {word}
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < line.length) {
    nodes.push(
      <span key={`t-${last}`} style={{ color: 'var(--mantine-color-text)' }}>
        {line.slice(last)}
      </span>,
    )
  }
  return (
    <Box
      component="div"
      ff="monospace"
      fz={11}
      lh={1.45}
      py={2}
      style={{ fontVariantNumeric: 'tabular-nums', wordBreak: 'break-all' }}
    >
      {nodes.length > 0 ? nodes : line}
    </Box>
  )
}
