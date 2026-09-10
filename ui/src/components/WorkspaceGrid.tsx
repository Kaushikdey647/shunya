import { Grid, type GridProps } from '@mantine/core'
import type { ReactNode } from 'react'

type WorkspaceGridProps = {
  /** Column span (out of 12) for the primary panel at the `lg` breakpoint; secondary gets 12 - primarySpan. */
  primarySpan?: number
  primary: ReactNode
  secondary: ReactNode
} & Omit<GridProps, 'children'>

/** Standard two-panel workspace split (e.g. chart+feed, editor+rail): stacks below `lg`. */
export default function WorkspaceGrid({
  primarySpan = 7,
  primary,
  secondary,
  gap = 'lg',
  ...rest
}: WorkspaceGridProps) {
  return (
    <Grid gap={gap} {...rest}>
      <Grid.Col span={{ base: 12, lg: primarySpan }}>{primary}</Grid.Col>
      <Grid.Col span={{ base: 12, lg: 12 - primarySpan }}>{secondary}</Grid.Col>
    </Grid>
  )
}
