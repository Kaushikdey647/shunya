import { Container, Stack, type StackProps } from '@mantine/core'
import type { ReactNode } from 'react'

type Props = {
  children: ReactNode
  size?: React.ComponentProps<typeof Container>['size']
} & Omit<StackProps, 'children'>

/** Standard page column: max-width container + vertical stack. */
export default function PageScaffold({
  children,
  size = 'xl',
  gap = 'md',
  px = { base: 'sm', sm: 'md' },
  ...stackProps
}: Props) {
  return (
    <Container size={size} px={px}>
      <Stack gap={gap} {...stackProps}>
        {children}
      </Stack>
    </Container>
  )
}
