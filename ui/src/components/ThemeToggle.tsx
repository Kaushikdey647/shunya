import { Button, useMantineColorScheme } from '@mantine/core'
import { useCallback } from 'react'

export default function ThemeToggle() {
  const { colorScheme, setColorScheme } = useMantineColorScheme()

  const onClick = useCallback(() => {
    setColorScheme(colorScheme === 'dark' ? 'light' : 'dark')
  }, [colorScheme, setColorScheme])

  return (
    <Button
      type="button"
      variant="default"
      size="compact-sm"
      onClick={onClick}
      title={colorScheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
      aria-label={colorScheme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {colorScheme === 'dark' ? 'Light' : 'Dark'}
    </Button>
  )
}
