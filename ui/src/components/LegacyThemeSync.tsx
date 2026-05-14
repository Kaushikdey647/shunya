import { useComputedColorScheme } from '@mantine/core'
import { useEffect } from 'react'

/** Keeps `html[data-theme]` aligned with Mantine for legacy CSS during migration. */
export default function LegacyThemeSync() {
  const scheme = useComputedColorScheme('light', { getInitialValueInEffect: false })

  useEffect(() => {
    document.documentElement.dataset.theme = scheme
  }, [scheme])

  return null
}
