import { Button } from '@mantine/core'
import { useEffect, useState } from 'react'
import {
  applyTableDensity,
  resolveInitialTableDensity,
  toggleTableDensity,
  type TableDensity,
} from '../density'

export default function DensityToggle() {
  const [density, setDensity] = useState<TableDensity>(() => resolveInitialTableDensity())

  useEffect(() => {
    applyTableDensity(density)
  }, [density])

  return (
    <Button
      type="button"
      variant="default"
      size="compact-sm"
      title={`Table density: ${density}. Click to toggle.`}
      aria-label={`Table density ${density}, toggle comfortable or compact`}
      onClick={() => setDensity((d) => toggleTableDensity(d))}
    >
      {density === 'comfortable' ? 'Comfort' : 'Compact'}
    </Button>
  )
}
