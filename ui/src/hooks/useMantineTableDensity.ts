import type { MantineSize } from '@mantine/core'
import { useEffect, useState } from 'react'
import { resolveInitialTableDensity, type TableDensity } from '../density'

export type MantineTableDensityProps = {
  verticalSpacing: MantineSize | (string & {}) | number
  horizontalSpacing: MantineSize | (string & {}) | number
  fz: MantineSize | (string & {}) | number
}

function readDensity(): TableDensity {
  const v = document.documentElement.dataset.tableDensity
  return v === 'compact' ? 'compact' : 'comfortable'
}

/** Maps HTML data-table-density to Mantine Table spacing (syncs with DensityToggle). */
export function useMantineTableDensity(): MantineTableDensityProps {
  const [density, setDensity] = useState<TableDensity>(() =>
    typeof document !== 'undefined' ? readDensity() : resolveInitialTableDensity(),
  )

  useEffect(() => {
    const el = document.documentElement
    const sync = () => setDensity(readDensity())
    sync()
    const mo = new MutationObserver(sync)
    mo.observe(el, { attributes: true, attributeFilter: ['data-table-density'] })
    return () => mo.disconnect()
  }, [])

  if (density === 'compact') {
    return { verticalSpacing: 'xs', horizontalSpacing: 'xs', fz: 'xs' }
  }
  return { verticalSpacing: 'sm', horizontalSpacing: 'sm', fz: 'sm' }
}
