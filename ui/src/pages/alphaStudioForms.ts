import { z } from 'zod'
import { defaultFinStratConfig } from '../api/defaultConfigs'
import type { FinStratConfig, Neutralization } from '../api/types'

export const alphaDetailsNameSchema = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[a-zA-Z0-9_-]+$/)
  .optional()

export const alphaDetailsSchema = z.object({
  name: alphaDetailsNameSchema,
  description: z.string().max(2048).optional().nullable(),
})

export type AlphaDetailsFormValues = z.infer<typeof alphaDetailsSchema>

export function normalizeNeutralization(value: unknown): Neutralization | undefined {
  if (value === 'group') return 'sector'
  if (value === 'none' || value === 'market' || value === 'sector' || value === 'industry') {
    return value
  }
  return undefined
}

export function finstratFromServer(raw: Record<string, unknown> | null | undefined): FinStratConfig {
  if (!raw || typeof raw !== 'object') {
    return { ...defaultFinStratConfig }
  }
  const merged: FinStratConfig = { ...defaultFinStratConfig, ...(raw as FinStratConfig) }
  const n = normalizeNeutralization(raw.neutralization)
  if (n !== undefined) merged.neutralization = n
  return merged
}
