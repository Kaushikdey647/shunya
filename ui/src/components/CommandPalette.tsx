import {
  Button,
  Divider,
  Modal,
  ScrollAreaAutosize,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listAlphas, listBacktests, searchInstruments, instrumentDetailPath } from '../api/endpoints'
import type { AlphaOut, BacktestJobOut } from '../api/types'

function useDebouncedValue<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return debounced
}

function alphaMatches(a: AlphaOut, needle: string): boolean {
  if (!needle) return true
  const n = needle.toLowerCase()
  return (
    a.name.toLowerCase().includes(n) ||
    (a.description?.toLowerCase().includes(n) ?? false)
  )
}

function jobMatches(j: BacktestJobOut, needle: string): boolean {
  if (!needle) return true
  const n = needle.toLowerCase()
  return (
    j.id.toLowerCase().includes(n) ||
    (j.alpha_name?.toLowerCase().includes(n) ?? false) ||
    j.alpha_id.toLowerCase().includes(n) ||
    (j.index_code?.toLowerCase().includes(n) ?? false)
  )
}

type Props = {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [q, setQ] = useState('')
  const debounced = useDebouncedValue(q.trim(), 280)

  useEffect(() => {
    if (open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- reset query when overlay opens
      setQ('')
      window.setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [open])

  const alphasQ = useQuery({
    queryKey: ['alphas', 'command-palette'],
    queryFn: () => listAlphas({ limit: 500, offset: 0 }),
    enabled: open,
    staleTime: 60_000,
  })

  const backtestsQ = useQuery({
    queryKey: ['backtests', 'command-palette'],
    queryFn: () => listBacktests({ limit: 50, offset: 0 }),
    enabled: open,
    staleTime: 20_000,
  })

  const searchQ = useQuery({
    queryKey: ['instrument-search-palette', debounced],
    queryFn: () => searchInstruments(debounced),
    enabled: open && debounced.length >= 1,
    staleTime: 30_000,
  })

  const needle = q.trim()

  const alphaHits = useMemo(() => {
    const rows = alphasQ.data ?? []
    return rows.filter((a) => alphaMatches(a, needle)).slice(0, 14)
  }, [alphasQ.data, needle])

  const jobHits = useMemo(() => {
    const rows = backtestsQ.data ?? []
    return rows.filter((j) => jobMatches(j, needle)).slice(0, 10)
  }, [backtestsQ.data, needle])

  const quoteHits = searchQ.data?.quotes?.slice(0, 8) ?? []

  const go = (path: string) => {
    onClose()
    navigate(path)
  }

  return (
    <Modal
      opened={open}
      onClose={onClose}
      centered
      size="lg"
      padding="md"
      withCloseButton={false}
      aria-label="Command palette"
      transitionProps={{ transition: 'fade', duration: 220 }}
      overlayProps={{ backgroundOpacity: 0.45, blur: 3 }}
      data-command-palette-root
    >
      <Stack gap="sm">
        <TextInput
          ref={inputRef}
          type="search"
          placeholder="Jump to symbol, alpha, backtest, or page…"
          autoComplete="off"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              e.preventDefault()
              onClose()
            }
          }}
        />
        <Text size="xs" c="dimmed">
          <span className="tabular-nums">⌘K</span> / Ctrl+K · Esc closes
        </Text>
        <Divider />
        <ScrollAreaAutosize mah="min(60vh, 420px)" type="auto">
          <Stack gap="lg">
            {!needle && (
              <Stack gap="xs">
                <Title order={6} tt="uppercase" c="dimmed" fw={600}>
                  Go to
                </Title>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/')}>
                  Home
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/studio')}>
                  <Stack gap={0} align="flex-start">
                    <Text size="sm">Alpha Studio</Text>
                    <Text size="xs" c="dimmed">
                      Edit & run
                    </Text>
                  </Stack>
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/backtests')}>
                  Backtests list
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/data')}>
                  Data summary
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/portfolios')}>
                  Portfolios
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/live')}>
                  Live cockpit
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/execution')}>
                  Execution tracer
                </Button>
                <Button variant="subtle" justify="flex-start" onClick={() => go('/risk')}>
                  Risk center
                </Button>
              </Stack>
            )}

            {debounced.length >= 1 && (
              <Stack gap="xs">
                <Title order={6} tt="uppercase" c="dimmed" fw={600}>
                  Instruments
                </Title>
                {searchQ.isLoading && (
                  <Text size="sm" c="dimmed">
                    Searching…
                  </Text>
                )}
                {searchQ.isError && (
                  <Text size="sm" c="dimmed">
                    Instrument search failed.
                  </Text>
                )}
                {!searchQ.isLoading &&
                  !searchQ.isError &&
                  quoteHits.length === 0 &&
                  debounced.length >= 1 && (
                    <Text size="sm" c="dimmed">
                      No matching instruments.
                    </Text>
                  )}
                {quoteHits.map((row) => (
                  <Button
                    key={`${row.symbol}-${row.exchange ?? ''}`}
                    variant="light"
                    color="yellow"
                    justify="flex-start"
                    onClick={() => go(instrumentDetailPath(row.symbol, row.quote_type))}
                  >
                    <Stack gap={0} align="flex-start">
                      <Text size="sm" ff="monospace" fw={600}>
                        {row.symbol}
                      </Text>
                      <Text size="xs" c="dimmed">
                        {row.shortname ?? row.longname ?? ''}
                      </Text>
                    </Stack>
                  </Button>
                ))}
              </Stack>
            )}

            <Stack gap="xs">
              <Title order={6} tt="uppercase" c="dimmed" fw={600}>
                Alphas
              </Title>
              {alphasQ.isLoading && (
                <Text size="sm" c="dimmed">
                  Loading alphas…
                </Text>
              )}
              {!alphasQ.isLoading && alphaHits.length === 0 && needle && (
                <Text size="sm" c="dimmed">
                  No matching alphas.
                </Text>
              )}
              {alphaHits.map((a) => (
                <Button
                  key={a.id}
                  variant="subtle"
                  justify="flex-start"
                  onClick={() => go(`/studio/${encodeURIComponent(a.id)}`)}
                >
                  <Stack gap={0} align="flex-start">
                    <Text size="sm">{a.name}</Text>
                    <Text size="xs" c="dimmed" ff="monospace">
                      {a.id.slice(0, 8)}…
                    </Text>
                  </Stack>
                </Button>
              ))}
            </Stack>

            <Stack gap="xs">
              <Title order={6} tt="uppercase" c="dimmed" fw={600}>
                Recent backtests
              </Title>
              {backtestsQ.isLoading && (
                <Text size="sm" c="dimmed">
                  Loading…
                </Text>
              )}
              {!backtestsQ.isLoading && jobHits.length === 0 && needle && (
                <Text size="sm" c="dimmed">
                  No matching jobs.
                </Text>
              )}
              {jobHits.map((j) => (
                <Button
                  key={j.id}
                  variant="subtle"
                  justify="flex-start"
                  onClick={() => go(`/backtests/${encodeURIComponent(j.id)}`)}
                >
                  <Stack gap={0} align="flex-start">
                    <Text size="sm" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {j.id.slice(0, 8)}…
                    </Text>
                    <Text size="xs" c="dimmed">
                      {j.alpha_name ?? j.alpha_id} · {j.status}
                    </Text>
                  </Stack>
                </Button>
              ))}
            </Stack>
          </Stack>
        </ScrollAreaAutosize>
      </Stack>
    </Modal>
  )
}
