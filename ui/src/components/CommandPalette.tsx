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
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
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

type PaletteAction = {
  key: string
  section: string
  onSelect: () => void
  label: ReactNode
}

type Props = {
  open: boolean
  onClose: () => void
}

export default function CommandPalette({ open, onClose }: Props) {
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const [q, setQ] = useState('')
  const [highlightIndex, setHighlightIndex] = useState(0)
  const debounced = useDebouncedValue(q.trim(), 280)

  const go = useCallback(
    (path: string) => {
      onClose()
      navigate(path)
    },
    [navigate, onClose],
  )

  useEffect(() => {
    if (open) {
      /* eslint-disable react-hooks/set-state-in-effect -- reset when overlay opens */
      setQ('')
      setHighlightIndex(0)
      /* eslint-enable react-hooks/set-state-in-effect */
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

  const quoteHits = useMemo(() => searchQ.data?.quotes?.slice(0, 8) ?? [], [searchQ.data])

  const actions = useMemo((): PaletteAction[] => {
    const out: PaletteAction[] = []
    if (!needle) {
      out.push(
        { key: 'goto-home', section: 'Go to', onSelect: () => go('/'), label: 'Home' },
        {
          key: 'goto-studio',
          section: 'Go to',
          onSelect: () => go('/studio'),
          label: (
            <Stack gap={0} align="flex-start">
              <Text size="sm">Alpha Studio</Text>
              <Text size="xs" c="dimmed">
                Edit & run
              </Text>
            </Stack>
          ),
        },
        { key: 'goto-bt', section: 'Go to', onSelect: () => go('/backtests'), label: 'Backtests list' },
        { key: 'goto-data', section: 'Go to', onSelect: () => go('/data'), label: 'Data summary' },
        { key: 'goto-port', section: 'Go to', onSelect: () => go('/portfolios'), label: 'Portfolios' },
        { key: 'goto-live', section: 'Go to', onSelect: () => go('/live'), label: 'Live cockpit' },
        { key: 'goto-ex', section: 'Go to', onSelect: () => go('/execution'), label: 'Execution tracer' },
        { key: 'goto-risk', section: 'Go to', onSelect: () => go('/risk'), label: 'Risk center' },
      )
    }
    if (debounced.length >= 1) {
      for (const row of quoteHits) {
        out.push({
          key: `q-${row.symbol}-${row.exchange ?? ''}`,
          section: 'Instruments',
          onSelect: () => go(instrumentDetailPath(row.symbol, row.quote_type)),
          label: (
            <Stack gap={0} align="flex-start">
              <Text size="sm" ff="monospace" fw={600}>
                {row.symbol}
              </Text>
              <Text size="xs" c="dimmed">
                {row.shortname ?? row.longname ?? ''}
              </Text>
            </Stack>
          ),
        })
      }
    }
    for (const a of alphaHits) {
      out.push({
        key: `a-${a.id}`,
        section: 'Alphas',
        onSelect: () => go(`/studio/${encodeURIComponent(a.id)}`),
        label: (
          <Stack gap={0} align="flex-start">
            <Text size="sm">{a.name}</Text>
            <Text size="xs" c="dimmed" ff="monospace">
              {a.id.slice(0, 8)}…
            </Text>
          </Stack>
        ),
      })
    }
    for (const j of jobHits) {
      out.push({
        key: `j-${j.id}`,
        section: 'Recent backtests',
        onSelect: () => go(`/backtests/${encodeURIComponent(j.id)}`),
        label: (
          <Stack gap={0} align="flex-start">
            <Text size="sm" ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
              {j.id.slice(0, 8)}…
            </Text>
            <Text size="xs" c="dimmed">
              {j.alpha_name ?? j.alpha_id} · {j.status}
            </Text>
          </Stack>
        ),
      })
    }
    return out
  }, [needle, debounced.length, quoteHits, alphaHits, jobHits, go])

  const actionCount = actions.length

  const actionsRef = useRef<PaletteAction[]>([])

  useEffect(() => {
    actionsRef.current = actions
  }, [actions])

  useEffect(() => {
    if (!open) return
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clamp when list changes
    setHighlightIndex((i) => Math.min(i, Math.max(0, actionCount - 1)))
  }, [open, actionCount])

  const highlightRef = useRef(0)
  useEffect(() => {
    highlightRef.current = highlightIndex
  }, [highlightIndex])

  const onInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
      return
    }
    if (e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
      if (actionCount <= 0) return
      e.preventDefault()
      const max = actionCount - 1
      if (e.key === 'ArrowDown') setHighlightIndex((i) => Math.min(i + 1, max))
      else setHighlightIndex((i) => Math.max(i - 1, 0))
      return
    }
    if (e.key === 'Enter') {
      if (actionCount <= 0) return
      e.preventDefault()
      const a = actionsRef.current[highlightRef.current]
      if (a) a.onSelect()
    }
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
          onKeyDown={onInputKeyDown}
          role="combobox"
          aria-activedescendant={
            open && actionCount > 0 ? `cmd-palette-opt-${highlightIndex}` : undefined
          }
          aria-controls="cmd-palette-listbox"
        />
        <Text size="xs" c="dimmed">
          <span className="tabular-nums">⌘K</span> / Ctrl+K · <span className="tabular-nums">⇧Space</span> ticker ·
          ⇧↑⇧↓ · Enter / <span className="tabular-nums">⌘↵</span> · Esc closes
        </Text>
        <Divider />
        <ScrollAreaAutosize mah="min(60vh, 420px)" type="auto" id="cmd-palette-listbox" role="listbox">
          <Stack gap="md">
            {debounced.length >= 1 && (
              <Stack gap="xs">
                {searchQ.isLoading && (
                  <Text size="sm" c="dimmed">
                    Searching instruments…
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
              </Stack>
            )}

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

            {backtestsQ.isLoading && (
              <Text size="sm" c="dimmed">
                Loading backtests…
              </Text>
            )}
            {!backtestsQ.isLoading && jobHits.length === 0 && needle && (
              <Text size="sm" c="dimmed">
                No matching jobs.
              </Text>
            )}

            <Stack gap={4}>
              {actions.map((a, globalIndex) => (
                <Fragment key={a.key}>
                  {(globalIndex === 0 || actions[globalIndex - 1].section !== a.section) && (
                    <Title order={6} tt="uppercase" c="dimmed" fw={600} mt={globalIndex > 0 ? 'md' : 0}>
                      {a.section}
                    </Title>
                  )}
                  <Button
                    id={`cmd-palette-opt-${globalIndex}`}
                    role="option"
                    aria-selected={highlightIndex === globalIndex}
                    variant={highlightIndex === globalIndex ? 'light' : 'subtle'}
                    color={a.section === 'Instruments' ? 'yellow' : undefined}
                    justify="flex-start"
                    onMouseEnter={() => setHighlightIndex(globalIndex)}
                    onClick={() => a.onSelect()}
                  >
                    {typeof a.label === 'string' ? <Text size="sm">{a.label}</Text> : a.label}
                  </Button>
                </Fragment>
              ))}
            </Stack>
          </Stack>
        </ScrollAreaAutosize>
      </Stack>
    </Modal>
  )
}
