import { Box, Button, Popover, ScrollAreaAutosize, Stack, Text, TextInput } from '@mantine/core'
import { APP_TICKER_SEARCH_ROOT } from '../keyboard/domGuards'
import { useCallback, useEffect, useId, useImperativeHandle, useMemo, useRef, useState, forwardRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { searchInstruments, instrumentDetailPath } from '../api/endpoints'
import type { InstrumentSearchQuote } from '../api/types'

const DEBOUNCE_MS = 300
const MIN_QUERY = 1

export type TickerSearchHandle = {
  focusAndOpen: () => void
}

function useDebouncedValue<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return debounced
}

const TickerSearch = forwardRef<TickerSearchHandle>(function TickerSearch(_, ref) {
  const navigate = useNavigate()
  const listId = useId()
  const inputRef = useRef<HTMLInputElement>(null)
  const [q, setQ] = useState('')
  const [opened, setOpened] = useState(false)
  const [highlightIndex, setHighlightIndex] = useState(0)
  const debounced = useDebouncedValue(q.trim(), DEBOUNCE_MS)

  const searchQ = useQuery({
    queryKey: ['instrument-search', debounced],
    queryFn: () => searchInstruments(debounced),
    enabled: debounced.length >= MIN_QUERY,
    staleTime: 30_000,
  })

  const quotes = useMemo(() => searchQ.data?.quotes?.slice(0, 8) ?? [], [searchQ.data])
  const showList = opened && debounced.length >= MIN_QUERY
  const selectableCount = quotes.length + (showList ? 1 : 0)

  useImperativeHandle(ref, () => ({
    focusAndOpen() {
      setOpened(true)
      window.setTimeout(() => inputRef.current?.focus(), 0)
    },
  }))

  useEffect(() => {
    if (!showList) return
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset highlight when results change
    setHighlightIndex(0)
  }, [showList, debounced, quotes.length])

  useEffect(() => {
    if (!showList) return
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clamp when count shrinks
    setHighlightIndex((i) => Math.min(i, Math.max(0, selectableCount - 1)))
  }, [showList, selectableCount])

  const goSearch = useCallback(
    (query: string) => {
      const t = query.trim()
      if (!t) return
      setOpened(false)
      navigate(`/search?q=${encodeURIComponent(t)}`)
    },
    [navigate],
  )

  const goInstrument = useCallback(
    (symbol: string, quoteType?: string | null) => {
      setOpened(false)
      setQ('')
      navigate(instrumentDetailPath(symbol, quoteType))
    },
    [navigate],
  )

  const pickHighlighted = useCallback(() => {
    if (!showList || selectableCount <= 0) {
      goSearch(q)
      return
    }
    if (highlightIndex < quotes.length) {
      const row = quotes[highlightIndex]
      goInstrument(row.symbol, row.quote_type)
      return
    }
    goSearch(q)
  }, [showList, selectableCount, highlightIndex, quotes, goInstrument, goSearch, q])

  const onInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (
      showList &&
      e.shiftKey &&
      !e.metaKey &&
      !e.ctrlKey &&
      !e.altKey &&
      (e.key === 'ArrowDown' || e.key === 'ArrowUp')
    ) {
      e.preventDefault()
      const max = Math.max(0, selectableCount - 1)
      if (e.key === 'ArrowDown') setHighlightIndex((i) => Math.min(i + 1, max))
      else setHighlightIndex((i) => Math.max(i - 1, 0))
      return
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      if (showList && selectableCount > 0) {
        pickHighlighted()
      } else {
        goSearch(q)
      }
    }
  }

  return (
    <Box {...{ [APP_TICKER_SEARCH_ROOT]: '' }} style={{ width: '100%', minWidth: 0 }}>
    <Popover
      width="target"
      position="bottom"
      shadow="md"
      opened={showList}
      onChange={setOpened}
      transitionProps={{ transition: 'pop', duration: 200 }}
    >
      <Popover.Target>
        <TextInput
          ref={inputRef}
          type="search"
          placeholder="Symbol or company… (⇧Space)"
          autoComplete="off"
          maxLength={64}
          aria-label="Search tickers"
          aria-expanded={showList}
          aria-controls={showList ? listId : undefined}
          aria-autocomplete="list"
          aria-activedescendant={showList && selectableCount > 0 ? `${listId}-opt-${highlightIndex}` : undefined}
          role="combobox"
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setOpened(true)
          }}
          onFocus={() => setOpened(true)}
          onBlur={() => {
            window.setTimeout(() => setOpened(false), 180)
          }}
          onKeyDown={onInputKeyDown}
        />
      </Popover.Target>
      <Popover.Dropdown p={0} id={listId} role="listbox" aria-label="Ticker matches">
        <ScrollAreaAutosize mah={320} type="auto">
          <Stack gap={0}>
            {searchQ.isLoading && (
              <Box p="sm">
                <Text size="sm" c="dimmed">
                  Searching…
                </Text>
              </Box>
            )}
            {searchQ.isError && (
              <Box p="sm">
                <Text size="sm" c="dimmed">
                  Search failed
                </Text>
              </Box>
            )}
            {!searchQ.isLoading &&
              !searchQ.isError &&
              quotes.length === 0 &&
              debounced.length >= MIN_QUERY && (
                <Box p="sm">
                  <Text size="sm" c="dimmed">
                    No matches
                  </Text>
                </Box>
              )}
            {quotes.map((row: InstrumentSearchQuote, i: number) => (
              <Button
                key={`${row.symbol}-${row.exchange ?? ''}`}
                id={`${listId}-opt-${i}`}
                type="button"
                variant={highlightIndex === i ? 'light' : 'subtle'}
                color="gray"
                fullWidth
                justify="flex-start"
                styles={{
                  inner: { justifyContent: 'flex-start' },
                  label: { width: '100%' },
                }}
                role="option"
                aria-selected={highlightIndex === i}
                onMouseDown={(ev) => ev.preventDefault()}
                onClick={() => goInstrument(row.symbol, row.quote_type)}
                onMouseEnter={() => setHighlightIndex(i)}
              >
                <Stack gap={2} align="flex-start">
                  <Text size="sm" fw={600} ff="monospace" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    {row.symbol}
                  </Text>
                  {(row.shortname || row.longname) && (
                    <Text size="xs" c="dimmed">
                      {row.shortname ?? row.longname}
                    </Text>
                  )}
                </Stack>
              </Button>
            ))}
            {showList && (
              <Button
                id={`${listId}-opt-${quotes.length}`}
                type="button"
                variant={highlightIndex === quotes.length ? 'filled' : 'light'}
                color="yellow"
                fullWidth
                radius={0}
                role="option"
                aria-selected={highlightIndex === quotes.length}
                onMouseDown={(ev) => ev.preventDefault()}
                onClick={() => goSearch(q)}
                onMouseEnter={() => setHighlightIndex(quotes.length)}
              >
                View all results
              </Button>
            )}
          </Stack>
        </ScrollAreaAutosize>
      </Popover.Dropdown>
    </Popover>
    </Box>
  )
})

export default TickerSearch
