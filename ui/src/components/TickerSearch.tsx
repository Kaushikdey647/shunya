import { Box, Button, Popover, ScrollAreaAutosize, Stack, Text, TextInput } from '@mantine/core'
import { useCallback, useEffect, useId, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { searchInstruments, instrumentDetailPath } from '../api/endpoints'
import type { InstrumentSearchQuote } from '../api/types'

const DEBOUNCE_MS = 300
const MIN_QUERY = 1

function useDebouncedValue<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return debounced
}

export default function TickerSearch() {
  const navigate = useNavigate()
  const listId = useId()
  const [q, setQ] = useState('')
  const [opened, setOpened] = useState(false)
  const debounced = useDebouncedValue(q.trim(), DEBOUNCE_MS)

  const searchQ = useQuery({
    queryKey: ['instrument-search', debounced],
    queryFn: () => searchInstruments(debounced),
    enabled: debounced.length >= MIN_QUERY,
    staleTime: 30_000,
  })

  const quotes = searchQ.data?.quotes ?? []

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

  const showList = opened && debounced.length >= MIN_QUERY

  return (
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
          type="search"
          placeholder="Symbol or company…"
          autoComplete="off"
          maxLength={64}
          aria-label="Search tickers"
          aria-expanded={showList}
          aria-controls={showList ? listId : undefined}
          aria-autocomplete="list"
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setOpened(true)
          }}
          onFocus={() => setOpened(true)}
          onBlur={() => {
            window.setTimeout(() => setOpened(false), 180)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              goSearch(q)
            }
          }}
        />
      </Popover.Target>
      <Popover.Dropdown p={0} id={listId}>
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
            {quotes.slice(0, 8).map((row: InstrumentSearchQuote) => (
              <Button
                key={`${row.symbol}-${row.exchange ?? ''}`}
                type="button"
                variant="subtle"
                color="gray"
                fullWidth
                justify="flex-start"
                styles={{
                  inner: { justifyContent: 'flex-start' },
                  label: { width: '100%' },
                }}
                role="option"
                onMouseDown={(ev) => ev.preventDefault()}
                onClick={() => goInstrument(row.symbol, row.quote_type)}
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
            <Button
              type="button"
              variant="light"
              color="yellow"
              fullWidth
              radius={0}
              onMouseDown={(ev) => ev.preventDefault()}
              onClick={() => goSearch(q)}
            >
              View all results
            </Button>
          </Stack>
        </ScrollAreaAutosize>
      </Popover.Dropdown>
    </Popover>
  )
}
