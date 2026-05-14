import { Anchor, Group, Stack, Table, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { searchInstruments, instrumentDetailPath } from '../api/endpoints'
import ApiErrorAlert from '../components/ApiErrorAlert'
import PageScaffold from '../components/PageScaffold'
import { useMantineTableDensity } from '../hooks/useMantineTableDensity'

export default function SearchResultsPage() {
  const [params] = useSearchParams()
  const q = (params.get('q') ?? '').trim()
  const tableProps = useMantineTableDensity()

  const query = useQuery({
    queryKey: ['instrument-search-page', q],
    queryFn: () => searchInstruments(q),
    enabled: q.length > 0,
  })

  if (!q) {
    return (
      <PageScaffold>
        <Title order={1}>Search</Title>
        <Text c="dimmed" size="sm">
          Enter a symbol or company name in the top search bar.
        </Text>
      </PageScaffold>
    )
  }

  return (
    <PageScaffold>
      <Title order={1}>Results for “{q}”</Title>
      {query.isLoading && (
        <Text c="dimmed" size="sm">
          Loading…
        </Text>
      )}
      <ApiErrorAlert error={query.error} />
      {query.data && (
        <Stack gap="xl">
          <Stack gap="sm">
            <Title order={3}>Instruments</Title>
            {query.data.quotes.length === 0 ? (
              <Text c="dimmed" size="sm">
                No matching instruments.
              </Text>
            ) : (
              <Table.ScrollContainer minWidth={400}>
                <Table {...tableProps}>
                  <Table.Thead>
                    <Table.Tr>
                      <Table.Th>Symbol</Table.Th>
                      <Table.Th>Name</Table.Th>
                      <Table.Th>Exchange</Table.Th>
                      <Table.Th>Type</Table.Th>
                    </Table.Tr>
                  </Table.Thead>
                  <Table.Tbody>
                    {query.data.quotes.map((row) => (
                      <Table.Tr key={`${row.symbol}-${row.exchange ?? ''}`}>
                        <Table.Td>
                          <Anchor
                            component={Link}
                            to={instrumentDetailPath(row.symbol, row.quote_type)}
                            ff="monospace"
                            size="sm"
                          >
                            {row.symbol}
                          </Anchor>
                        </Table.Td>
                        <Table.Td>{row.shortname ?? row.longname ?? '—'}</Table.Td>
                        <Table.Td>
                          <Text size="sm" c="dimmed">
                            {row.exchange ?? '—'}
                          </Text>
                        </Table.Td>
                        <Table.Td>
                          <Text size="sm" c="dimmed">
                            {row.quote_type ?? '—'}
                          </Text>
                        </Table.Td>
                      </Table.Tr>
                    ))}
                  </Table.Tbody>
                </Table>
              </Table.ScrollContainer>
            )}
          </Stack>

          <Stack gap="sm">
            <Title order={3}>News</Title>
            {query.data.news.length === 0 ? (
              <Text c="dimmed" size="sm">
                No news in this search response.
              </Text>
            ) : (
              <Stack gap="sm">
                {query.data.news.map((n, i) => (
                  <div key={`${n.title}-${i}`}>
                    {n.link ? (
                      <Anchor href={n.link} target="_blank" rel="noopener noreferrer" size="sm" c="yellow">
                        {n.title}
                      </Anchor>
                    ) : (
                      <Text size="sm">{n.title}</Text>
                    )}
                    {n.publisher ? (
                      <Text span size="sm" c="dimmed">
                        {' '}
                        — {n.publisher}
                      </Text>
                    ) : null}
                  </div>
                ))}
              </Stack>
            )}
          </Stack>

          {query.data.nav_links.length > 0 && (
            <Stack gap="sm">
              <Title order={3}>Links</Title>
              <Group gap="md">
                {query.data.nav_links.map((nl) => (
                  <Anchor key={nl.url} href={nl.url} target="_blank" rel="noopener noreferrer" size="sm">
                    {nl.title}
                  </Anchor>
                ))}
              </Group>
            </Stack>
          )}
        </Stack>
      )}
    </PageScaffold>
  )
}
