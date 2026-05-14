import { Anchor, Card, Divider, Group, Stack, Text, Title } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getMarketHeadlines } from '../../api/endpoints'
import ApiErrorAlert from '../ApiErrorAlert'
import { HeadlineListSkeleton } from './homeDashboardSkeletons'

function fmtTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return iso.slice(0, 16)
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

export default function MarketHeadlines() {
  const q = useQuery({
    queryKey: ['market', 'headlines'],
    queryFn: () => getMarketHeadlines({ limit: 30 }),
    staleTime: 120_000,
  })

  return (
    <Card padding="md" radius="md" withBorder>
      <Stack gap="md">
        <Title order={5}>Market headlines</Title>
        <ApiErrorAlert error={q.error} />
        {q.isLoading && <HeadlineListSkeleton />}
        {!q.isLoading && q.data && (
          <Stack gap={0}>
            {q.data.headlines.length === 0 ? (
              <Text c="dimmed" size="sm">
                No headlines
              </Text>
            ) : (
              q.data.headlines.map((h, i) => (
                <div key={`${h.title}-${i}`}>
                  {i > 0 && <Divider my="sm" />}
                  <Stack gap={4}>
                    {h.link ? (
                      <Anchor href={h.link} target="_blank" rel="noreferrer" size="sm" fw={500} c="yellow">
                        {h.title}
                      </Anchor>
                    ) : (
                      <Text size="sm" fw={500}>
                        {h.title}
                      </Text>
                    )}
                    <Group gap="md" wrap="wrap">
                      <Text size="xs" c="dimmed">
                        {h.publisher ?? '—'}
                      </Text>
                      <Text size="xs" c="dimmed" ff="monospace">
                        {fmtTime(h.published_at)}
                      </Text>
                    </Group>
                  </Stack>
                </div>
              ))
            )}
          </Stack>
        )}
      </Stack>
    </Card>
  )
}
