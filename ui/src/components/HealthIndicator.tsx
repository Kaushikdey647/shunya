import { Box } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getHealth } from '../api/endpoints'
import { isHealthResponse, type HealthResponse } from '../api/types'

function healthTitle(data: HealthResponse) {
  return `API ${data.status} · backend ${data.backend.status} ${data.backend.latency_ms}ms · db ${data.database.status} ${data.database.latency_ms}ms · yfinance ${data.yfinance.status} ${data.yfinance.latency_ms}ms`
}

function Dot({ color, title, label }: { color: string; title: string; label: string }) {
  return (
    <Box
      component="span"
      title={title}
      aria-label={label}
      style={{
        width: 10,
        height: 10,
        borderRadius: '50%',
        backgroundColor: color,
        flexShrink: 0,
      }}
    />
  )
}

export default function HealthIndicator() {
  const q = useQuery({ queryKey: ['health'], queryFn: getHealth, staleTime: 15_000 })

  if (q.isLoading) {
    return <Dot color="var(--mantine-color-gray-5)" title="API status" label="Checking API" />
  }
  if (q.isError || !q.data || !isHealthResponse(q.data)) {
    return (
      <Dot
        color="var(--mantine-color-red-6)"
        title={q.isError || !q.data ? 'API unreachable' : 'GET /health returned an unexpected body — check VITE_API_BASE'}
        label="API error"
      />
    )
  }
  const { data } = q
  if (data.status === 'error') {
    return <Dot color="var(--mantine-color-red-6)" title={healthTitle(data)} label="API unhealthy" />
  }
  if (data.status === 'degraded') {
    return (
      <Dot color="var(--mantine-color-yellow-6)" title={healthTitle(data)} label="API degraded" />
    )
  }
  return <Dot color="var(--mantine-color-teal-6)" title={healthTitle(data)} label="API healthy" />
}
