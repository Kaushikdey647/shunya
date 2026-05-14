import { Stack, Text } from '@mantine/core'
import { useQuery } from '@tanstack/react-query'
import { getInstrumentOptionIvHeatmap } from '../../api/endpoints'
import ApiErrorAlert from '../ApiErrorAlert'
import ImpliedVolatilityHeatmap from './ImpliedVolatilityHeatmap'

type Props = {
  symbol: string
  enabled: boolean
}

export default function InstrumentOptionsChainPanel({ symbol, enabled }: Props) {
  const heatmapQ = useQuery({
    queryKey: ['instrument-option-iv-heatmap', symbol],
    queryFn: () => getInstrumentOptionIvHeatmap(symbol),
    enabled: enabled && symbol.length > 0,
    staleTime: 30_000,
  })

  const data = heatmapQ.data
  const showHeatmaps =
    data?.available &&
    data.strikes.length > 0 &&
    data.expirations.length > 0 &&
    data.iv_calls.length > 0 &&
    data.iv_puts.length > 0

  return (
    <Stack gap="md">
      <ApiErrorAlert error={heatmapQ.error} />
      {heatmapQ.isLoading && (
        <Text size="sm" c="dimmed">
          Loading implied volatility surface…
        </Text>
      )}
      {data && !data.available && (
        <Text size="sm" c="dimmed">
          No option IV surface for this symbol.
        </Text>
      )}
      {showHeatmaps && data && (
        <Stack gap="xl">
          <ImpliedVolatilityHeatmap
            title="Calls IV"
            variant="calls"
            expirations={data.expirations}
            strikes={data.strikes}
            ivMatrix={data.iv_calls}
          />
          <ImpliedVolatilityHeatmap
            title="Puts IV"
            variant="puts"
            expirations={data.expirations}
            strikes={data.strikes}
            ivMatrix={data.iv_puts}
          />
        </Stack>
      )}
      {data?.available && data.strikes.length > 0 && data.expirations.length === 0 && (
        <Text size="sm" c="dimmed">
          No expirations with quotes for the IV grid.
        </Text>
      )}
    </Stack>
  )
}
