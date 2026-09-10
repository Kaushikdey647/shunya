import {
  Badge,
  Button,
  Group,
  NumberInput,
  Paper,
  SegmentedControl,
  Stack,
  Text,
  TextInput,
  Title,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { placeOmsOrder } from '../../api/endpoints'
import { ApiError } from '../../api/client'
import type { OmsOrderSide, OmsOrderType } from '../../api/types'

type Props = {
  tradeDeskToken: string
  paper: boolean | null
}

export default function OrderEntryForm({ tradeDeskToken, paper }: Props) {
  const queryClient = useQueryClient()
  const [symbol, setSymbol] = useState('')
  const [side, setSide] = useState<OmsOrderSide>('BUY')
  const [orderType, setOrderType] = useState<OmsOrderType>('MARKET')
  const [quantity, setQuantity] = useState<number | ''>('')
  const [limitPrice, setLimitPrice] = useState<number | ''>('')

  const placeMutation = useMutation({
    mutationFn: () =>
      placeOmsOrder(tradeDeskToken, {
        symbol: symbol.trim().toUpperCase(),
        side,
        quantity: Number(quantity),
        order_type: orderType,
        limit_price: orderType === 'LIMIT' ? Number(limitPrice) : null,
      }),
    onSuccess: (order) => {
      notifications.show({
        color: 'teal',
        title: 'Order submitted',
        message: `${order.side} ${order.quantity_ordered} ${order.symbol} (${order.state})`,
      })
      void queryClient.invalidateQueries({ queryKey: ['oms', 'orders'] })
      setSymbol('')
      setQuantity('')
      setLimitPrice('')
    },
    onError: (err) => {
      const message = err instanceof ApiError ? err.message : 'Order submission failed.'
      notifications.show({ color: 'red', title: 'Order rejected', message })
    },
  })

  const canSubmit =
    tradeDeskToken.length > 0 &&
    symbol.trim().length > 0 &&
    typeof quantity === 'number' &&
    quantity > 0 &&
    (orderType === 'MARKET' || (typeof limitPrice === 'number' && limitPrice > 0))

  return (
    <Paper withBorder p="md" radius="md">
      <Group justify="space-between" mb="sm">
        <Title order={4}>Order entry</Title>
        {paper != null && (
          <Badge color={paper ? 'yellow' : 'red'} variant="light">
            {paper ? 'PAPER' : 'LIVE'}
          </Badge>
        )}
      </Group>
      <Stack gap="sm">
        <TextInput
          label="Symbol"
          placeholder="AAPL"
          value={symbol}
          onChange={(e) => setSymbol(e.currentTarget.value.toUpperCase())}
          disabled={!tradeDeskToken}
        />
        <SegmentedControl
          value={side}
          onChange={(v) => setSide(v as OmsOrderSide)}
          data={[
            { label: 'Buy', value: 'BUY' },
            { label: 'Sell', value: 'SELL' },
          ]}
          disabled={!tradeDeskToken}
        />
        <NumberInput
          label="Quantity (shares)"
          min={1}
          step={1}
          value={quantity}
          onChange={(v) => setQuantity(typeof v === 'number' ? v : '')}
          disabled={!tradeDeskToken}
        />
        <SegmentedControl
          value={orderType}
          onChange={(v) => setOrderType(v as OmsOrderType)}
          data={[
            { label: 'Market', value: 'MARKET' },
            { label: 'Limit', value: 'LIMIT' },
          ]}
          disabled={!tradeDeskToken}
        />
        {orderType === 'LIMIT' && (
          <NumberInput
            label="Limit price"
            min={0.01}
            step={0.01}
            decimalScale={2}
            value={limitPrice}
            onChange={(v) => setLimitPrice(typeof v === 'number' ? v : '')}
            disabled={!tradeDeskToken}
          />
        )}
        <Button
          color={side === 'BUY' ? 'teal' : 'red'}
          disabled={!canSubmit || placeMutation.isPending}
          loading={placeMutation.isPending}
          onClick={() => placeMutation.mutate()}
        >
          {side === 'BUY' ? 'Buy' : 'Sell'} {symbol.trim().toUpperCase() || '—'}
        </Button>
        {!tradeDeskToken && (
          <Text size="xs" c="dimmed">
            Apply a trade-desk token above to enable order entry.
          </Text>
        )}
      </Stack>
    </Paper>
  )
}
