import {
  Button,
  Group,
  Modal,
  NumberInput,
  Select,
  Stack,
  Text,
  TextInput,
} from '@mantine/core'
import { useMemo, useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { createPortfolio, deployAlphaToPortfolio } from '../../lib/tradeDeskStore'
import { useTradeDesk } from '../../hooks/useTradeDesk'

type Props = {
  opened: boolean
  onClose: () => void
  alphaId: string
  alphaName?: string | null
  title?: string
  sourceJobId?: string | null
}

export default function AddToPortfolioModal({
  opened,
  onClose,
  alphaId,
  alphaName,
  title = 'Add to portfolio',
  sourceJobId,
}: Props) {
  const desk = useTradeDesk()
  const [mode, setMode] = useState<'existing' | 'new'>('existing')
  const [newName, setNewName] = useState('Live sleeve')
  const [portfolioId, setPortfolioId] = useState<string | null>(null)
  const [weight, setWeight] = useState<number | string>(1)

  const options = useMemo(
    () =>
      desk.portfolios.map((p) => ({
        value: p.id,
        label: p.name,
      })),
    [desk.portfolios],
  )

  useEffect(() => {
    if (!opened) return
    if (desk.portfolios.length === 0) {
      setMode('new')
    } else {
      setMode('existing')
      setPortfolioId((cur) => cur ?? desk.portfolios[0]!.id)
    }
  }, [opened, desk.portfolios])

  const submit = () => {
    const w = typeof weight === 'number' ? weight : Number(weight)
    const safeW = Number.isFinite(w) && w > 0 ? w : 1
    if (mode === 'new') {
      const row = createPortfolio(newName.trim() || 'New portfolio')
      deployAlphaToPortfolio({
        portfolioId: row.id,
        alphaId,
        alphaName,
        weight: safeW,
        sourceJobId: sourceJobId ?? undefined,
      })
      onClose()
      return
    }
    if (desk.portfolios.length === 0) {
      const row = createPortfolio('Default portfolio')
      deployAlphaToPortfolio({
        portfolioId: row.id,
        alphaId,
        alphaName,
        weight: safeW,
        sourceJobId: sourceJobId ?? undefined,
      })
      onClose()
      return
    }
    if (!portfolioId) return
    deployAlphaToPortfolio({
      portfolioId,
      alphaId,
      alphaName,
      weight: safeW,
      sourceJobId: sourceJobId ?? undefined,
    })
    onClose()
  }

  return (
    <Modal opened={opened} onClose={onClose} title={title} size="md">
      <Stack gap="md">
        {sourceJobId && (
          <Text size="xs" c="dimmed">
            Source job <span className="tabular-nums">{sourceJobId.slice(0, 10)}…</span> — adds a{' '}
            <Text span fw={600}>
              StrategySpec
            </Text>{' '}
            slot (client-side until PM API lands).
          </Text>
        )}
        <Select
          label="Target"
          data={[
            { value: 'existing', label: 'Existing portfolio' },
            { value: 'new', label: 'Create new portfolio' },
          ]}
          value={mode}
          onChange={(v) => v && setMode(v as 'existing' | 'new')}
          disabled={desk.portfolios.length === 0}
        />
        {mode === 'new' ? (
          <TextInput label="Portfolio name" value={newName} onChange={(e) => setNewName(e.currentTarget.value)} />
        ) : (
          <Select
            label="Portfolio"
            placeholder="Pick portfolio"
            data={options}
            value={portfolioId}
            onChange={setPortfolioId}
            searchable
            nothingFoundMessage="No portfolios"
          />
        )}
        <NumberInput
          label="Weight"
          description="Relative sleeve weight before normalization in PortfolioManager."
          min={0.01}
          step={0.05}
          decimalScale={3}
          value={weight}
          onChange={setWeight}
        />
        <Group justify="flex-end">
          <Button variant="default" onClick={onClose}>
            Cancel
          </Button>
          <Button color="yellow" disabled={mode === 'existing' && desk.portfolios.length > 0 && !portfolioId} onClick={submit}>
            Save
          </Button>
        </Group>
        <Button component={Link} to="/portfolios" variant="light" size="compact-xs" onClick={onClose}>
          Open portfolio registry
        </Button>
      </Stack>
    </Modal>
  )
}
