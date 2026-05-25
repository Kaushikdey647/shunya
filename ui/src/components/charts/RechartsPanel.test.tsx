import { act, render } from '@testing-library/react'
import { Line, LineChart, XAxis, YAxis } from 'recharts'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { RechartsPanel } from './RechartsPanel'

describe('RechartsPanel', () => {
  const RO = globalThis.ResizeObserver
  let gbcrSpy: ReturnType<typeof vi.spyOn> | undefined

  beforeEach(() => {
    globalThis.ResizeObserver = vi.fn().mockImplementation(() => ({
      observe: vi.fn(),
      unobserve: vi.fn(),
      disconnect: vi.fn(),
    }))
    // jsdom does not lay out CSS grid; Recharts and this regression test need non-zero geometry.
    gbcrSpy = vi
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockReturnValue({
        x: 0,
        y: 0,
        width: 400,
        height: 200,
        top: 0,
        left: 0,
        right: 400,
        bottom: 200,
        toJSON: () => ({}),
      } as DOMRect)
  })

  afterEach(() => {
    gbcrSpy?.mockRestore()
    globalThis.ResizeObserver = RO
  })

  it('wrapper has positive width inside minmax(0,1fr) grid after layout', async () => {
    const { getByTestId } = render(
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', width: 400 }}>
        <RechartsPanel heightPx={200} dataLength={2}>
          <LineChart data={[{ t: 0, v: 1 }, { t: 1, v: 2 }]}>
            <XAxis type="number" dataKey="t" />
            <YAxis />
            <Line type="stepAfter" dataKey="v" dot={false} isAnimationActive={false} />
          </LineChart>
        </RechartsPanel>
      </div>,
    )

    await act(async () => {
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve())
      })
    })

    const el = getByTestId('recharts-panel')
    expect(el.getBoundingClientRect().width).toBeGreaterThan(10)
    expect(['0', '0px']).toContain((el as HTMLElement).style.minWidth)
  })
})
