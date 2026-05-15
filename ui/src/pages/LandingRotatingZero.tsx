import { useEffect, useRef } from 'react'
import styles from './LandingRotatingZero.module.css'

/** Luminance ramp (donut-style); drawn in theme amber with per-char alpha. */
const LUM = ` .'\`^",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvcXZYUJCLQ0OZmwqpdbkhao*#MW&8%@$`

const AMBER = { r: 252, g: 176, b: 0 }

type Sample = { ch: string; depth: number; L: number }

function torusNormal(u: number, v: number, R: number, r: number) {
  const cv = Math.cos(v)
  const sv = Math.sin(v)
  const cu = Math.cos(u)
  const su = Math.sin(u)
  const x = (R + r * cv) * cu
  const y = (R + r * cv) * su
  const z = r * sv
  const duX = -(R + r * cv) * su
  const duY = (R + r * cv) * cu
  const duZ = 0
  const dvX = -r * sv * cu
  const dvY = -r * sv * su
  const dvZ = r * cv
  let nx = duY * dvZ - duZ * dvY
  let ny = duZ * dvX - duX * dvZ
  let nz = duX * dvY - duY * dvX
  const len = Math.hypot(nx, ny, nz) || 1
  nx /= len
  ny /= len
  nz /= len
  return { x, y, z, nx, ny, nz }
}

function rotXZ(
  x: number,
  y: number,
  z: number,
  nx: number,
  ny: number,
  nz: number,
  cA: number,
  sA: number,
  cB: number,
  sB: number,
) {
  let y1 = y * cA - z * sA
  let z1 = y * sA + z * cA
  let x1 = x
  let ny1 = ny * cA - nz * sA
  let nz1 = ny * sA + nz * cA
  let nx1 = nx

  const x2 = x1 * cB - y1 * sB
  const y2 = x1 * sB + y1 * cB
  const z2 = z1
  const nx2 = nx1 * cB - ny1 * sB
  const ny2 = nx1 * sB + ny1 * cB
  const nz2 = nz1
  return { x: x2, y: y2, z: z2, nx: nx2, ny: ny2, nz: nz2 }
}

function renderFrame(
  cols: number,
  rows: number,
  A: number,
  B: number,
): Sample[][] {
  const grid: Sample[][] = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({ ch: ' ', depth: -1e9, L: 0 })),
  )

  const R = 1.55
  const r = 0.42
  const K = 4.2
  /** Fit torus inside the grid with padding (avoids top/right clipping at large canvas sizes). */
  const modelSpan = R + r + 0.2
  const pad = 4
  const oozDenom = Math.max(K - modelSpan * 0.92, 0.55)
  const oozSafe = K / oozDenom
  const oozCap = Math.min(Math.max(oozSafe, 0.9), 2.05)
  const sx = (cols / 2 - pad) / (modelSpan * oozCap)
  const sy = (rows / 2 - pad) / (modelSpan * oozCap * 0.52)
  const scale = Math.max(4, Math.min(sx, sy))

  const cA = Math.cos(A)
  const sA = Math.sin(A)
  const cB = Math.cos(B)
  const sB = Math.sin(B)
  const lx = 0.25
  const ly = -0.55
  const lz = -0.8
  const llen = Math.hypot(lx, ly, lz) || 1
  const lxn = lx / llen
  const lyn = ly / llen
  const lzn = lz / llen

  const du = 0.065
  const dv = 0.045
  for (let u = 0; u < Math.PI * 2; u += du) {
    for (let v = 0; v < Math.PI * 2; v += dv) {
      const p = torusNormal(u, v, R, r)
      const q = rotXZ(p.x, p.y, p.z, p.nx, p.ny, p.nz, cA, sA, cB, sB)
      const ooz = K / (K + q.z)
      const xi = Math.floor(cols / 2 + scale * ooz * q.x)
      const yi = Math.floor(rows / 2 + scale * 0.52 * ooz * q.y)
      if (xi < 0 || xi >= cols || yi < 0 || yi >= rows) continue
      let L = q.nx * lxn + q.ny * lyn + q.nz * lzn
      L = 0.5 * (L + 1)
      if (L < 0) L = 0
      if (L > 1) L = 1
      const cell = grid[yi][xi]
      if (ooz > cell.depth) {
        const idx = Math.min(LUM.length - 1, Math.floor(L * (LUM.length - 1)))
        cell.ch = LUM[idx]!
        cell.depth = ooz
        cell.L = L
      }
    }
  }
  return grid
}

export default function LandingRotatingZero() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf = 0
    let t0 = performance.now()

    const paint = (A: number, B: number) => {
      const cssW = canvas.clientWidth
      const cssH = canvas.clientHeight
      if (cssW < 8 || cssH < 8) return
      const dpr = Math.min(window.devicePixelRatio ?? 1, 2)
      canvas.width = Math.floor(cssW * dpr)
      canvas.height = Math.floor(cssH * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const charW = 6.9
      const charH = 12.5
      const cols = Math.max(16, Math.floor(cssW / charW))
      const rows = Math.max(10, Math.floor(cssH / charH))
      const grid = renderFrame(cols, rows, A, B)
      ctx.clearRect(0, 0, cssW, cssH)
      ctx.font = `${Math.floor(charH * 0.92)}px Consolas, "Liberation Mono", monospace`
      ctx.textBaseline = 'top'
      const gridW = cols * charW
      const gridH = rows * charH
      const xOff = Math.max(0, (cssW - gridW) * 0.5)
      const yOff = Math.max(0, (cssH - gridH) * 0.5)
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const { ch, L } = grid[y]![x]!
          if (ch === ' ') continue
          const a = 0.22 + L * 0.78
          ctx.fillStyle = `rgba(${AMBER.r},${AMBER.g},${AMBER.b},${a})`
          ctx.fillText(ch, xOff + x * charW, yOff + y * charH)
        }
      }
    }

    const loop = (now: number) => {
      const t = (now - t0) / 1000
      const A = t * 0.85
      const B = t * 0.55
      paint(A, B)
      raf = requestAnimationFrame(loop)
    }

    const ro = new ResizeObserver(() => {
      if (reduced) paint(0.7, 0.4)
    })
    ro.observe(canvas)

    if (reduced) {
      paint(0.7, 0.4)
    } else {
      raf = requestAnimationFrame(loop)
    }

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [])

  return (
    <div className={styles.wrap} aria-hidden>
      <canvas ref={canvasRef} className={styles.canvas} />
    </div>
  )
}
