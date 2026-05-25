import { Anchor, Button } from '@mantine/core'
import { useEffect, useRef } from 'react'
import { Link } from 'react-router-dom'
import { shunyaCanvasFont } from '../theme/typography'
import LandingRotatingZero from './LandingRotatingZero'
import styles from './LandingPage.module.css'

const DOCS_URL = 'https://kaushikdey647.github.io/shunya/'

const MATRIX_CHARS =
  'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄ0123456789ABCDEFﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ'

/** Bloomberg-leaning amber rain (matches `yellowBrand` mid in theme.ts). */
const RAIN_HEAD = '#fff8e1'
const RAIN_TRAIL = (alpha: number) => `rgba(252, 176, 0, ${alpha})`

export default function LandingPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf = 0
    const fontSize = 14
    let drops: number[] = []

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio ?? 1, 2)
      const w = window.innerWidth
      const h = window.innerHeight
      canvas.width = Math.floor(w * dpr)
      canvas.height = Math.floor(h * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const cols = Math.ceil(w / fontSize) + 1
      drops = Array.from({ length: cols }, () => Math.random() * -50)
    }

    resize()
    window.addEventListener('resize', resize)

    const charAt = (i: number, j: number) =>
      MATRIX_CHARS[(i * 17 + j * 31 + Math.floor(performance.now() / 200)) % MATRIX_CHARS.length]

    const tick = () => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      const cols = drops.length
      ctx.fillStyle = 'rgba(0, 0, 0, 0.14)'
      ctx.fillRect(0, 0, w, h)
      ctx.font = shunyaCanvasFont(fontSize)
      for (let i = 0; i < cols; i++) {
        const x = i * fontSize
        const y = drops[i] * fontSize
        const head = charAt(i, Math.floor(drops[i]))
        ctx.fillStyle = RAIN_HEAD
        ctx.fillText(head, x, y)
        for (let k = 1; k < 8; k++) {
          const yy = y - k * fontSize
          if (yy < 0) continue
          ctx.fillStyle = RAIN_TRAIL(0.1 + (8 - k) * 0.07)
          ctx.fillText(charAt(i, Math.floor(drops[i]) - k), x, yy)
        }
        drops[i] += 0.35 + (i % 5) * 0.06
        if (y > h + 80 || Math.random() < 0.002) drops[i] = Math.random() * -20
      }
      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <section className={styles.root} aria-label="Shunya landing">
      <canvas ref={canvasRef} className={styles.canvas} aria-hidden />
      <div className={styles.scanlines} aria-hidden />
      <div className={styles.vignette} aria-hidden />
      <div className={styles.layout}>
        <div className={styles.columnText}>
          <h1 className={styles.title}>Shunya</h1>
          <p className={styles.sub}>
            <span aria-hidden>{'> '}</span>
            <Anchor
              href={DOCS_URL}
              target="_blank"
              rel="noreferrer"
              c="yellow.4"
              fz="inherit"
              ff="monospace"
              underline="hover"
            >
              read the manual — documentation
            </Anchor>
          </p>
          <p className={styles.tagline}>
            A research and execution desk for systematic equity work: macro context, instrument search, alpha
            authoring, universes and backtests, coverage-aware data views, and trade-desk surfaces — wired to
            your Shunya API when you are ready to run.
          </p>
          <ul className={styles.features}>
            <li>Research dashboard — movers, headlines, watchlist, and recent simulations at a glance.</li>
            <li>Alpha Studio — Monaco editor, lint, optional assist, enqueue backtests from the browser.</li>
            <li>Universes and jobs — saved equity sets, membership, and backtest history with rich detail pages.</li>
            <li>Data integrity — interval coverage and risk versus log total return from stored closes.</li>
            <li>Trade — portfolios, live cockpit, execution tracer, risk command center, account hooks.</li>
          </ul>
          <div className={styles.ctaWrap}>
            <Button component={Link} to="/dashboard" color="yellow" size="md">
              Enter the desk
            </Button>
          </div>
          <p className={styles.footer}>No uplink on this screen — static landing only.</p>
        </div>
        <div className={styles.columnZero}>
          <LandingRotatingZero />
        </div>
      </div>
    </section>
  )
}
