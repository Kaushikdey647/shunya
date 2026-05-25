# UI design tokens (typography, layout, charts)

This complements the Mantine theme in [`ui/src/mantine/theme.ts`](../../ui/src/mantine/theme.ts) and legacy chart tokens in [`ui/src/theme.css`](../../ui/src/theme.css) (kept in sync with Mantine color scheme via `LegacyThemeSync`).

## Persona: quantitative research workstation

The desk UI targets **equity and quantitative researchers** (long sessions with code, factors, and tables). Prefer:

- **IDE + terminal** ergonomics: high-contrast cool neutrals, minimal decorative chrome, data-forward layouts.
- **Research vs execution:** keep **large filled “trade” CTAs** off primary research surfaces (Studio, universes analytics, portfolio blender). Prefer **compact**, **subtle**, or **outline** actions for live/risk paths unless the screen is explicitly execution-focused.

## Dark palette (IDE-standard cool neutral)

Desk **dark mode** aligns with **GitHub / VS Code–style** surfaces for legibility:

- **Canvas (body):** `#0D1117`
- **Panels / elevated surfaces:** `#161B22`
- **Hover:** `#21262D`
- **Borders:** `rgba(255,255,255,0.12)` (subtle but scannable)
- **Primary text:** `#E6EDF3` (legacy `--text` in `theme.css`)
- **Secondary / labels:** `#8B949E` class (`--text-muted`)

Mantine [`cssVariablesResolver`](../../ui/src/mantine/cssVariablesResolver.ts) and `other.darkPageBg` / `other.darkPanelBg` / `other.darkBorder` mirror these. **Paper** defaults lean **borderless** with a slight background lift; use **`withBorder`** only when a frame is intentional.

**Global radius:** **`6px`** (`defaultRadius`, Card, shared chrome).

## Sans vs monospace

- **Body / UI chrome:** system sans stack from Mantine `fontFamily`.
- **Code and market data:** **`SHUNYA_FONT_MONO`** in [`ui/src/theme/typography.ts`](../../ui/src/theme/typography.ts) — **`JetBrains Mono`**, **`SF Mono`**, **`Roboto Mono`**, then **`ui-monospace`**, **`monospace`**. Optional **JetBrains Mono** is loaded from Google Fonts in [`ui/index.html`](../../ui/index.html) for consistent digit width on Linux.
- **CSS:** **`--font-mono`** in `theme.css` must stay aligned with `SHUNYA_FONT_MONO` for non-Mantine surfaces.
- **Monaco** (`AlphaSourceEditor`) and **`shunyaCanvasFont(px)`** (landing canvas) use the same stack in TS because canvas `ctx.font` cannot read CSS variables reliably.

### Numeric presentation and tables

- **Numeric columns:** `Table.Th` / `Table.Td` with numbers — **`ta="right"`**, **`ff="monospace"`**, **`style={{ fontVariantNumeric: 'tabular-nums' }}`** so decimals and magnitudes align when scanning matrices, backtests, and ledgers.
- **Tickers, alpha ids, UUIDs:** monospace, typically **left**-aligned in the first column(s).
- **Large financial magnitudes:** [`ui/src/lib/formatCompact.ts`](../../ui/src/lib/formatCompact.ts) — **`formatUsdCompact`**, **`formatShareCount`**, **`formatInstrumentFinancialValue`**, **`formatCompactNumber`**.

## Charts (discrete vs continuous)

**Default:** Time series that represent **discrete samples**, **bucketed OHLCV closes**, **fixed-interval metrics**, or **events** must not imply false continuity.

- **Recharts:** prefer **`type="step"`**, **`stepBefore`**, **`stepAfter`**, or **Bar** / justified **Area** over **`monotone`** unless the underlying phenomenon is genuinely smooth (model curves, etc.). Document exceptions next to the chart.
- **lightweight-charts:** use **`LineType.WithSteps`** for piecewise-constant series; histogram / bar for bucketed counts; area fills between **two explicit series** (e.g. bid vs ask) when the metric is bounded.

**Data-ink:** Prefer faint grid lines (`--chart-grid` or `rgba(255,255,255,0.06–0.1)`) on research charts so clusters and loadings remain the focus.

**Exceptions:** Only use smooth lines when continuity is honest; note the reason in code or docs.

## Breakpoints

Defaults match Mantine / [`ui/postcss.config.cjs`](../../ui/postcss.config.cjs) (`sm` = **48em**, `md` = **62em**, …). Shared media-query strings:

| Export            | Use case                                      |
|-------------------|-----------------------------------------------|
| **`MEDIA_BELOW_SM`** | Burger menu, compact header (`< 48em`).     |
| **`MEDIA_MIN_MD`**   | Optional chrome such as header clock strip. |

Landing layout CSS uses **`max-width: 62em`** for stacked columns (aligned with Mantine **`md`**).

## Page layout

Desk routes should use **`PageScaffold`** ([`ui/src/components/PageScaffold.tsx`](../../ui/src/components/PageScaffold.tsx)) — `Container` + vertical **`Stack`** with consistent horizontal padding — unless a full-bleed **`size="fluid"`** layout is intentional (e.g. some trade views).

## Errors in the shell

**`ApiErrorAlert`** supports **`variant="outline"`** and **`compact`** for dashboard-style surfaces so errors read as a **bordered callout** instead of a heavy filled slab in dark mode.

## See also

- [Research area (sidebar)](research.md)
- [Keyboard shortcuts](keyboard-shortcuts.md)
