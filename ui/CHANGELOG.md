# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Keyboard:** **⌘/Ctrl+K** command palette (row highlight with **⇧↑⇧↓**, **Enter**); **⇧Space** ticker search; **⇧↑⇧↓** primary nav order; roving **⇧↑⇧↓** / **Home**/**End** and **⌘/Ctrl+Enter** on data tables (Studio hub, Backtests, Universes, Portfolios); **⌘/Ctrl+Enter** in alpha workspace submits **Run backtest** when enabled. Sidebar active route uses **yellow** label styling. Settings page summarizes shortcuts; Monaco host marked for guard logic (`ui/src/keyboard/`).

- **Universes** (`/universe`, `/universe/:id`): list and edit saved equity universes (API **`/universes`**), sector/industry and fundamentals summary; **Alpha Studio** optional **default universe** per alpha; **Backtest** panel can enqueue with **`universe_id`** + benchmark; **Instrument** detail **Add to universe** for equities; **Portfolio** workspace shows union of slot alphas’ default-universe tickers and snapshots **last portfolio universe** for the Execution hub.
- **Trade** navigation group: **Portfolios** (`/portfolios`, `/portfolios/:id`), **Live** (`/live`), **Execution** hub and parent tracer (`/execution`, `/execution/:parentId`), **Risk** command center (`/risk`).
- Client-side **trade desk** state in [`src/lib/tradeDeskStore.ts`](src/lib/tradeDeskStore.ts) with `useTradeDesk` ([`src/hooks/useTradeDesk.ts`](src/hooks/useTradeDesk.ts)); persisted under **`localStorage`** key `shunya_trade_desk_v1` until backend OMS/EMS APIs are available.
- Trade-focused UI building blocks under [`src/components/trade/`](src/components/trade/) (for example add-to-portfolio flow, correlation heatmap, order stream line, distance cell).
- Optional mock metrics helper [`src/lib/tradeDeskMockMetrics.ts`](src/lib/tradeDeskMockMetrics.ts) for demos.

- **Landing (`/`):** static marketing screen (matrix rain, feature list, docs link, **Enter the desk** → **`/dashboard`**); no shell and no API traffic on initial load. **`/dashboard`** holds the former home dashboard (`LandingPage.tsx`, `DashboardPage.tsx`, `App.tsx`).

### Changed

- **Primary nav:** first Research item is **Dashboard** at **`/dashboard`** (was **Home** at **`/`**). Command palette **Go to → Dashboard**; header brand and data-summary crumb link to **`/dashboard`**.

- **Data summary (`/data`, `DataSummaryPage.tsx`):** removed coverage heatmap and sortable ticker table; added **missing-coverage** pie (100% − completeness, top ten symbols in the legend); scatter **y** axis uses **`log_return_pct`** from **`GET /data/dashboard`** (not a client-side transform); risk–return scatter **winsorizes** vol and log return at **P1/P99** for plotted position with raw values in tooltips.

- **`README.md`:** home dashboard screenshot (`docs/dashboard.png`) next to the intro.
- **Home / shell health:** **`GET /health`** parsing and the system health mini-card include **Alpaca** status and latency (or **`skipped`** when the API trade desk is disabled); the header dot tooltip lists Alpaca alongside backend, database, and Yahoo.
- **Theme:** Bloomberg-terminal–style presentation — amber-on-dark accent palette, warm neutral dark scale, monospace stack tuned for data-dense screens ([`src/mantine/theme.ts`](src/mantine/theme.ts), [`src/mantine/cssVariablesResolver.ts`](src/mantine/cssVariablesResolver.ts), [`src/index.css`](src/index.css)).
- **App shell / nav:** Side navigation grouped into **Research**, **Studio**, and **Trade** ([`src/components/SideNav.tsx`](src/components/SideNav.tsx)); command palette and home widgets aligned with the new IA ([`src/components/AppShell.tsx`](src/components/AppShell.tsx), [`src/components/CommandPalette.tsx`](src/components/CommandPalette.tsx), [`src/components/home/*`](src/components/home/)).

### Notes

- Live fills, parent orders, and risk vetting still require **[shunya](https://github.com/Kaushikdey647/shunya)** API routes when they land; the UI does not submit EMS orders today.
