# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Notifications:** **`@mantine/notifications`**, **`NotificationStreamProvider`** (WebSocket **`/notifications/stream`**, reconnect backoff), session notification list, header bell menu, and **`apiFetch`** integration so HTTP errors appear as the same toasts/history.

- **Header market clock:** **`MarketClockProvider`** + **`MarketClockStrip`** use WebSocket **`/settings/market-clock/stream`** (server ticks, reconnect backoff) instead of polling **`GET /settings/market-clock`**.

- **Instrument detail — Live Data:** dedicated **Live Data** tab with **WebSocket** **IEX L1** streaming (Alpaca **quotes** + **trades**, stocks/ETFs) when the API reports **`environment.alpaca_enabled`**; stepped mid/spread (lightweight-charts), Recharts imbalance bubble + histogram, tape; Vite dev proxy sets **`ws: true`** for **`/api`** upgrades.

- **Docker:** **`ui/Dockerfile`** (multi-stage **`npm run build`** with **`VITE_API_BASE=/api`**, **nginx** runtime with **`ui/docker/nginx.conf`** reverse proxy for **`/api/`** → FastAPI). **`ui/.dockerignore`** trims build context.

- **Keyboard:** **⌘/Ctrl+K** command palette (row highlight with **⇧↑⇧↓**, **Enter**); **⇧Space** ticker search; **⇧↑⇧↓** primary nav order; roving **⇧↑⇧↓** / **Home**/**End** and **⌘/Ctrl+Enter** on data tables (Studio hub, Backtests, Universes, Portfolios); **⌘/Ctrl+Enter** in alpha workspace submits **Run backtest** when enabled. Sidebar active route uses **yellow** label styling. Settings page summarizes shortcuts; Monaco host marked for guard logic (`ui/src/keyboard/`).

- **Universes** (`/universe`, `/universe/:id`): list and edit saved equity universes (API **`/universes`**), sector/industry and fundamentals summary; **Alpha Studio** optional **default universe** per alpha; **Backtest** panel can enqueue with **`universe_id`** + benchmark; **Instrument** detail **Add to universe** for equities; **Portfolio** workspace shows union of slot alphas’ default-universe tickers and snapshots **last portfolio universe** for the Execution hub.
- **Trade** navigation group: **Portfolios** (`/portfolios`, `/portfolios/:id`), **Live** (`/live`), **Execution** hub and parent tracer (`/execution`, `/execution/:parentId`), **Risk** command center (`/risk`).
- Client-side **trade desk** state in [`src/lib/tradeDeskStore.ts`](src/lib/tradeDeskStore.ts) with `useTradeDesk` ([`src/hooks/useTradeDesk.ts`](src/hooks/useTradeDesk.ts)); persisted under **`localStorage`** key `shunya_trade_desk_v1` until backend OMS/EMS APIs are available.
- Trade-focused UI building blocks under [`src/components/trade/`](src/components/trade/) (for example add-to-portfolio flow, correlation heatmap, order stream line, distance cell).
- Optional mock metrics helper [`src/lib/tradeDeskMockMetrics.ts`](src/lib/tradeDeskMockMetrics.ts) for demos.

- **Landing (`/`):** static marketing screen (matrix rain, feature list, docs link, **Enter the desk** → **`/dashboard`**); no shell and no API traffic on initial load. **`/dashboard`** holds the former home dashboard (`LandingPage.tsx`, `DashboardPage.tsx`, `App.tsx`).

### Changed

- **Quant research workstation:** IDE dark tokens (**`#0D1117`**, **`#161B22`**, higher text contrast); Alpha Studio **7/5 split**, taller editor, **tab bar** on Monaco; **FinStrat** compact inspector grid; universe risk **lighter chart grids** + PCA tooltip + actionable **empty** states; portfolio **header** live/risk links + **small** blend control + **dense** ledger; instrument tables **right-aligned** numerics; movers/watchlist/backtest table alignment pass. See [`docs/ui/design-tokens.md`](../../docs/ui/design-tokens.md).

- **Desk polish (dark, charts, data):** cool-slate dark tokens and **6px** radius; **JetBrains Mono** stack + Google Fonts link; **Recharts** stepped lines (no misleading **`monotone`** on bucketed series); **MacroStrip** area under line; **L1** bid/ask/mid stepped chart, split bid/ask **bubble** scatters, **Grid** live layout + taller **OFI**; **ApiErrorAlert** `outline`/`compact`; **HealthMiniCard** compact status row; universe sector/industry **dashed + skeleton** empties; instrument **underline Tabs** + **`formatCompact`** on financials/holders; execution tracer **Grid**, slim **Progress**, larger BBO, demo **OFI** strip. **`docs/ui/design-tokens.md`** charts + palette rules.

- **Design system:** earlier **Consolas-first** mono, **Paper**/**Title**/**Tabs** defaults, **`MEDIA_*`** breakpoints, **`PageScaffold`** on dashboard — superseded in part by the polish line above; see **`design-tokens.md`** for current mono stack and chart semantics.

- **Instrument detail — Live Data:** removed the US RTH “market closed” alert and clock-based **Connect** disable; **Connect** always tries the L1 WebSocket when the symbol is eligible. Outside RTH the API sends **`us_rth_closed`** on the instrument socket and **`publish_notification`** to **`/notifications/stream`** (toast + bell). The in-tab red error line is suppressed for **`us_rth_closed`** so messaging is not duplicated.

- **Vite dev proxy:** **`API_PROXY_TARGET`** environment variable (default **`http://127.0.0.1:8000`**) replaces a hard-coded proxy target in **`vite.config.ts`**. **`ws: true`** enables **WebSocket** proxying for **`/api`** (instrument Alpaca L1 stream).

- **Primary nav:** first Research item is **Dashboard** at **`/dashboard`** (was **Home** at **`/`**). Command palette **Go to → Dashboard**; header brand and data-summary crumb link to **`/dashboard`**.

- **Data summary (`/data`, `DataSummaryPage.tsx`):** removed coverage heatmap and sortable ticker table; added **missing-coverage** pie (100% − completeness, top ten symbols in the legend); scatter **y** axis uses **`log_return_pct`** from **`GET /data/dashboard`** (not a client-side transform); risk–return scatter **winsorizes** vol and log return at **P1/P99** for plotted position with raw values in tooltips.

- **`README.md`:** home dashboard screenshot (`docs/dashboard.png`) next to the intro.
- **Home / shell health:** **`GET /health`** parsing and the system health mini-card include **Alpaca** status and latency (or **`skipped`** when the API trade desk is disabled); the header dot tooltip lists Alpaca alongside backend, database, and Yahoo.
- **Theme:** IDE-style dark (**`#0D1117`** / **`#161B22`**) panels, **6px** radius, **JetBrains-first** monospace, **amber/yellow** accents for primary actions (see **Quant research workstation** and [`docs/ui/design-tokens.md`](../../docs/ui/design-tokens.md); [`src/mantine/theme.ts`](src/mantine/theme.ts), [`src/mantine/cssVariablesResolver.ts`](src/mantine/cssVariablesResolver.ts), [`src/index.css`](src/index.css)).
- **App shell / nav:** Side navigation grouped into **Research**, **Studio**, and **Trade** ([`src/components/SideNav.tsx`](src/components/SideNav.tsx)); command palette and home widgets aligned with the new IA ([`src/components/AppShell.tsx`](src/components/AppShell.tsx), [`src/components/CommandPalette.tsx`](src/components/CommandPalette.tsx), [`src/components/home/*`](src/components/home/)).

### Fixed

- **Trade account page:** Alpaca token is edited as a draft and applied explicitly; equity/history/activities queries (and their notifications) run only against the applied token, not on every character typed.

- **Live Data (IEX L1):** stepped mid/spread charts no longer crash the tab with lightweight-charts **`Value is null`** when many BBO updates arrive in the same UTC second (duplicate `time` after second truncation). Series data now keeps one point per second (last quote wins) and drops malformed quote/trade payloads before state ingest.

- **Live Data (IEX L1):** when the browser WebSocket closes **unexpectedly** while the UI still shows **Streaming** or **connecting** (for example Alpaca **connection limit** or a failed **`send_json`** on the server), the store moves to **error** with guidance to wait and press **Connect** again instead of looking healthy with empty charts. Intentional **Disconnect** (effect teardown) does not surface that error.

- **Live Data (IEX L1):** after **20 seconds** in **Streaming** with no quotes and no trades, a yellow **No BBO or trades yet** alert explains possible causes (off-hours, quiet tape, upstream **`alpaca_upstream`** / subscription issues) and points to **`scripts/diag_alpaca_l1_ws.py`**. WebSocket **`subscription`** frames from the API are accepted and ignored by state (no crash); **`error`** with **`code: alpaca_upstream`** uses the same **error** handling path as other server errors.

### Notes

- Live fills, parent orders, and risk vetting still require **[shunya](https://github.com/Kaushikdey647/shunya)** API routes when they land; the UI does not submit EMS orders today.
