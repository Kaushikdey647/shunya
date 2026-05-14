# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Trade** navigation group: **Portfolios** (`/portfolios`, `/portfolios/:id`), **Live** (`/live`), **Execution** hub and parent tracer (`/execution`, `/execution/:parentId`), **Risk** command center (`/risk`).
- Client-side **trade desk** state in [`src/lib/tradeDeskStore.ts`](src/lib/tradeDeskStore.ts) with `useTradeDesk` ([`src/hooks/useTradeDesk.ts`](src/hooks/useTradeDesk.ts)); persisted under **`localStorage`** key `shunya_trade_desk_v1` until backend OMS/EMS APIs are available.
- Trade-focused UI building blocks under [`src/components/trade/`](src/components/trade/) (for example add-to-portfolio flow, correlation heatmap, order stream line, distance cell).
- Optional mock metrics helper [`src/lib/tradeDeskMockMetrics.ts`](src/lib/tradeDeskMockMetrics.ts) for demos.

### Changed

- **Home / shell health:** **`GET /health`** parsing and the system health mini-card include **Alpaca** status and latency (or **`skipped`** when the API trade desk is disabled); the header dot tooltip lists Alpaca alongside backend, database, and Yahoo.
- **Theme:** Bloomberg-terminal–style presentation — amber-on-dark accent palette, warm neutral dark scale, monospace stack tuned for data-dense screens ([`src/mantine/theme.ts`](src/mantine/theme.ts), [`src/mantine/cssVariablesResolver.ts`](src/mantine/cssVariablesResolver.ts), [`src/index.css`](src/index.css)).
- **App shell / nav:** Side navigation grouped into **Research**, **Studio**, and **Trade** ([`src/components/SideNav.tsx`](src/components/SideNav.tsx)); command palette and home widgets aligned with the new IA ([`src/components/AppShell.tsx`](src/components/AppShell.tsx), [`src/components/CommandPalette.tsx`](src/components/CommandPalette.tsx), [`src/components/home/*`](src/components/home/)).

### Notes

- Live fills, parent orders, and risk vetting still require **[shunya](https://github.com/Kaushikdey647/shunya)** API routes when they land; the UI does not submit EMS orders today.
