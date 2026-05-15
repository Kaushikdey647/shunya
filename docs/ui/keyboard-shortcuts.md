# Keyboard shortcuts (web UI)

The Shunya web UI supports **keyboard-first** navigation: global chords, list movement inside focused regions, and a few **in-context** actions. On Windows/Linux, **Ctrl** is used where macOS uses **⌘** (Command).

**macOS note:** Ticker focus uses **⇧Space** (Shift+Space), not ⌘Space, so it does not conflict with **Spotlight**.

## Global

| Shortcut | Action |
|----------|--------|
| **⌘K** / **Ctrl+K** | Open or close the **command palette** (jump to pages, symbols, alphas, recent backtests). |
| **⇧Space** (Shift+Space) | Focus **ticker search** in the header (no ⌘/Ctrl). Skipped in modals, inside the ticker widget, in editable fields, and in Monaco. |
| **⇧↑** / **⇧↓** (Shift+Arrow, no ⌘/Ctrl/⌥) | Move to the **previous / next** primary sidebar destination (same order as the nav: Research → Studio → Trade). Same skip rules as **⇧Space**. |

Plain **Arrow Up / Down** (without Shift) are **not** used for global nav or table row moves so the **Monaco** editor and normal field behavior stay predictable.

## Command palette

When the palette is open:

- **⇧↑** / **⇧↓** move the **highlighted** row (from the search field).
- **Enter** activates the highlighted row (same as **⌘↵** / **Ctrl+Enter** where you use a primary action chord elsewhere).
- **Esc** closes the palette.

## Ticker search (header)

With the popover open (query length ≥ 1 after debounce):

- **⇧↑** / **⇧↓** move the highlight among instrument rows and **View all results**.
- **Enter** activates the highlighted row, or runs a full search when the list is not used.

## Data tables (Studio list, Backtests, Universes, Portfolios)

1. **Tab** to the table scroll area (it is focusable) or click inside the table.
2. **⇧↑** / **⇧↓** move the **active row** (outline). **Home** / **End** jump to first / last row.
3. **⌘↵** / **Ctrl+Enter** opens the row’s primary destination (same as the main link): alpha workspace, backtest detail, universe detail, or portfolio workspace.

Checkboxes and other inputs keep their usual keys; **⇧**-arrow navigation is ignored while focus is in an **input**, **textarea**, **select**, or **contenteditable** field, or in Monaco.

## Alpha Studio (single alpha workspace)

- **⌘↵** / **Ctrl+Enter** submits **Run backtest** (same as the yellow button tied to the backtest form), when the submit control is enabled. This works even when focus is in the **Monaco** editor. It does **not** fire while another modal is open (for example **Add to portfolio**).

## See also

- [Web application overview](overview.md) — feature map and screenshots.
- [Alpha Studio](studio.md) — workspace layout and backtest flow.
