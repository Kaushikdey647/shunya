# `scripts/` — local bootstrap

Run commands from the **repository root** unless noted.

| Script | Purpose |
|--------|---------|
| **`local-dev-all.sh`** | Docker TimescaleDB, `uv sync`, migrate, API :8000, Vite UI. Options: **`--seed-alphas`**, **`--help`**. |
| **`bootstrap_example_alphas.py`** | Insert bundled example alphas into `api_alphas` (Studio / `GET /alphas`). |
| **`bootstrap_sp100_timescale.py`** | SP100-focused OHLCV + fundamentals + classifications + `^OEX`. |
| **`gapfill_sp100_universe_metadata.py`** | SP100-only **`symbol_classifications`** + **`fundamentals_daily`** (universe overview); skips when DB shows no gaps (`--force` to refresh). |
| **`bootstrap_ts_data.py`** | Full PyTickerSymbols index-union OHLCV ingest + memberships (longer run). |

**Suggested order:** database up → `uv run shunya-timescale migrate` → optional OHLCV bootstrap → **`gapfill_sp100_universe_metadata.py`** if the overview lacks sectors/fundamentals → `bootstrap_example_alphas.py` → start API + UI.

Full narrative, flags, and links: **[Bootstrap scripts (API + UI + database)](https://kaushikdey647.github.io/shunya/how-to/bootstrap-scripts/)** (`docs/how-to/bootstrap-scripts.md` in the repo).
