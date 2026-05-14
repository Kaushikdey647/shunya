# Alpha Studio: AI assist and DSL

Alpha Studio is the Monaco-based editor in the **`ui/`** app. **Lint**, **completion**, and **optional Ollama-backed assist** call the FastAPI service; execution still happens in the **Python worker**, not in the browser.

## Ollama (API side)

On the API process set:

- **`SHUNYA_API_OLLAMA_HOST`** — base URL (for example `http://127.0.0.1:11434`)
- **`SHUNYA_API_OLLAMA_MODEL`** — default model id (optional)

Model choice and HTTP timeout can also be adjusted via **`PATCH /settings/app`** when `DATABASE_URL` and the **`api_runtime_config`** migration are in place (see [api/README.md](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md)).

## Alpha body DSL (editor contract)

The UI wraps your alpha body with imports and a function signature. Inside the body you use the same conceptual surfaces as **`FinStrat`**:

| Root | Meaning |
|------|---------|
| **`ctx.*`** | Panel context: `open`, `high`, `low`, `close`, `adj_volume`, `n_tickers`, `feature_names`, `ctx.feature("name")`, … |
| **`ts.*`** | Time-series helpers: `delay`, `delta`, `sum`, `mean`, `std`, `zscore`, `rank`, `regression`, `humpdecay`, … |
| **`cs.*`** | Cross-section: `rank`, `zscore`, `scale`, `sign`, `winsorize`, `neutralize_market`, `neutralize_groups`, … |
| **`fun.*`** | Fundamental-derived series; names should stay aligned with `shunya/data/fundamentals.py` |
| **`jnp.*`** | Common `jax.numpy` helpers (`array`, `where`, `sqrt`, …) |

### `fun.*` names (maintenance)

The completion catalog is **`ui/src/alphaEditor/alphaDslCatalog.ts`**. **`fun.*`** statement-style fields include Revenue, Net_Income, EPS_Diluted, Operating_Cash_Flow, Free_Cash_Flow, balance-sheet and ratio-style names; daily-style fields include Market_Cap, Enterprise_Value, Trailing_PE, Beta, Dividend_Yield, etc. When adding fundamentals in Python, update **both** `fundamentals.py` and **`alphaDslCatalog.ts`** so hints stay accurate.

## See also

- [FinStrat and FinBT](../documentation/finstrat-finbt.md) (code)
- [Alpha styles and patterns](../concepts/alpha-styles-and-shunya-patterns.md) (finance)
- [Studio](../ui/studio.md)
