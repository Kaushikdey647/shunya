# Portfolio construction and risk

## `PortfolioConstructionService`

`PortfolioConstructionService` (`shunya.algorithm.portfolio_manager`) is the preferred entry point for turning signals or blend configs into **USD targets**.

It wraps either:

- **`TargetBlendConfig`** — same semantics as the legacy **`PortfolioManager`**, or
- **`AlphaBlendConfig`** — same semantics as the legacy **`AlphaBlendPortfolioManager`**.

Use **`construct(...)`** to obtain a **`PortfolioConstructionResult`**: targets, requested/active capital, correlation flags, ticker list — useful for risk logging and OMS. **`net_targets`** remains a thin wrapper over `construct(...).targets`.

**`PortfolioManager`** and **`AlphaBlendPortfolioManager`** are legacy facades over the same engines.

## Rolling Sharpe

Rolling Sharpe tracking uses **caller-supplied** simple returns, or an optional **`StrategyReturnFeed`** when `record_returns_from_feed_on_construct=True`.

## `PortfolioRiskEngine`

Optional **pre-trade** checks are available via **`PortfolioRiskEngine`** and **`RiskVetConfig`** / **`RiskVetResult`**. CVX-backed helpers require the **`[risk]`** PyPI extra.

Portfolio math is intentionally **decoupled** from market-data transports and from order routing; wire PCS output into OMS/EMS or the HTTP trade desk as needed.

## Further reading

- [Execution: OMS, EMS, live desk](oms-ems.md) — how targets flow toward brokers.
- [Paper trading cycle (API)](../how-to/paper-cycle-api.md) — one-shot paper cycle via FastAPI.
- [Python API reference](../reference/library.md).
