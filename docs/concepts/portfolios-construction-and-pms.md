# Portfolios, construction, and portfolio management systems

For **`PortfolioConstructionService`**, blend configs, and **`PortfolioRiskEngine`** APIs, see [Portfolio construction and risk](../documentation/portfolio-risk.md).

## What is a portfolio?

A **portfolio** is a set of **weights** \(w_{i,t}\) (or dollar holdings \(h_{i,t}\)) over tradable assets. Weights are usually **normalized** (e.g. sum to 1 long-only, or sum to 0 dollar-neutral long-short with leverage constraint).

- **Economic view:** the portfolio is a **bundle of exposures** to factors, sectors, and idiosyncratic risk.
- **Operational view:** the PMS holds **targets** and reconciles them to **actual positions** vs the custodian or prime broker.

## How portfolios are constructed (conceptually)

1. **Signals** from one or more alphas (scores per name).
2. **Combination** — linear or rules-based blend (e.g. 60% momentum + 40% value), or hierarchical “waterfall”.
3. **Risk and constraints** — sector caps, beta neutrality, max position size, leverage.
4. **Optimization (optional)** — classical **mean–variance** chooses weights to maximize \(\mu^\top w - \frac{\lambda}{2} w^\top \Sigma w\) subject to constraints; many desks use **heuristic** or **risk model** pipelines instead of a single optimizer.

Shunya’s **`PortfolioConstructionService`** is the **library entry point** for turning configured blends into **USD targets** (`construct(...)`), decoupled from data transport and broker routing.

## How alphas are combined

Two broad patterns (both map to Shunya configs; see [documentation](../documentation/portfolio-risk.md)):

- **Target blend** — combine **already desired weights or sleeves** (e.g. “70% index + 30% satellite”), similar to legacy **`PortfolioManager`** semantics.
- **Alpha blend** — combine **multiple alpha signals** into one target set before or after normalization, similar to legacy **`AlphaBlendPortfolioManager`** semantics.

In either case, **correlation** between alphas matters: two momentum variants may not diversify.

## Tracking returns and risk

- **Performance:** cumulative return, volatility, Sharpe/Sortino (see [Alphas, metrics, and evaluation](alphas-metrics-and-evaluation.md)).
- **Ex-ante risk:** factor covariances, stress scenarios (Shunya’s **`PortfolioRiskEngine`** focuses on **pre-trade vetting**; CVX-backed pieces need the **`[risk]`** extra).
- **Rolling metrics:** **`RollingSharpeTracker`** in Shunya uses **caller-supplied** returns or an optional **`StrategyReturnFeed`** so Sharpe can update when you call `construct` in a live loop.

## How portfolio management systems (PMS) work

A **PMS** (often paired with an **OMS**) typically provides:

| Function | Role |
|----------|------|
| **Book of record** | Authoritative positions and cash. |
| **Target management** | Accepts model weights from research or external optimizers. |
| **Compliance / limits** | Blocks or trims illegal or policy-breaking trades. |
| **Rebalance workflow** | Schedules rolls, corporate actions handling, cash management. |
| **Handoff to OMS** | Emits parent orders or deltas vs benchmark.

Shunya does **not** ship a full commercial PMS UI; it gives **library building blocks** (`PortfolioConstructionService`, risk vet, OMS/EMS) and a **FastAPI + React** stack for research and **paper** workflows.

## Further reading

- [OMS, EMS, and order routing](oms-ems-and-order-routing.md)
- [Pipeline: alpha to execution](pipeline-alpha-to-execution.md)
