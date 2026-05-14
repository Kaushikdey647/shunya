# Concepts (quantitative finance)

These pages explain **ideas from asset management and systematic trading**, then relate them to **types and workflows in Shunya**. For API types, function signatures, and configuration knobs, see **[Documentation](../documentation/system-overview.md)**.

| Topic | What you will learn |
|-------|---------------------|
| [Alphas, metrics, and evaluation](alphas-metrics-and-evaluation.md) | What an alpha is economically, how returns arise, common risk and performance metrics (with formulas). |
| [Portfolios, construction, and PMS](portfolios-construction-and-pms.md) | Portfolios as weights, blending alphas, tracking performance, how a PMS sits in the stack. |
| [OMS, EMS, and order routing](oms-ems-and-order-routing.md) | Order vs execution management, components, Alpaca order types, from signal to fills. |
| [Pipeline: alpha to execution](pipeline-alpha-to-execution.md) | End-to-end diagram: research through broker. |
| [Alpha styles and Shunya patterns](alpha-styles-and-shunya-patterns.md) | Momentum, reversal, conditional, fundamental, technical — with Shunya-flavored examples. |
| [Fundamentals for alphas](fundamentals-for-alphas.md) | Each `fun.*` field: meaning and usage. |
| [Alpha design: rank, z-score, gates](alpha-design-rank-zscore-and-gates.md) | When to rank, z-score, use raw values, or gate with `trade_when`. |
