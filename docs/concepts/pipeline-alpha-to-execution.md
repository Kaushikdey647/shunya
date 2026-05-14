# Pipeline: alpha to execution

This page ties together **research**, **simulation**, **portfolio construction**, **risk**, and **broker execution** as Shunya models them. For code-level wiring, see [Documentation](../documentation/system-overview.md) and [OMS / EMS](../documentation/oms-ems.md).

## Research and simulation path (library + HTTP backtest)

```mermaid
flowchart TD
  subgraph dataPlane [Data]
    MD[MarketDataProvider]
    FTS[finTs_panel]
    MD --> FTS
  end
  subgraph signalPlane [Signal]
    FS[FinStrat_pass_]
    FTS --> FS
  end
  subgraph simPlane [Simulation]
    FB[FinBT_backtrader]
    FS --> FB
  end
  subgraph outPlane [Outputs]
    MET[Metrics_and_equity_curve]
    FB --> MET
  end
```

- **`finTs`** builds the panel; **`FinStrat`** maps `algorithm(ctx)` to targets each bar; **`FinBT`** applies broker-like **costs** and produces **`results`**.

When you use the **FastAPI** service, the **worker** runs the same broad idea: persisted alpha + config → simulation job → JSON results for the UI.

## Live and paper path (desk + OMS + EMS)

```mermaid
flowchart TD
  subgraph research [Research]
    FTS2[finTs]
    FS2[FinStrat]
    FTS2 --> FS2
  end
  subgraph portfolio [Portfolio]
    PCS[PortfolioConstructionService]
    FS2 --> PCS
  end
  subgraph riskLayer [Risk]
    PRE[PortfolioRiskEngine]
    PCS --> PRE
  end
  subgraph omsEms [Execution_stack]
    OMS[InstitutionalOMS]
    EMS[EMSParentRunner]
    GW[AlpacaBrokerGateway]
    PRE --> OMS
    OMS --> EMS
    EMS --> GW
  end
  subgraph venue [Venue]
    ALP[Alpaca]
    GW --> ALP
  end
```

**`InstitutionalPaperDesk`** wires **PCS → PRE → OMS → EMS** with the Alpaca stream for a **paper** cycle; **`shunya-paper`** exposes a CLI entry point.

## One diagram: both worlds

```mermaid
flowchart LR
  A[Alpha_signal] --> B[Targets]
  B --> C{Risk_vet}
  C -->|pass| D[OMS_parents]
  D --> E[EMS_children]
  E --> F[Broker]
  B -->|research_only| G[FinBT_sim]
  G --> H[Backtest_metrics]
```

- **Left branch:** production-style **risk → OMS → EMS → broker** (live or paper).
- **Right branch:** **FinBT** for historical **what-if** without sending orders.

## Further reading

- [Alphas, metrics, and evaluation](alphas-metrics-and-evaluation.md)
- [Portfolios, construction, and PMS](portfolios-construction-and-pms.md)
