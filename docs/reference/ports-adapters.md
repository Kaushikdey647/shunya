# Ports and adapters

Protocols stay importable without pulling a concrete engine. [ADR 001](../adr/001-architecture-layers.md) is the layering decision. The HTTP worker calls `BacktraderBacktestEngine`, which delegates to `FinBT`.

## Backtest port

::: shunya.ports.backtest_engine
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"

## Cross-section port

::: shunya.ports.cross_section_ops
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"

## Backtrader adapter

::: shunya.adapters.backtrader_engine
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"

## NumPy cross-section

::: shunya.adapters.numpy_cross_section
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"

## JAX cross-section

::: shunya.adapters.jax_cross_section_ops
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"
