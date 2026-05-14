# Operators and cross-section

Shunya ships JIT-friendly and research-friendly operator modules used inside **`FinStrat`** algorithms (and in standalone NumPy/JAX code).

## `cross_section` (`shunya.algorithm.cross_section`)

Helpers such as **`rank`**, **`zscore`**, **`scale`**, **`sign`**, **`winsorize`**, **`neutralize_market`**, **`neutralize_groups`**.

**Note:** `rank(x)` is **increasing** in `x` (smallest values map toward ~0, largest toward ~1). Use `rank(-x)` to flip direction.

## `logical` (`shunya.algorithm.logical`)

- **`trade_when(condition, alpha, otherwise, exit_condition=...)`**
- **`if_else`**, **`logical_and`**, **`logical_or`**, **`logical_not`**

## `time_series` (`shunya.algorithm.time_series`)

- **`tsdelta`**, **`tsdelay`**, **`tssum`**, **`tsmean`**, **`tsrank`**, **`tszscore`**, **`tsstddev`**
- **`tsregression(y, x, window, lag, retval)`** with `retval in {"error", "a", "b", "estimate"}`
- **`humpdecay`**

## `group_ops` (`shunya.algorithm.group_ops`)

- **`group_rank`**, **`group_zscore`**, **`group_mean`**, **`group_neutralize`**

## Example

```python
import jax.numpy as jnp
from shunya.algorithm import cross_section, group_ops, logical, time_series

signal = cross_section.zscore(jnp.array([1.0, 2.0, 3.0]))
gated = logical.trade_when(signal > 0, signal, 0.0)
```

## API reference

Generated signatures and docstrings: [Python package (`shunya`)](../reference/library.md).
