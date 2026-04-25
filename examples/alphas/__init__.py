"""
Example alpha collection using the context-based API.

Each alpha follows ``def alpha(ctx) -> jnp.ndarray`` and can be passed directly
to ``FinStrat``.
"""

from .breakout_20 import alpha as breakout_20
from .mean_reversion_20 import alpha as mean_reversion_20
from .sma20_deviation_rank import alpha as sma20_deviation_rank
from .sma20_deviation_zscore import alpha as sma20_deviation_zscore
from .sma_ratio_20 import alpha as sma_ratio_20

ALL_ALPHAS = {
    "sma_ratio_20": sma_ratio_20,
    "mean_reversion_20": mean_reversion_20,
    "breakout_20": breakout_20,
    "sma20_deviation_rank": sma20_deviation_rank,
    "sma20_deviation_zscore": sma20_deviation_zscore,
}

__all__ = [
    "ALL_ALPHAS",
    "sma_ratio_20",
    "mean_reversion_20",
    "breakout_20",
    "sma20_deviation_rank",
]

