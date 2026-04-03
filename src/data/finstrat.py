from __future__ import annotations

from collections.abc import Callable
from typing import List, Tuple, Union

import jax.numpy as jnp
import pandas as pd

from . import indicators
from .fints import finTs


class FinStrat:
    """
    WorldQuant Brain–style engine: bind a :class:`finTs` dataset and a **full-panel**
    JAX ``algorithm``.

    ``algorithm(panel)`` receives ``panel`` shaped ``(n_stocks, n_features)`` and
    returns raw scores ``(n_stocks,)``. Use :mod:`src.data.cross_section` for
    ``rank``, ``zscore`` on column slices (e.g. ``rank(-panel[:, IX_LIVE.LOG_RET])``).

    Dollar notionals from :meth:`pass_` demean scores cross-sectionally, then scale
    so ``sum(abs(orders)) == capital`` when dispersion is non-zero.
    """

    def __init__(
        self,
        fin_ts: finTs,
        algorithm: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:
        self._ts = fin_ts
        self._algorithm = algorithm

    def panel_at(
        self,
        date: Union[str, pd.Timestamp],
        *,
        live: bool = True,
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Cross-section at ``date``: rows follow ``fin_ts.ticker_list`` order; only
        tickers with a row at that date and **no NaN** in the feature block are kept.

        Returns ``(panel, tickers)`` with ``panel`` shaped ``(n, n_features)`` and
        features in ``STRATEGY_FEATURES_LIVE`` (default) or ``STRATEGY_FEATURES``.

        Requires a MultiIndex ``(Ticker, Date)`` dataframe (multi-ticker ``finTs``).
        """
        df = self._ts.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("finTs dataframe is empty")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "panel_at requires multi-ticker finTs data with MultiIndex (Ticker, Date)"
            )
        if tuple(df.index.names) != ("Ticker", "Date"):
            raise ValueError(
                f"Expected index names ('Ticker', 'Date'), got {tuple(df.index.names)!r}"
            )

        dt = pd.Timestamp(date)
        feat_cols = list(
            indicators.STRATEGY_FEATURES_LIVE if live else indicators.STRATEGY_FEATURES
        )
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Dataframe missing feature columns: {missing}")

        rows: List[jnp.ndarray] = []
        tickers_out: List[str] = []
        for t in self._ts.ticker_list:
            key = (t, dt)
            if key not in df.index:
                continue
            row = df.loc[key, feat_cols]
            if isinstance(row, pd.DataFrame):
                continue
            if row.isna().any():
                continue
            rows.append(jnp.asarray(row.to_numpy(dtype=float), dtype=jnp.float32))
            tickers_out.append(t)

        if not rows:
            raise ValueError(
                f"No complete feature rows for date {dt!s} (check ticker_list and NaNs)"
            )
        panel = jnp.stack(rows, axis=0)
        return panel, tickers_out

    def scores(self, indicators: jnp.ndarray) -> jnp.ndarray:
        """
        ``indicators``: ``(n_equities, n_features)``; returns ``(n_equities,)``.
        """
        indicators = jnp.asarray(indicators)
        if indicators.ndim != 2:
            raise ValueError(f"Expected 2D panel, got shape {indicators.shape}")
        return self._algorithm(indicators)

    def pass_(
        self,
        daily_indicators: jnp.ndarray,
        capital: Union[float, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Run the full-panel algorithm, subtract the cross-sectional mean, then scale
        gross notional to ``capital``.

        ``pass_`` replaces the name ``pass`` (Python keyword).
        """
        s = self.scores(daily_indicators)
        s0 = s - jnp.mean(s)
        cap = jnp.asarray(capital, dtype=s0.dtype)
        g = jnp.sum(jnp.abs(s0))
        return jnp.where(g > 0, s0 / g * cap, jnp.zeros_like(s0))
