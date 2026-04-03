from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pandas as pd

from ..data.fints import finTs
from ..utils import indicators
from . import cross_section

Neutralization = Literal["none", "market", "group"]


@jax.jit
def _jit_scale_gross(s: jnp.ndarray, capital: jnp.ndarray) -> jnp.ndarray:
    """Scale scores to dollar notionals with gross ``sum(|w|) == capital``."""
    cap = jnp.asarray(capital, dtype=s.dtype)
    g = jnp.sum(jnp.abs(s))
    return jnp.where(g > 0, s / g * cap, jnp.zeros_like(s))


@jax.jit
def _jit_clip_rescale_gross(
    notionals: jnp.ndarray, capital: jnp.ndarray, lim: jnp.ndarray
) -> jnp.ndarray:
    """Clip each leg to ±``lim`` then rescale gross to ``capital``."""
    clipped = jnp.clip(notionals, -lim, lim)
    cap = jnp.asarray(capital, dtype=notionals.dtype)
    g = jnp.sum(jnp.abs(clipped))
    return jnp.where(g > 0, clipped / g * cap, jnp.zeros_like(clipped))


class FinStrat:
    """
    WorldQuant BRAIN–style pipeline: raw alpha from a full-panel JAX ``algorithm``,
    then optional temporal **decay**, **truncation** (winsorize), **neutralization**,
    and dollar scaling to book size.

    Typical BRAIN order: decay (smooth alpha through time) → truncate tails →
    neutralize → scale gross notional to ``capital``. Truncation, neutralization
    (except label encoding for groups), and gross scaling use JAX JIT via
    :mod:`src.algorithm.cross_section` and module helpers here.
    """

    def __init__(
        self,
        fin_ts: finTs,
        algorithm: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        decay: float = 0.0,
        neutralization: Neutralization = "market",
        truncation: float = 0.0,
        max_single_weight: Optional[float] = None,
        jit_algorithm: bool = False,
    ) -> None:
        """
        Args:
            fin_ts: Multi-ticker panel source.
            algorithm: ``f(panel) -> (n_stocks,)`` scores.
            decay: Temporal EMA on raw scores per ticker,
                ``smoothed = (1 - decay) * raw + decay * smoothed_prev``.
                ``0`` disables (no memory). Use ``decay`` in ``(0, 1)`` for smoothing
                (larger → slower to move, more weight on history). Requires
                ``tickers`` in :meth:`pass_` when ``decay > 0``.
            neutralization: ``\"market\"`` (demean), ``\"none\"``, or ``\"group\"``
                (demean within groups; pass ``group_ids`` to :meth:`pass_`).
            truncation: Symmetric winsor fraction in ``[0, 0.5)`` (BRAIN truncation).
            max_single_weight: If set, cap each name's absolute notionals to this
                fraction of ``capital``, then rescale gross to ``capital`` when possible.
            jit_algorithm: If True, wrap ``algorithm`` in :func:`jax.jit` so the raw
                score step is XLA-compiled (best for pure JAX alphas with fixed shapes).
        """
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if neutralization not in ("none", "market", "group"):
            raise ValueError(f"neutralization must be 'none', 'market', or 'group', got {neutralization!r}")
        if truncation < 0 or truncation >= 0.5:
            raise ValueError(f"truncation must be in [0, 0.5), got {truncation}")
        if max_single_weight is not None and not (0.0 < max_single_weight <= 1.0):
            raise ValueError(f"max_single_weight must be in (0, 1], got {max_single_weight}")

        self._ts = fin_ts
        self._algorithm = jax.jit(algorithm) if jit_algorithm else algorithm
        self._decay = float(decay)
        self._neutralization = neutralization
        self._truncation = float(truncation)
        self._max_single_weight = max_single_weight
        self._ema_prev: dict[str, float] = {}

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def neutralization(self) -> Neutralization:
        return self._neutralization

    @property
    def truncation(self) -> float:
        return self._truncation

    def reset_pipeline_state(self) -> None:
        """Clear temporal decay state (call before a new backtest or walk-forward run)."""
        self._ema_prev.clear()

    def panel_at(
        self,
        date: Union[str, pd.Timestamp],
        *,
        live: bool = True,
        pasteurize: bool = False,
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Cross-section at ``date``: rows follow ``fin_ts.ticker_list`` order.

        By default keeps only tickers with **no NaN** in the feature block.
        If ``pasteurize`` is True (BRAIN-style pasteurization), impute column NaNs with
        the cross-sectional mean for that date (then still skip rows that are all-NaN).

        Returns ``(panel, tickers)`` with features in ``STRATEGY_FEATURES_LIVE`` or
        ``STRATEGY_FEATURES`` when ``live`` is False.
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
        use_live = live
        feat_cols = list(
            indicators.STRATEGY_FEATURES_LIVE if use_live else indicators.STRATEGY_FEATURES
        )
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Dataframe missing feature columns: {missing}")

        raw_rows: List[pd.Series] = []
        tickers_for_row: List[str] = []
        for t in self._ts.ticker_list:
            key = (t, dt)
            if key not in df.index:
                continue
            row = df.loc[key, feat_cols]
            if isinstance(row, pd.DataFrame):
                continue
            raw_rows.append(row)
            tickers_for_row.append(t)

        if not raw_rows:
            raise ValueError(
                f"No feature rows for date {dt!s} (check ticker_list and index)"
            )

        if pasteurize:
            block = pd.DataFrame([r.astype(float) for r in raw_rows], columns=feat_cols)
            means = block.mean(axis=0, skipna=True)
            block = block.fillna(means)
            rows = [jnp.asarray(block.iloc[i].to_numpy(dtype=float), dtype=jnp.float32) for i in range(len(block))]
            tickers_out = tickers_for_row
        else:
            rows = []
            tickers_out = []
            for t, row in zip(tickers_for_row, raw_rows, strict=True):
                if row.isna().any():
                    continue
                rows.append(
                    jnp.asarray(row.to_numpy(dtype=float), dtype=jnp.float32)
                )
                tickers_out.append(t)

        if not rows:
            raise ValueError(
                f"No complete feature rows for date {dt!s} (check ticker_list and NaNs)"
            )
        panel = jnp.stack(rows, axis=0)
        return panel, tickers_out

    def group_labels_at(
        self,
        date: Union[str, pd.Timestamp],
        tickers: Sequence[str],
        column: str,
    ) -> jnp.ndarray:
        """
        Read ``column`` from ``fin_ts.df`` at ``(ticker, date)`` for each ticker.
        Labels may be strings or ints; used with ``neutralization='group'``.
        """
        df = self._ts.df
        if not isinstance(df, pd.DataFrame) or column not in df.columns:
            raise KeyError(f"Column {column!r} not in fin_ts dataframe")
        dt = pd.Timestamp(date)
        out: List[object] = []
        for t in tickers:
            key = (t, dt)
            if key not in df.index:
                raise KeyError(f"No row for ({t!r}, {dt})")
            v = df.loc[key, column]
            if isinstance(v, pd.Series):
                raise ValueError(f"Duplicate index rows for ({t!r}, {dt})")
            if pd.isna(v):
                raise ValueError(f"Missing group label for {t!r} at {dt}")
            out.append(v)
        return jnp.asarray(out, dtype=object)

    def scores(self, indicators: jnp.ndarray) -> jnp.ndarray:
        """
        ``indicators``: ``(n_equities, n_features)``; returns ``(n_equities,)``.
        """
        indicators = jnp.asarray(indicators)
        if indicators.ndim != 2:
            raise ValueError(f"Expected 2D panel, got shape {indicators.shape}")
        return self._algorithm(indicators)

    def _apply_decay(self, raw: jnp.ndarray, tickers: Sequence[str]) -> jnp.ndarray:
        d = self._decay
        if d <= 0.0:
            return raw
        if len(tickers) != int(raw.shape[0]):
            raise ValueError("tickers length must match number of rows in scores")
        out_list: List[float] = []
        raw_np = [float(x) for x in jnp.asarray(raw).reshape(-1)]
        for name, rv in zip(tickers, raw_np, strict=True):
            prev = self._ema_prev.get(name)
            if prev is None:
                sm = rv
            else:
                sm = (1.0 - d) * rv + d * prev
            self._ema_prev[name] = sm
            out_list.append(sm)
        return jnp.asarray(out_list, dtype=jnp.float32)

    @staticmethod
    def _scale_gross(s: jnp.ndarray, capital: jnp.ndarray) -> jnp.ndarray:
        cap = jnp.asarray(capital, dtype=s.dtype)
        return _jit_scale_gross(jnp.asarray(s, dtype=jnp.float32), cap)

    def _apply_max_weight_rescale(
        self, notionals: jnp.ndarray, capital: jnp.ndarray
    ) -> jnp.ndarray:
        m = self._max_single_weight
        if m is None:
            return notionals
        cap_f = float(jnp.asarray(capital))
        lim = jnp.array(float(m) * cap_f, dtype=notionals.dtype)
        return _jit_clip_rescale_gross(
            notionals, jnp.asarray(capital, dtype=notionals.dtype), lim
        )

    def pass_(
        self,
        daily_indicators: jnp.ndarray,
        capital: Union[float, jnp.ndarray],
        *,
        tickers: Optional[Sequence[str]] = None,
        group_ids: Optional[Union[jnp.ndarray, Sequence[object]]] = None,
    ) -> jnp.ndarray:
        """
        Run the pipeline and return dollar notionals whose gross sum equals ``capital``
        when dispersion is non-zero (after neutralization), unless caps zero everyone
        out.

        Args:
            daily_indicators: ``(n_stocks, n_features)``.
            capital: Target gross booksize.
            tickers: Ticker strings, same order as panel rows. **Required** if
                ``decay > 0``.
            group_ids: Same length as rows; required if ``neutralization == 'group'``.
        """
        raw = self.scores(daily_indicators)

        if self._decay > 0.0:
            if tickers is None:
                raise ValueError("tickers is required when decay > 0")
            s = self._apply_decay(raw, tickers)
        else:
            s = raw

        if self._truncation > 0.0:
            s = cross_section.winsorize(s, self._truncation)

        if self._neutralization == "market":
            s = cross_section.neutralize_market(s)
        elif self._neutralization == "group":
            if group_ids is None:
                raise ValueError("group_ids is required when neutralization='group'")
            gid = jnp.asarray(group_ids)
            if gid.shape[0] != s.shape[0]:
                raise ValueError("group_ids must have same length as scores")
            s = cross_section.neutralize_groups(s, gid)
        # else: none

        cap = jnp.asarray(capital, dtype=s.dtype)
        notionals = self._scale_gross(s, cap)
        return self._apply_max_weight_rescale(notionals, cap)
