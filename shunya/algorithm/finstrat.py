from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, List, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..data.fints import finTs
from ..data.timeframes import bar_spec_is_intraday, normalize_bar_timestamp
from ..utils import indicators
from .alpha_context import AlphaContext, AlphaSeries
from . import cross_section

Neutralization = Literal["none", "market", "group"]
DecayMode = Literal["ema", "linear"]
NanPolicy = Literal["strict", "zero_fill"]
TemporalMode = Literal["bar_step", "elapsed_trading_time"]
STANDARD_GROUP_COLUMNS: Tuple[str, ...] = ("Sector", "Industry", "SubIndustry")


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
    WorldQuant BRAIN-style pipeline: raw alpha from a context-based ``algorithm``,
    then optional temporal **decay**, **truncation** (winsorize), **neutralization**,
    and dollar scaling to book size.

    Typical BRAIN order: decay (smooth alpha through time) → truncate tails →
    neutralize → scale gross notional to ``capital``. Truncation, neutralization
    (except label encoding for groups), and gross scaling use JAX JIT via
    :mod:`shunya.algorithm.cross_section` and module helpers here.
    """

    def __init__(
        self,
        fin_ts: finTs,
        algorithm: Callable[[AlphaContext], Union[AlphaSeries, jnp.ndarray]],
        *,
        decay_mode: DecayMode = "ema",
        decay: float = 0.0,
        decay_window: int = 1,
        signal_delay: int = 0,
        intraday_session_isolated_lag: bool = False,
        nan_policy: NanPolicy = "strict",
        temporal_mode: TemporalMode = "bar_step",
        neutralization: Neutralization = "market",
        truncation: float = 0.0,
        max_single_weight: Optional[float] = None,
        jit_algorithm: bool = False,
        panel_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            fin_ts: Multi-ticker panel source.
            algorithm: ``f(ctx) -> (n_stocks,)`` scores where ``ctx`` is
                :class:`~shunya.algorithm.alpha_context.AlphaContext`.
            decay_mode: ``\"ema\"`` or ``\"linear`` (BRAIN integer window).
            decay: EMA coefficient in ``[0, 1)`` when ``decay_mode=\"ema\"``.
            decay_window: BRAIN linear window ``n \\ge 1`` when ``decay_mode=\"linear\"``.
            signal_delay: Signal lag in **bars** on the panel calendar (``1`` = prior bar;
                for daily panels, typically the prior trading day).
            intraday_session_isolated_lag: When True and ``fin_ts.bar_spec`` is intraday,
                forbids ``signal_delay`` lookback across overnight session boundaries (raises
                on the session open for ``signal_delay >= 1``), and :class:`FinBT` resets
                decay state at each new session. ``signal_delay=0`` still trades the open bar
                using same-bar features.
            nan_policy: ``strict`` or ``zero_fill`` for non-finite raw scores.
            neutralization: ``\"market\"``, ``\"none\"``, or ``\"group\"``.
            truncation: Symmetric winsor fraction in ``[0, 0.5)``.
            max_single_weight: Optional per-name gross cap as fraction of ``capital``.
            jit_algorithm: Deprecated for context-based API; must be ``False``.
            panel_columns: Optional column subset for :meth:`panel_at`.
        """
        if decay_mode not in ("ema", "linear"):
            raise ValueError(f"decay_mode must be 'ema' or 'linear', got {decay_mode!r}")
        if decay_mode == "ema":
            if not (0.0 <= decay < 1.0):
                raise ValueError(f"decay must be in [0, 1) for ema mode, got {decay}")
        else:
            if decay_window < 1:
                raise ValueError(f"decay_window must be >= 1 for linear mode, got {decay_window}")
        if signal_delay < 0:
            raise ValueError(f"signal_delay must be non-negative, got {signal_delay}")
        if nan_policy not in ("strict", "zero_fill"):
            raise ValueError(f"nan_policy must be 'strict' or 'zero_fill', got {nan_policy!r}")
        if temporal_mode not in ("bar_step", "elapsed_trading_time"):
            raise ValueError(
                f"temporal_mode must be 'bar_step' or 'elapsed_trading_time', got {temporal_mode!r}"
            )
        if neutralization not in ("none", "market", "group"):
            raise ValueError(f"neutralization must be 'none', 'market', or 'group', got {neutralization!r}")
        if truncation < 0 or truncation >= 0.5:
            raise ValueError(f"truncation must be in [0, 0.5), got {truncation}")
        if max_single_weight is not None and not (0.0 < max_single_weight <= 1.0):
            raise ValueError(f"max_single_weight must be in (0, 1], got {max_single_weight}")
        if not callable(algorithm):
            raise TypeError("algorithm must be callable and accept AlphaContext")
        if jit_algorithm:
            raise ValueError(
                "jit_algorithm is not supported in context-based alpha mode; "
                "JAX processing is managed internally by FinStrat."
            )

        self._ts = fin_ts
        self._algorithm = algorithm
        self._decay_mode: DecayMode = decay_mode
        self._decay = float(decay)
        self._decay_window = int(decay_window)
        self._signal_delay = int(signal_delay)
        self._intraday_session_isolated_lag = bool(intraday_session_isolated_lag)
        self._nan_policy: NanPolicy = nan_policy
        self._temporal_mode: TemporalMode = temporal_mode
        self._neutralization = neutralization
        self._truncation = float(truncation)
        self._max_single_weight = max_single_weight
        self._ema_prev: dict[str, float] = {}
        self._linear_hist: dict[str, deque[float]] = {}
        self._last_decay_timestamp: Optional[pd.Timestamp] = None
        self._panel_columns: Optional[Tuple[str, ...]] = (
            tuple(panel_columns) if panel_columns is not None else None
        )
        if self._panel_columns is not None and not self._panel_columns:
            raise ValueError("panel_columns must be non-empty when provided")

    @property
    def decay(self) -> float:
        return self._decay

    @property
    def decay_mode(self) -> DecayMode:
        return self._decay_mode

    @property
    def decay_window(self) -> int:
        return self._decay_window

    @property
    def signal_delay(self) -> int:
        return self._signal_delay

    @property
    def intraday_session_isolated_lag(self) -> bool:
        return self._intraday_session_isolated_lag

    @property
    def neutralization(self) -> Neutralization:
        return self._neutralization

    @property
    def temporal_mode(self) -> TemporalMode:
        return self._temporal_mode

    @property
    def truncation(self) -> float:
        return self._truncation

    def reset_pipeline_state(self) -> None:
        """Clear temporal decay state (call before a new backtest or walk-forward run)."""
        self._ema_prev.clear()
        self._linear_hist.clear()
        self._last_decay_timestamp = None

    def panel_date_for_execution(self, execution_date: Union[str, pd.Timestamp]) -> pd.Timestamp:
        """
        Map backtest **execution** bar time to the timestamp used to load the feature panel
        (after applying :attr:`signal_delay`).
        """
        if self._signal_delay <= 0:
            return normalize_bar_timestamp(execution_date, self._ts.bar_spec)
        return self._ts.execution_lag_calendar_date(
            execution_date,
            lag=self._signal_delay,
            forbid_cross_session=(
                self._intraday_session_isolated_lag
                and bar_spec_is_intraday(self._ts.bar_spec)
            ),
        )

    def panel_at(
        self,
        date: Union[str, pd.Timestamp],
        *,
        live: bool = True,
        pasteurize: bool = False,
    ) -> Tuple[jnp.ndarray, List[str]]:
        """
        Cross-section for **execution** date ``date``: rows follow ``fin_ts.ticker_list``
        order. If :attr:`signal_delay` is positive, features load from
        :meth:`panel_date_for_execution`.

        By default keeps only tickers with **no NaN** in the feature block.
        If ``pasteurize`` is True (BRAIN-style pasteurization), impute column NaNs with
        the cross-sectional mean for that date (then still skip rows that are all-NaN).

        Returns ``(panel, tickers)``. Column order is ``STRATEGY_FEATURES_LIVE`` or
        ``STRATEGY_FEATURES`` when ``live`` matches, unless ``panel_columns`` was set on
        :class:`FinStrat` (then that tuple defines ``feat_cols``).
        """
        dt = self.panel_date_for_execution(date)
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

        use_live = live
        if self._panel_columns is not None:
            feat_cols = list(self._panel_columns)
        else:
            feat_cols = list(
                indicators.STRATEGY_FEATURES_LIVE
                if use_live
                else indicators.STRATEGY_FEATURES
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
        Read ``column`` from ``fin_ts.df`` at ``(ticker, panel_date)`` for each ticker,
        using :meth:`panel_date_for_execution` when ``signal_delay > 0``.

        Labels may be strings or ints; used with ``neutralization='group'``.
        """
        dt = self.panel_date_for_execution(date)
        df = self._ts.df
        if not isinstance(column, str) or not column.strip():
            raise ValueError("group column must be a non-empty string")
        if not isinstance(df, pd.DataFrame) or column not in df.columns:
            raise KeyError(
                f"Column {column!r} not in fin_ts dataframe. "
                f"Common classification columns: {STANDARD_GROUP_COLUMNS!r}"
            )
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
        return np.asarray(out)

    def tickers_at(self, date: Union[str, pd.Timestamp]) -> List[str]:
        """
        Tickers with finite OHLCV at the panel date used for ``date`` execution.
        """
        dt = self.panel_date_for_execution(date)
        df = self._ts.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("finTs dataframe is empty")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "tickers_at requires multi-ticker finTs data with MultiIndex (Ticker, Date)"
            )
        needed = ("Open", "High", "Low", "Close", "Volume")
        miss = [c for c in needed if c not in df.columns]
        if miss:
            raise KeyError(f"Dataframe missing required OHLCV columns: {miss}")
        out: List[str] = []
        for t in self._ts.ticker_list:
            key = (t, dt)
            if key not in df.index:
                continue
            row = df.loc[key, list(needed)]
            if isinstance(row, pd.DataFrame):
                continue
            vals = row.to_numpy(dtype=float)
            if np.isfinite(vals).all():
                out.append(t)
        return out

    def context_at(
        self,
        execution_date: Union[str, pd.Timestamp],
        *,
        tickers: Sequence[str],
    ) -> AlphaContext:
        """
        Build :class:`AlphaContext` with OHLCV history up to execution timestamp.
        """
        if not tickers:
            raise ValueError("tickers must be non-empty")
        dt = self.panel_date_for_execution(execution_date)
        cal = self._ts.get_trading_calendar(mode="observed").sort_values()
        pos = int(cal.searchsorted(dt, side="left"))
        if pos >= len(cal) or cal[pos] != dt:
            raise ValueError(f"panel date {dt!s} is not on trading calendar")
        hist = cal[: pos + 1]
        if len(hist) == 0:
            raise ValueError("empty historical calendar for context")

        df = self._ts.df
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("finTs dataframe is empty")
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("context_at requires MultiIndex (Ticker, Date) data")

        spec = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_volume": "Volume",
        }
        extra_cols = list(getattr(self._ts, "fundamental_feature_columns", tuple()))
        for col in spec.values():
            if col not in df.columns:
                raise KeyError(f"Dataframe missing required column {col!r}")
        for col in extra_cols:
            if col not in df.columns:
                raise KeyError(f"Dataframe missing context feature column {col!r}")

        tensors: dict[str, List[np.ndarray]] = {k: [] for k in spec}
        extra_tensors: dict[str, List[np.ndarray]] = {col: [] for col in extra_cols}
        for t in tickers:
            if t not in df.index.get_level_values("Ticker"):
                raise KeyError(f"ticker {t!r} missing from panel index")
            sub = df.xs(t, level="Ticker").sort_index()
            block = sub.reindex(hist)
            for key, col in spec.items():
                tensors[key].append(
                    pd.to_numeric(block[col], errors="coerce").to_numpy(dtype=float)
                )
            for col in extra_cols:
                extra_tensors[col].append(
                    pd.to_numeric(block[col], errors="coerce").to_numpy(dtype=float)
                )

        out: dict[str, jnp.ndarray] = {}
        for key, cols in tensors.items():
            mat = np.column_stack(cols) if cols else np.empty((len(hist), 0), dtype=float)
            out[key] = jnp.asarray(mat, dtype=jnp.float32)
        extra_out: dict[str, jnp.ndarray] = {}
        for key, cols in extra_tensors.items():
            mat = np.column_stack(cols) if cols else np.empty((len(hist), 0), dtype=float)
            extra_out[key] = jnp.asarray(mat, dtype=jnp.float32)
        return AlphaContext(
            open=out["open"],
            high=out["high"],
            low=out["low"],
            close=out["close"],
            adj_volume=out["adj_volume"],
            features=extra_out,
        )

    def scores_at(
        self,
        execution_date: Union[str, pd.Timestamp],
        *,
        tickers: Sequence[str],
    ) -> jnp.ndarray:
        """
        Execute the context-based alpha and return cross-sectional raw scores.
        """
        ctx = self.context_at(execution_date, tickers=tickers)
        try:
            raw: Any = self._algorithm(ctx)
        except TypeError as exc:
            raise TypeError(
                "alpha callable must accept a single AlphaContext argument: "
                "def alpha(ctx) -> scores"
            ) from exc
        if isinstance(raw, AlphaSeries):
            vec = jnp.asarray(raw.latest, dtype=jnp.float32)
        else:
            vec = jnp.asarray(raw, dtype=jnp.float32)
            if vec.ndim == 2:
                vec = vec[-1, :]
        if vec.ndim != 1:
            raise ValueError(
                f"alpha must return a 1D cross-section (or 2D history), got shape {vec.shape}"
            )
        if int(vec.shape[0]) != int(ctx.n_tickers):
            raise ValueError(
                f"alpha output length {vec.shape[0]} does not match ticker count {ctx.n_tickers}"
            )
        return vec

    def scores(self, indicators: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "scores(indicators) was removed. Use context-based alpha authoring and "
            "call scores_at(execution_date=..., tickers=...)."
        )

    def _apply_ema_decay(
        self, raw: jnp.ndarray, tickers: Sequence[str], *, gap_steps: int = 1
    ) -> jnp.ndarray:
        d = self._decay
        if d <= 0.0:
            return raw
        if len(tickers) != int(raw.shape[0]):
            raise ValueError("tickers length must match number of rows in scores")
        out_list: List[float] = []
        raw_np = [float(x) for x in jnp.asarray(raw).reshape(-1)]
        g = max(1, int(gap_steps))
        d_eff = float(d) ** float(g)
        for name, rv in zip(tickers, raw_np, strict=True):
            prev = self._ema_prev.get(name)
            if prev is None:
                sm = rv
            else:
                sm = (1.0 - d_eff) * rv + d_eff * prev
            self._ema_prev[name] = sm
            out_list.append(sm)
        return jnp.asarray(out_list, dtype=jnp.float32)

    def _apply_linear_decay(
        self, raw: jnp.ndarray, tickers: Sequence[str], *, gap_steps: int = 1
    ) -> jnp.ndarray:
        """
        BRAIN linear decay: weights ``n, n-1, ...`` on the last ``k`` raw scores
        (oldest ``n-k+1``, newest ``n``).
        """
        nwin = self._decay_window
        if nwin <= 1:
            return raw
        if len(tickers) != int(raw.shape[0]):
            raise ValueError("tickers length must match number of rows in scores")
        raw_np = [float(x) for x in jnp.asarray(raw).reshape(-1)]
        out_list: List[float] = []
        g = max(1, int(gap_steps))
        for name, rv in zip(tickers, raw_np, strict=True):
            hist = self._linear_hist.setdefault(name, deque(maxlen=nwin))
            for _ in range(g):
                hist.append(rv)
            k = len(hist)
            vals = list(hist)
            weights = [nwin - k + 1 + i for i in range(k)]
            den = float(sum(weights))
            num = sum(v * w for v, w in zip(vals, weights, strict=True))
            out_list.append(num / den)
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
        daily_indicators: Optional[jnp.ndarray],
        capital: Union[float, jnp.ndarray],
        *,
        tickers: Optional[Sequence[str]] = None,
        group_ids: Optional[Union[jnp.ndarray, Sequence[object]]] = None,
        execution_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> jnp.ndarray:
        """
        Run the pipeline and return dollar notionals whose gross sum equals ``capital``
        when dispersion is non-zero (after neutralization), unless caps zero everyone
        out.

        Args:
            daily_indicators: Legacy panel argument (must be ``None`` in context mode).
            capital: Target gross booksize.
            tickers: Ticker strings, same order as panel rows. **Required** if
                temporal decay is enabled, and for context-based alpha execution.
            group_ids: Same length as rows; required if ``neutralization == 'group'``.
        """
        if daily_indicators is not None:
            raise ValueError(
                "pass_(daily_indicators=...) was removed. Pass None and provide "
                "execution_date + tickers for context-based alpha execution."
            )
        if tickers is None:
            raise ValueError("tickers is required for context-based alpha execution")
        if execution_date is None:
            raise ValueError("execution_date is required for context-based alpha execution")
        raw = self.scores_at(execution_date, tickers=tickers)

        if self._nan_policy == "strict":
            if not bool(jnp.all(jnp.isfinite(raw))):
                raise ValueError("non-finite raw scores with nan_policy='strict'")
        elif self._nan_policy == "zero_fill":
            raw = jnp.where(jnp.isfinite(raw), raw, 0.0)

        needs_ema = self._decay_mode == "ema" and self._decay > 0.0
        needs_linear = self._decay_mode == "linear" and self._decay_window > 1
        if needs_ema or needs_linear:
            gap_steps = 1
            if self._temporal_mode == "elapsed_trading_time":
                if execution_date is None:
                    raise ValueError(
                        "execution_date is required when temporal_mode='elapsed_trading_time'"
                    )
                exec_ts = normalize_bar_timestamp(execution_date, self._ts.bar_spec)
                if self._last_decay_timestamp is not None:
                    bars, _seconds = self._ts.trading_distance(
                        self._last_decay_timestamp, exec_ts, mode="canonical"
                    )
                    gap_steps = max(1, int(abs(bars)))
                self._last_decay_timestamp = exec_ts
            else:
                if execution_date is not None:
                    self._last_decay_timestamp = normalize_bar_timestamp(
                        execution_date, self._ts.bar_spec
                    )
            if self._decay_mode == "ema":
                s = self._apply_ema_decay(raw, tickers, gap_steps=gap_steps)
            else:
                s = self._apply_linear_decay(raw, tickers, gap_steps=gap_steps)
        else:
            s = raw

        if self._truncation > 0.0:
            s = cross_section.winsorize(s, self._truncation)

        if self._neutralization == "market":
            s = cross_section.neutralize_market(s)
        elif self._neutralization == "group":
            if group_ids is None:
                raise ValueError("group_ids is required when neutralization='group'")
            gid = np.asarray(group_ids)
            if gid.shape[0] != s.shape[0]:
                raise ValueError("group_ids must have same length as scores")
            s = cross_section.neutralize_groups(s, gid)
        # else: none

        cap = jnp.asarray(capital, dtype=s.dtype)
        notionals = self._scale_gross(s, cap)
        return self._apply_max_weight_rescale(notionals, cap)
