"""Panel-schedule portfolio construction: target blend, alpha blend, and rolling metrics.

This module is intentionally independent of market-data transports and broker OMS.
Callers refresh prices or panels elsewhere, compute USD targets, then hand results to
execution separately.

Prefer :class:`PortfolioConstructionService` for new integrations (risk, OMS, EMS);
:class:`PortfolioManager` and :class:`AlphaBlendPortfolioManager` remain thin facades.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..data.fints import finTs
from .alpha_context import AlphaContext
from .finstrat import FinStrat
from .targets import target_usd_universe

StrategySlot = Tuple[str, FinStrat, float]
PORTFOLIO_PERF_KEY = "__portfolio__"


class TickerUniversePolicy(str, Enum):
    """How sub-strategy universes are aligned before stacking raw scores or ``pass_``."""

    STRICT_INTERSECTION = "strict_intersection"
    """Require identical ``tickers_at(execution_date)`` sets across all sub-strategies (default)."""

    # SUPERSET_MASKED reserved for a future path: union universe + per-strat zero padding.


class BlendModeKind(str, Enum):
    TARGET = "target"
    ALPHA = "alpha"


def _execution_date_iso(execution_date: pd.Timestamp) -> str:
    ts = pd.Timestamp(execution_date)
    if hasattr(ts, "isoformat"):
        return str(ts.isoformat())
    return str(ts)


def _validate_weights(weights: Sequence[float]) -> None:
    ws = [float(w) for w in weights]
    if not ws:
        raise ValueError("strategy weights must be non-empty")
    if any(w < 0.0 for w in ws):
        raise ValueError("strategy weights must be non-negative")
    s = float(sum(ws))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("strategy weights must sum to a positive finite value")
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"strategy weights must sum to 1.0, got {s}")


def _normalize_positive_convictions(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if np.any(arr < 0):
        raise ValueError("convictions must be non-negative")
    s = float(np.sum(arr))
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("convictions must sum to a positive finite value")
    return arr / s


def allocate_proportional_by_request(
    filled_usd: float,
    requests_by_strategy: Mapping[str, float],
) -> Dict[str, float]:
    """
    Split a signed fill across sub-strategies in proportion to their signed requests.

    ``sum(output.values()) == filled_usd`` when ``sum(requests) != 0``.
    """
    if not requests_by_strategy:
        return {}
    s = float(sum(requests_by_strategy.values()))
    if abs(s) < 1e-18:
        n = len(requests_by_strategy)
        return {k: float(filled_usd) / n for k in requests_by_strategy}
    return {k: float(filled_usd * float(v) / s) for k, v in requests_by_strategy.items()}


def inverse_vol_weights(vol_by_id: Mapping[str, float], *, eps: float = 1e-12) -> Dict[str, float]:
    """
    Map non-negative volatility estimates to normalized weights ``∝ 1/(vol+eps)``.

    Caller supplies vols (e.g. rolling std of realized returns); ids with missing or
    non-finite vol are treated as ``vol=0``.
    """
    if not vol_by_id:
        return {}
    inv: Dict[str, float] = {}
    for sid, v in vol_by_id.items():
        vv = float(v) if np.isfinite(v) and float(v) >= 0 else 0.0
        inv[str(sid)] = 1.0 / (vv + eps)
    s = float(sum(inv.values()))
    if s <= 0.0:
        n = max(len(inv), 1)
        return {k: 1.0 / n for k in inv}
    return {k: float(v / s) for k, v in inv.items()}


def combine_weighted_targets(
    weighted_targets: Sequence[Tuple[Mapping[str, float], float]],
) -> Dict[str, float]:
    """Net USD targets: ``sum_i w_i * T_i,k`` when each ``T_i`` is scaled to a full book.

    For targets already sized with ``capital_i = w_i * capital`` per strategy, use
    :func:`sum_target_maps` instead so weights are not applied twice.
    """
    out: Dict[str, float] = {}
    for targets, w in weighted_targets:
        wf = float(w)
        for sym, v in targets.items():
            k = str(sym)
            out[k] = out.get(k, 0.0) + wf * float(v)
    return out


def sum_target_maps(maps: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Elementwise sum of USD target maps (e.g. strategies each sized to a slice of capital)."""
    out: Dict[str, float] = {}
    for m in maps:
        for sym, v in m.items():
            k = str(sym)
            out[k] = out.get(k, 0.0) + float(v)
    return out


@dataclass(frozen=True)
class StrategySpec:
    """Registered sub-advisor for alpha-blend or configuration metadata."""

    strategy_id: str
    sub_strat: FinStrat
    conviction_z: float
    turnover_class: Optional[str] = None
    capacity_notes: Optional[str] = None

    def __post_init__(self) -> None:
        if float(self.conviction_z) < 0.0:
            raise ValueError("conviction_z must be non-negative")
        if not str(self.strategy_id).strip():
            raise ValueError("strategy_id must be non-empty")


@dataclass(frozen=True)
class StrategyRegistry:
    """Immutable book of :class:`StrategySpec` entries."""

    specs: Tuple[StrategySpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("StrategyRegistry requires at least one StrategySpec")
        ids = [s.strategy_id for s in self.specs]
        if len(set(ids)) != len(ids):
            raise ValueError("strategy_id values must be unique within StrategyRegistry")

    @staticmethod
    def from_specs(specs: Sequence[StrategySpec]) -> StrategyRegistry:
        return StrategyRegistry(tuple(specs))


@dataclass
class VirtualLedger:
    """
    In-memory theoretical USD holdings per sub-strategy (not the broker cash view).

    Mutate via :meth:`apply_delta`; use :func:`allocate_proportional_by_request` to
    split fills across strategies before applying.
    """

    _rows: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self._rows.clear()

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {sid: dict(m) for sid, m in self._rows.items()}

    def apply_delta(self, strategy_id: str, symbol: str, delta_usd: float) -> None:
        sid, sym = str(strategy_id), str(symbol)
        row = self._rows.setdefault(sid, {})
        row[sym] = float(row.get(sym, 0.0)) + float(delta_usd)

    def position_usd(self, strategy_id: str, symbol: str) -> float:
        return float(self._rows.get(str(strategy_id), {}).get(str(symbol), 0.0))


def mark_to_market_strategy_pnl_usd(
    ledger: VirtualLedger,
    mark_px: Mapping[str, float],
    prior_mark_px: Mapping[str, float],
) -> Dict[str, float]:
    """
    Per-strategy one-bar mark-to-market PnL from USD-notional ledger rows.

    For each open symbol with both marks, adds ``position_usd * (mark/prior - 1)``.
    Symbols missing from either price map are skipped. Pure helper for tests and
    future live attribution; no broker or transport coupling.
    """
    out: Dict[str, float] = {}
    snap = ledger.snapshot()
    for sid, row in snap.items():
        pnl = 0.0
        for sym, pos_usd in row.items():
            if abs(float(pos_usd)) < 1e-18:
                continue
            p0 = prior_mark_px.get(sym)
            p1 = mark_px.get(sym)
            if p0 is None or p1 is None:
                continue
            p0f, p1f = float(p0), float(p1)
            if not (np.isfinite(p0f) and np.isfinite(p1f)) or abs(p0f) < 1e-18:
                continue
            pnl += float(pos_usd) * (p1f / p0f - 1.0)
        out[str(sid)] = pnl
    return out


@dataclass
class RollingSharpeTracker:
    """Fixed-length deque of simple returns per strategy or portfolio key."""

    window: int
    annualization_factor: float = 252.0
    _series: Dict[str, Deque[float]] = field(default_factory=dict)

    def record_return(self, key: str, daily_return: float) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2 for Sharpe estimation")
        if key not in self._series:
            self._series[key] = deque(maxlen=int(self.window))
        self._series[key].append(float(daily_return))

    def rolling_sharpe(self, key: str) -> Optional[float]:
        dq = self._series.get(key)
        if dq is None or len(dq) < 2:
            return None
        arr = np.asarray(dq, dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1))
        if sd <= 1e-12 or not np.isfinite(sd):
            return None
        return (mu / sd) * float(np.sqrt(self.annualization_factor))

    def rolling_std(self, key: str) -> Optional[float]:
        dq = self._series.get(key)
        if dq is None or len(dq) < 2:
            return None
        arr = np.asarray(dq, dtype=float)
        sd = float(np.std(arr, ddof=1))
        return sd if sd > 1e-18 and np.isfinite(sd) else None


class StrategyReturnFeed(Protocol):
    """Optional caller-implemented source of simple daily returns keyed by ``strategy_id``."""

    def daily_returns_asof(self, execution_date: pd.Timestamp) -> Optional[Mapping[str, float]]:
        """Return mapping of strategy id to simple return for the bar ending at ``execution_date``, or None."""


@dataclass(frozen=True)
class PortfolioConstructionResult:
    """Structured output for risk, OMS audit, and logging (targets remain the primary field)."""

    targets: Dict[str, float]
    blend_mode: str
    requested_capital: float
    active_capital: float
    correlation_penalty_applied: bool
    max_pairwise_correlation: Optional[float]
    tickers: Tuple[str, ...]
    execution_date_iso: str


def _assert_shared_fin_ts(strategies: Sequence[FinStrat]) -> finTs:
    first_ts = strategies[0]._ts
    for strat in strategies[1:]:
        if strat._ts is not first_ts:
            raise ValueError(
                "all FinStrat instances must share the same finTs object as the first strategy"
            )
    return first_ts


def _aligned_ticker_list(strategies: Sequence[FinStrat], execution_date: pd.Timestamp) -> List[str]:
    names_ref = list(strategies[0].tickers_at(execution_date))
    ref_set = set(names_ref)
    for strat in strategies[1:]:
        if set(strat.tickers_at(execution_date)) != ref_set:
            raise ValueError("tickers_at(execution_date) must match across all strategies")
    return names_ref


def _rolling_cross_section_path_correlation(
    raw_history: Mapping[str, Deque[np.ndarray]],
    id_a: str,
    id_b: str,
) -> Optional[float]:
    da = raw_history.get(str(id_a))
    db = raw_history.get(str(id_b))
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None
    L = min(len(da), len(db))
    try:
        A = np.stack([np.asarray(da[-L + k], dtype=float) for k in range(L)])
        B = np.stack([np.asarray(db[-L + k], dtype=float) for k in range(L)])
    except ValueError:
        return None
    if A.shape != B.shape:
        return None
    ca = A.ravel()
    cb = B.ravel()
    sa = float(np.std(ca, ddof=1))
    sb = float(np.std(cb, ddof=1))
    if sa < 1e-12 or sb < 1e-12:
        return None
    return float(np.corrcoef(ca, cb)[0, 1])


def _max_pairwise_path_correlation(
    raw_history: Mapping[str, Deque[np.ndarray]],
    strategy_ids: Sequence[str],
) -> Optional[float]:
    ids = list(strategy_ids)
    if len(ids) < 2:
        return None
    best: Optional[float] = None
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            c = _rolling_cross_section_path_correlation(raw_history, ids[i], ids[j])
            if c is None:
                continue
            a = abs(float(c))
            best = a if best is None else max(best, a)
    return best


def _blend_raws(raws: Sequence[jnp.ndarray], z: np.ndarray) -> jnp.ndarray:
    stacked = jnp.stack([jnp.asarray(r, dtype=jnp.float32) for r in raws], axis=0)
    w = jnp.asarray(z, dtype=jnp.float32).reshape(-1, 1)
    return jnp.sum(stacked * w, axis=0)


def _last_sub_strategy_raw_scores(raw_history: Mapping[str, Deque[np.ndarray]]) -> Dict[str, jnp.ndarray]:
    return {sid: jnp.asarray(list(dq)[-1]) for sid, dq in raw_history.items() if dq}


@dataclass(frozen=True)
class TargetBlendConfig:
    """Book definition for target blending (late aggregation)."""

    strategies: Tuple[StrategySlot, ...]

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ValueError("TargetBlendConfig requires at least one strategy slot")
        _validate_weights([w for _, _, w in self.strategies])
        _assert_shared_fin_ts([t for _, t, _ in self.strategies])


@dataclass
class AlphaBlendConfig:
    """Book definition for alpha blending (early aggregation + master risk pass)."""

    registry: StrategyRegistry
    master: FinStrat
    raw_history_maxlen: int = 126
    correlation_max_pairwise_threshold: Optional[float] = None
    correlation_penalty: float = 0.85
    cash_sweep: Optional[Callable[[float], Mapping[str, float]]] = None
    virtual: VirtualLedger = field(default_factory=VirtualLedger)

    def __post_init__(self) -> None:
        if int(self.raw_history_maxlen) < 2:
            raise ValueError("raw_history_maxlen must be >= 2")
        if not (0.0 < float(self.correlation_penalty) <= 1.0):
            raise ValueError("correlation_penalty must be in (0, 1]")
        subs = [s.sub_strat for s in self.registry.specs]
        fts = _assert_shared_fin_ts(subs)
        if self.master._ts is not fts:
            raise ValueError("master FinStrat must share the same finTs object as sub_strat instances")


def _construct_target_blend(
    book: TargetBlendConfig,
    *,
    capital: float,
    execution_date: pd.Timestamp,
    group_column: Optional[str],
) -> PortfolioConstructionResult:
    fts = _assert_shared_fin_ts([t for _, t, _ in book.strategies])
    names = _aligned_ticker_list([t for _, t, _ in book.strategies], execution_date)
    maps: List[Dict[str, float]] = []
    for sid, strat, w in book.strategies:
        pass_kw: dict = {"tickers": names}
        if strat.neutralization == "group":
            col = group_column or "Sector"
            pass_kw["group_ids"] = strat.group_labels_at(execution_date, names, col)
        cap_i = float(w) * float(capital)
        vec = jnp.asarray(strat.pass_(None, cap_i, execution_date=execution_date, **pass_kw))
        maps.append(target_usd_universe(names, np.asarray(vec, dtype=float), fts.ticker_list))
    targets = sum_target_maps(maps)
    cap_f = float(capital)
    return PortfolioConstructionResult(
        targets=dict(targets),
        blend_mode=BlendModeKind.TARGET.value,
        requested_capital=cap_f,
        active_capital=cap_f,
        correlation_penalty_applied=False,
        max_pairwise_correlation=None,
        tickers=tuple(names),
        execution_date_iso=_execution_date_iso(execution_date),
    )


def _construct_alpha_blend(
    book: AlphaBlendConfig,
    raw_history: Dict[str, Deque[np.ndarray]],
    *,
    capital: float,
    execution_date: pd.Timestamp,
    group_column: Optional[str],
    record_raw_history: bool,
) -> PortfolioConstructionResult:
    fts = book.master._ts
    names = _aligned_ticker_list([s.sub_strat for s in book.registry.specs], execution_date)
    ref = book.registry.specs[0].sub_strat
    ctx: AlphaContext = ref.context_at(execution_date, tickers=names)

    raws: List[jnp.ndarray] = []
    for spec in book.registry.specs:
        vec = spec.sub_strat.scores_from_context(ctx)
        raws.append(vec)
        if record_raw_history:
            raw_history[spec.strategy_id].append(np.asarray(vec, dtype=float))

    z = _normalize_positive_convictions([s.conviction_z for s in book.registry.specs])
    thr = book.correlation_max_pairwise_threshold
    max_c: Optional[float] = None
    penalty_applied = False
    active_capital = float(capital)
    if thr is not None and len(book.registry.specs) > 1:
        ids = [s.strategy_id for s in book.registry.specs]
        max_c = _max_pairwise_path_correlation(raw_history, ids)
        if max_c is not None and max_c > float(thr):
            active_capital *= float(book.correlation_penalty)
            penalty_applied = True

    blended = _blend_raws(raws, z)

    pass_kw: dict = {}
    if book.master.neutralization == "group":
        col = group_column or "Sector"
        pass_kw["group_ids"] = book.master.group_labels_at(execution_date, names, col)

    notionals = book.master.process_raw_scores(
        blended,
        active_capital,
        tickers=names,
        execution_date=execution_date,
        **pass_kw,
    )
    targets = target_usd_universe(names, np.asarray(notionals, dtype=float), fts.ticker_list)
    return PortfolioConstructionResult(
        targets=dict(targets),
        blend_mode=BlendModeKind.ALPHA.value,
        requested_capital=float(capital),
        active_capital=float(active_capital),
        correlation_penalty_applied=penalty_applied,
        max_pairwise_correlation=max_c,
        tickers=tuple(names),
        execution_date_iso=_execution_date_iso(execution_date),
    )


@dataclass
class PortfolioConstructionService:
    """
    Canonical portfolio construction entry point for shared-``finTs`` books.

    Dispatches on :class:`TargetBlendConfig` vs :class:`AlphaBlendConfig`. Rolling
    Sharpe bookkeeping uses **caller-supplied** simple returns via
    :meth:`record_strategy_return` unless an optional :class:`StrategyReturnFeed` is
    installed and :attr:`record_returns_from_feed_on_construct` is True.

    Ticker alignment defaults to :attr:`ticker_universe_policy`
    ``STRICT_INTERSECTION`` (identical ``tickers_at`` per strategy); heterogeneous
    universes require a future superset-and-mask path.
    """

    book: Union[TargetBlendConfig, AlphaBlendConfig]
    sharpe_window: int = 60
    return_feed: Optional[StrategyReturnFeed] = None
    record_returns_from_feed_on_construct: bool = False
    ticker_universe_policy: TickerUniversePolicy = TickerUniversePolicy.STRICT_INTERSECTION
    sharpe: RollingSharpeTracker = field(init=False)
    _alpha_raw_history: Optional[Dict[str, Deque[np.ndarray]]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if int(self.sharpe_window) < 2:
            raise ValueError("sharpe_window must be >= 2")
        self.sharpe = RollingSharpeTracker(window=int(self.sharpe_window))
        if isinstance(self.book, AlphaBlendConfig):
            self._alpha_raw_history = {
                s.strategy_id: deque(maxlen=int(self.book.raw_history_maxlen))
                for s in self.book.registry.specs
            }
        else:
            self._alpha_raw_history = None

    def _maybe_record_returns_from_feed(self, execution_date: pd.Timestamp) -> None:
        if not self.record_returns_from_feed_on_construct or self.return_feed is None:
            return
        m = self.return_feed.daily_returns_asof(execution_date)
        if not m:
            return
        for k, v in m.items():
            self.sharpe.record_return(str(k), float(v))

    def construct(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
        record_raw_history: bool = True,
    ) -> PortfolioConstructionResult:
        self._maybe_record_returns_from_feed(execution_date)
        if isinstance(self.book, TargetBlendConfig):
            return _construct_target_blend(
                self.book,
                capital=capital,
                execution_date=execution_date,
                group_column=group_column,
            )
        assert self._alpha_raw_history is not None
        return _construct_alpha_blend(
            self.book,
            self._alpha_raw_history,
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
            record_raw_history=record_raw_history,
        )

    def net_targets(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
        record_raw_history: bool = True,
    ) -> Dict[str, float]:
        return self.construct(
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
            record_raw_history=record_raw_history,
        ).targets

    def targets_by_strategy(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
    ) -> List[Tuple[str, Dict[str, float]]]:
        if not isinstance(self.book, TargetBlendConfig):
            raise TypeError("targets_by_strategy is only defined for TargetBlendConfig")
        fts = _assert_shared_fin_ts([t for _, t, _ in self.book.strategies])
        names = _aligned_ticker_list([t for _, t, _ in self.book.strategies], execution_date)
        out: List[Tuple[str, Dict[str, float]]] = []
        for sid, strat, w in self.book.strategies:
            pass_kw: dict = {"tickers": names}
            if strat.neutralization == "group":
                col = group_column or "Sector"
                pass_kw["group_ids"] = strat.group_labels_at(execution_date, names, col)
            cap_i = float(w) * float(capital)
            vec = jnp.asarray(strat.pass_(None, cap_i, execution_date=execution_date, **pass_kw))
            targets = target_usd_universe(names, np.asarray(vec, dtype=float), fts.ticker_list)
            out.append((sid, targets))
        return out

    def rolling_cross_section_path_correlation(self, id_a: str, id_b: str) -> Optional[float]:
        if self._alpha_raw_history is None:
            raise TypeError("rolling_cross_section_path_correlation requires an alpha blend book")
        return _rolling_cross_section_path_correlation(self._alpha_raw_history, id_a, id_b)

    def last_sub_strategy_raw_scores(self) -> Dict[str, jnp.ndarray]:
        if self._alpha_raw_history is None:
            raise TypeError("last_sub_strategy_raw_scores requires an alpha blend book")
        return _last_sub_strategy_raw_scores(self._alpha_raw_history)

    def record_strategy_return(self, strategy_id: str, daily_simple_return: float) -> None:
        self.sharpe.record_return(str(strategy_id), daily_simple_return)

    def record_portfolio_return(self, daily_simple_return: float) -> None:
        self.sharpe.record_return(PORTFOLIO_PERF_KEY, daily_simple_return)

    def rolling_sharpe(self, strategy_id: str) -> Optional[float]:
        return self.sharpe.rolling_sharpe(str(strategy_id))

    def portfolio_rolling_sharpe(self) -> Optional[float]:
        return self.sharpe.rolling_sharpe(PORTFOLIO_PERF_KEY)


@dataclass
class PortfolioManager:
    """
    **Target blending (late aggregation):** each sub-:class:`FinStrat` runs full
    :meth:`FinStrat.pass_` with ``weight_i * capital``; USD maps are summed.

    Legacy convenience facade over :class:`PortfolioConstructionService` with
    :class:`TargetBlendConfig`; prefer the service for new risk / OMS integrations.
    """

    strategies: Sequence[StrategySlot]
    sharpe_window: int = 60
    _svc: PortfolioConstructionService = field(init=False, repr=False)
    sharpe: RollingSharpeTracker = field(init=False)

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ValueError("PortfolioManager requires at least one (strategy_id, FinStrat, weight)")
        self._svc = PortfolioConstructionService(
            TargetBlendConfig(tuple(self.strategies)),
            sharpe_window=int(self.sharpe_window),
        )
        self.sharpe = self._svc.sharpe

    def _shared_fin_ts(self) -> finTs:
        strats = [t for _, t, _ in self.strategies]
        return _assert_shared_fin_ts(strats)

    def _aligned_tickers(self, execution_date: pd.Timestamp) -> List[str]:
        strats = [t for _, t, _ in self.strategies]
        return _aligned_ticker_list(strats, execution_date)

    def targets_by_strategy(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Per-strategy USD targets after each strat's own ``pass_`` pipeline."""
        return self._svc.targets_by_strategy(
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
        )

    def net_targets(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
    ) -> Dict[str, float]:
        """Sum of per-strategy USD targets (each already sized with ``weight * capital``)."""
        return self._svc.net_targets(
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
        )

    def construct(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
    ) -> PortfolioConstructionResult:
        return self._svc.construct(
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
            record_raw_history=True,
        )

    def record_strategy_return(self, strategy_id: str, daily_simple_return: float) -> None:
        self._svc.record_strategy_return(strategy_id, daily_simple_return)

    def record_portfolio_return(self, daily_simple_return: float) -> None:
        self._svc.record_portfolio_return(daily_simple_return)

    def rolling_sharpe(self, strategy_id: str) -> Optional[float]:
        return self._svc.rolling_sharpe(strategy_id)

    def portfolio_rolling_sharpe(self) -> Optional[float]:
        return self._svc.portfolio_rolling_sharpe()


@dataclass
class AlphaBlendPortfolioManager:
    """
    **Alpha blending (early aggregation):** one shared :class:`AlphaContext`, raw
    scores from each sub-advisor, weighted sum, then a single **master**
    :meth:`FinStrat.process_raw_scores` pass (decay, truncation, neutralization, gross
    scaling on the master only). Sub-advisor :class:`FinStrat` neutralization settings
    do not apply to the blend path—only :meth:`FinStrat.scores_from_context` is used.

    Optional correlation dampening: when the maximum pairwise rolling path correlation
    exceeds ``correlation_max_pairwise_threshold``, **gross** is reduced by multiplying
    **capital** passed to the master by ``correlation_penalty`` (conviction weights are
    unchanged).

    Legacy convenience facade over :class:`PortfolioConstructionService` with
    :class:`AlphaBlendConfig`.
    """

    registry: StrategyRegistry
    master: FinStrat
    sharpe_window: int = 60
    raw_history_maxlen: int = 126
    correlation_max_pairwise_threshold: Optional[float] = None
    correlation_penalty: float = 0.85
    cash_sweep: Optional[Callable[[float], Mapping[str, float]]] = None
    virtual: VirtualLedger = field(default_factory=VirtualLedger)
    _svc: PortfolioConstructionService = field(init=False, repr=False)
    sharpe: RollingSharpeTracker = field(init=False)

    def __post_init__(self) -> None:
        if int(self.sharpe_window) < 2:
            raise ValueError("sharpe_window must be >= 2")
        cfg = AlphaBlendConfig(
            registry=self.registry,
            master=self.master,
            raw_history_maxlen=int(self.raw_history_maxlen),
            correlation_max_pairwise_threshold=self.correlation_max_pairwise_threshold,
            correlation_penalty=float(self.correlation_penalty),
            cash_sweep=self.cash_sweep,
            virtual=self.virtual,
        )
        self._svc = PortfolioConstructionService(
            cfg,
            sharpe_window=int(self.sharpe_window),
        )
        self.sharpe = self._svc.sharpe

    def _aligned_tickers(self, execution_date: pd.Timestamp) -> List[str]:
        subs = [s.sub_strat for s in self.registry.specs]
        return _aligned_ticker_list(subs, execution_date)

    def rolling_cross_section_path_correlation(self, id_a: str, id_b: str) -> Optional[float]:
        """
        Pearson correlation between flattened stacks of the last ``L`` raw cross-sections.

        Each history entry is one ``(n,)`` raw vector from :meth:`FinStrat.scores_from_context`.
        """
        return self._svc.rolling_cross_section_path_correlation(id_a, id_b)

    def last_sub_strategy_raw_scores(self) -> Dict[str, jnp.ndarray]:
        """Most recent raw cross-section per sub-strategy (after :meth:`net_targets`)."""
        return self._svc.last_sub_strategy_raw_scores()

    def net_targets(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
        record_raw_history: bool = True,
    ) -> Dict[str, float]:
        return self._svc.net_targets(
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
            record_raw_history=record_raw_history,
        )

    def construct(
        self,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        group_column: Optional[str] = "Sector",
        record_raw_history: bool = True,
    ) -> PortfolioConstructionResult:
        return self._svc.construct(
            capital=capital,
            execution_date=execution_date,
            group_column=group_column,
            record_raw_history=record_raw_history,
        )

    def record_strategy_return(self, strategy_id: str, daily_simple_return: float) -> None:
        self._svc.record_strategy_return(strategy_id, daily_simple_return)

    def record_portfolio_return(self, daily_simple_return: float) -> None:
        self._svc.record_portfolio_return(daily_simple_return)

    def rolling_sharpe(self, strategy_id: str) -> Optional[float]:
        return self._svc.rolling_sharpe(strategy_id)

    def portfolio_rolling_sharpe(self) -> Optional[float]:
        return self._svc.portfolio_rolling_sharpe()

    def convictions_from_inverse_vol(self) -> Dict[str, float]:
        """
        Replace book convictions using :func:`inverse_vol_weights` over rolling return
        vols from :meth:`RollingSharpeTracker.rolling_std` (requires recorded returns).
        """
        vols = {s.strategy_id: self.sharpe.rolling_std(s.strategy_id) or 0.0 for s in self.registry.specs}
        return inverse_vol_weights(vols)
