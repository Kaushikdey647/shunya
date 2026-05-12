"""Portfolio-level risk vetting between PM targets and OMS (broker-agnostic core).

Optional pieces (CVXPY, scikit-learn LedoitWolf) live behind extras; import guarded.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from .execution import OrderAttempt
from .targets import (
    apply_group_gross_cap,
    apply_group_net_cap,
    broker_deltas,
    cap_deltas_by_adv,
    enforce_turnover_budget,
    scale_signed_targets_to_gross_cap,
)

logger = logging.getLogger(__name__)

_BP_SUBSTRINGS = (
    "insufficient buying power",
    "buying power",
    "insufficient bp",
    "not enough buying power",
)


def cvxpy_available() -> bool:
    try:
        import cvxpy  # noqa: F401

        return True
    except ImportError:
        return False


def ledoit_wolf_available() -> bool:
    try:
        from sklearn.covariance import LedoitWolf  # noqa: F401

        return True
    except ImportError:
        return False


class ShortabilityMode(str, Enum):
    """How to treat symbols that are not shortable when targets imply more short."""

    ZERO_NEW_SHORT = "zero_new_short"
    """Block new shorts from flat/long; allow reducing an existing short toward zero."""
    WARN_ONLY = "warn_only"
    """Record a warning only (no change)."""


@dataclass
class RiskVetConfig:
    """Limits for :meth:`PortfolioRiskEngine.vet` (all optional fields skip that check)."""

    max_gross_fraction_of_equity: Optional[float] = None
    """If set, gross book ``sum|target|`` capped at this times ``equity_usd``."""

    max_gross_book_usd: Optional[float] = None
    """Absolute gross cap in USD; combined with fraction via ``min`` when both set."""

    max_single_name_fraction: Optional[float] = None
    """Cap ``max_i |target_i|`` at this times ``equity_usd`` via a global scalar on targets."""

    max_adv_fraction: Optional[float] = None
    """ADV participation cap on **deltas** (see :func:`shunya.algorithm.targets.cap_deltas_by_adv`)."""

    max_turnover_fraction: Optional[float] = None
    """Turnover vs current (see :func:`shunya.algorithm.targets.enforce_turnover_budget`)."""

    sector_gross_cap_fraction: Optional[float] = None
    sector_net_cap_fraction: Optional[float] = None
    constraints_on_breach: Literal["rescale", "raise"] = "rescale"

    vol_target_annual: Optional[float] = None
    """Annualized vol target for scaling when realized vol is computed."""

    portfolio_vol_scale: Optional[float] = None
    """If set, multiply all targets by this scalar after vol-from-cov step (manual override)."""

    use_cvxpy: bool = False
    """If True and CVXPY installed, run QP in weight space before sequential caps."""

    cvxpy_sector_net_abs: Optional[float] = None
    """If set with sector labels, QP adds ``|sum_{i in sector} w_i| <= abs`` per sector."""

    buying_power_buffer: float = 0.98
    """Fraction of ``buying_power_usd`` applied to incremental buy notional cap."""

    shortability_mode: ShortabilityMode = ShortabilityMode.ZERO_NEW_SHORT

    fat_finger_limit_pct: Optional[float] = None
    """If ``limit_prices`` passed to ``vet``, reject (raise) when limit/last deviates beyond this."""

    def __post_init__(self) -> None:
        if not (0.0 < float(self.buying_power_buffer) <= 1.0):
            raise ValueError("buying_power_buffer must be in (0, 1]")


@dataclass
class RiskEngineState:
    """Mutable feedback from OMS into the next vet (e.g. buying-power rejections)."""

    buying_power_tighten: float = 1.0
    """Multiplier applied to effective buying power (starts at 1; feedback ratchets down)."""

    last_rejection_reason: Optional[str] = None

    def reset_tighten(self) -> None:
        self.buying_power_tighten = 1.0
        self.last_rejection_reason = None


@dataclass
class RiskVetResult:
    """Output of :meth:`PortfolioRiskEngine.vet`."""

    targets_vetted: Dict[str, float]
    targets_raw: Dict[str, float]
    flags: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    gross_before: Optional[float] = None
    gross_after: Optional[float] = None


def _universe_keys(
    proposed: Mapping[str, float],
    current: Mapping[str, float],
    universe: Optional[Sequence[str]],
) -> List[str]:
    if universe is not None:
        return [str(s) for s in universe]
    keys: Set[str] = set()
    keys.update(str(k) for k in proposed)
    keys.update(str(k) for k in current)
    return sorted(keys)


def _gross(m: Mapping[str, float]) -> float:
    return float(sum(abs(float(v)) for v in m.values()))


def _vol_scale_from_covariance(
    weights: np.ndarray,
    cov: np.ndarray,
    vol_target_annual: float,
    *,
    trading_days: float = 252.0,
) -> float:
    """Annualized portfolio vol from daily cov; return scale = target/realized (capped)."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    c = np.asarray(cov, dtype=float)
    if w.size != c.shape[0] or c.shape[0] != c.shape[1]:
        raise ValueError("weights and covariance must be square-aligned")
    var_d = float(w @ c @ w)
    if var_d <= 0.0 or not math.isfinite(var_d):
        return 1.0
    sigma_ann = math.sqrt(var_d * trading_days)
    if sigma_ann <= 1e-12:
        return 1.0
    return float(vol_target_annual) / sigma_ann


def _ledoit_wolf_cov(returns_txn: np.ndarray) -> np.ndarray:
    """returns_txn: T x N matrix of simple returns."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(np.asarray(returns_txn, dtype=float))
    return np.asarray(lw.covariance_, dtype=float)


def _apply_shortability(
    targets: Dict[str, float],
    current_usd: Mapping[str, float],
    shortable: Mapping[str, bool],
    mode: ShortabilityMode,
    messages: List[str],
) -> None:
    """In-place clip of targets for shortability (see module docstring for rules)."""
    for sym in list(targets.keys()):
        t = float(targets[sym])
        cur = float(current_usd.get(sym, 0.0))
        ok = bool(shortable.get(sym, True))
        if ok or t >= 0:
            continue
        if mode is ShortabilityMode.WARN_ONLY:
            messages.append(f"shortability_warn:{sym}:target={t}")
            continue
        # ZERO_NEW_SHORT
        if cur >= 0.0 and t < 0.0:
            if abs(t) > 1e-12:
                messages.append(f"shortability_clip:{sym}:blocked_new_short")
            targets[sym] = 0.0
        elif cur < 0.0:
            new_t = max(t, cur)
            if new_t != t:
                messages.append(f"shortability_clip:{sym}:blocked_increase_short")
            targets[sym] = new_t


def _single_name_scale(targets: Dict[str, float], equity: float, max_frac: float) -> Tuple[bool, float]:
    """Scale all targets by k so max |t| <= max_frac * equity."""
    cap = float(max_frac) * float(equity)
    if cap <= 0.0:
        return False, 1.0
    vals = [abs(float(v)) for v in targets.values() if abs(float(v)) > 1e-15]
    if not vals:
        return False, 1.0
    m = max(vals)
    if m <= cap:
        return False, 1.0
    k = cap / m
    for s in list(targets):
        targets[s] = float(targets[s]) * k
    return True, k


def _buying_power_scale_positive_deltas(
    targets: Dict[str, float],
    current_usd: Mapping[str, float],
    universe: Sequence[str],
    buying_power: float,
    messages: List[str],
) -> bool:
    """
    Scale positive deltas uniformly so sum max(0, delta) <= buying_power.

    Returns True if scaling applied.
    """
    deltas = broker_deltas(targets, current_usd, universe)
    pos_sum = sum(max(0.0, float(d)) for d in deltas.values())
    if pos_sum <= buying_power + 1e-9:
        return False
    k = buying_power / pos_sum if pos_sum > 0 else 1.0
    k = min(1.0, k)
    for sym in universe:
        cur = float(current_usd.get(sym, 0.0))
        t = float(targets.get(sym, 0.0))
        d = t - cur
        if d > 0:
            targets[sym] = cur + d * k
    messages.append(f"buying_power_scale:k={k:.6f}")
    return True


def _try_cvxpy_qp(
    w_raw: np.ndarray,
    *,
    max_l1: float,
    per_asset_cap: np.ndarray,
    sector_matrix: Optional[np.ndarray] = None,
    sector_net_abs: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Return w_opt or None if CVXPY missing or solve fails."""
    try:
        import cvxpy as cp
    except ImportError:
        return None

    n = int(w_raw.shape[0])
    w = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(w - w_raw))
    cons = [cp.norm(w, 1) <= float(max_l1)]
    capv = np.asarray(per_asset_cap, dtype=float)
    for i in range(n):
        cons.append(cp.abs(w[i]) <= float(capv[i]))
    if sector_matrix is not None and sector_net_abs is not None:
        sm = np.asarray(sector_matrix, dtype=float)
        for row in sm:
            cons.append(cp.abs(row @ w) <= float(sector_net_abs))
    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception as e:
        logger.debug("cvxpy solve failed: %s", e)
        return None
    if w.value is None:
        return None
    out = np.asarray(w.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(out)):
        return None
    return out


class PortfolioRiskEngine:
    """
    Blocking middleware: PM USD targets -> vetted USD targets for :class:`OrderManager`
    or :class:`~shunya.oms.service.InstitutionalOMS` (see :mod:`shunya.oms.risk_bridge`).

    Pipeline order (see source of :meth:`vet` for truth; summarized in docstring):

    1. Optional covariance-based annual vol scale (``covariance`` + ``return_symbols``).
    2. Optional Ledoit-Wolf vol scale from ``returns_txn`` (sklearn ``[risk]`` extra).
    3. Optional ``portfolio_vol_scale`` scalar.
    4. Optional CVXPY weight-space QP (``[risk]`` extra).
    5. Gross cap, sector gross/net caps, turnover budget (``targets`` helpers).
    6. ADV cap on deltas vs ``current_usd``.
    7. Per-name concentration scalar vs equity.
    8. Shortability clip.
    9. Buying power cap on positive incremental notional.
    """

    def __init__(self, config: RiskVetConfig, *, state: Optional[RiskEngineState] = None) -> None:
        self._config = config
        self.state = state or RiskEngineState()

    def register_execution_feedback(self, attempts: Sequence[OrderAttempt]) -> None:
        """Ratchet buying-power tighten when Alpaca (or mocks) report BP-related errors."""
        for a in attempts:
            err = (a.error or "") + " " + (a.status_error or "")
            low = err.lower()
            if any(s in low for s in _BP_SUBSTRINGS):
                self.state.last_rejection_reason = err.strip() or "buying_power"
                self.state.buying_power_tighten *= 0.9
                logger.info(
                    "RiskEngineState: buying_power_tighten=%s",
                    self.state.buying_power_tighten,
                )
                break

    def vet(
        self,
        proposed_targets: Mapping[str, float],
        *,
        current_usd: Mapping[str, float],
        equity_usd: float,
        prices: Mapping[str, float],
        universe: Optional[Sequence[str]] = None,
        adv_usd: Optional[Mapping[str, float]] = None,
        groups: Optional[Mapping[str, str]] = None,
        shortable_by_symbol: Optional[Mapping[str, bool]] = None,
        buying_power_usd: Optional[float] = None,
        returns_txn: Optional[np.ndarray] = None,
        return_symbols: Optional[Sequence[str]] = None,
        covariance: Optional[np.ndarray] = None,
        limit_prices: Optional[Mapping[str, float]] = None,
    ) -> RiskVetResult:
        """
        Vet ``proposed_targets`` to produce ``targets_vetted`` (USD notionals).

        ``returns_txn`` when provided is ``T x N`` simple returns with columns aligned
        to ``return_symbols`` (which must match the intersection order used for weights).
        """
        flags: List[str] = []
        messages: List[str] = []

        raw = {str(k): float(v) for k, v in proposed_targets.items()}
        t = {str(k): float(v) for k, v in raw.items()}
        uni = _universe_keys(t, current_usd, universe)
        for k in uni:
            t.setdefault(k, 0.0)

        gross0 = _gross(t)
        eq = float(equity_usd)
        if eq <= 0.0 and (self._config.max_gross_fraction_of_equity or self._config.max_single_name_fraction):
            messages.append("equity_non_positive:skipping_fraction_caps")

        # Fat-finger: limit vs last
        if limit_prices and self._config.fat_finger_limit_pct is not None:
            pct = float(self._config.fat_finger_limit_pct)
            for sym, lim in limit_prices.items():
                last = float(prices.get(str(sym), 0.0))
                if last <= 0:
                    continue
                ratio = lim / last
                if ratio > 1.0 + pct or ratio < max(0.0, 1.0 - pct):
                    raise ValueError(f"fat_finger_limit_pct breach: {sym} limit={lim} last={last}")

        # --- 1–2: vol targeting ---
        if (
            self._config.vol_target_annual is not None
            and eq > 0.0
            and covariance is not None
            and return_symbols is not None
        ):
            sym_list = [str(s) for s in return_symbols]
            w_vec = np.array([t.get(s, 0.0) / eq for s in sym_list], dtype=float)
            c = np.asarray(covariance, dtype=float)
            try:
                sc = _vol_scale_from_covariance(w_vec, c, float(self._config.vol_target_annual))
                sc = float(np.clip(sc, 1e-6, 1e6))
                for s in sym_list:
                    if s in t:
                        t[s] = float(t[s]) * sc
                flags.append("vol_target_cov")
                messages.append(f"vol_scale:{sc:.6f}")
            except ValueError as e:
                messages.append(f"vol_target_skip:{e}")

        if (
            returns_txn is not None
            and self._config.vol_target_annual is not None
            and eq > 0.0
            and "vol_target_cov" not in flags
        ):
            if not ledoit_wolf_available():
                messages.append("vol_target_ledoit_skip:no_sklearn")
            elif return_symbols is None:
                messages.append("vol_target_ledoit_skip:no_return_symbols")
            else:
                sym_list = [str(s) for s in return_symbols]
                R = np.asarray(returns_txn, dtype=float)
                if R.ndim == 2 and R.shape[1] == len(sym_list):
                    try:
                        cov = _ledoit_wolf_cov(R)
                        w_vec = np.array([t.get(s, 0.0) / eq for s in sym_list], dtype=float)
                        sc = _vol_scale_from_covariance(w_vec, cov, float(self._config.vol_target_annual))
                        sc = float(np.clip(sc, 1e-6, 1e6))
                        for s in sym_list:
                            if s in t:
                                t[s] = float(t[s]) * sc
                        flags.append("vol_target_ledoit")
                        messages.append(f"vol_scale_ledoit:{sc:.6f}")
                    except Exception as e:
                        messages.append(f"vol_target_ledoit_fail:{e}")

        if self._config.portfolio_vol_scale is not None:
            s = float(self._config.portfolio_vol_scale)
            for k in list(t):
                t[k] = float(t[k]) * s
            flags.append("portfolio_vol_scale")
            messages.append(f"manual_vol_scale:{s}")

        # --- 3: CVXPY ---
        if self._config.use_cvxpy and eq > 0.0:
            w_raw = np.array([t.get(s, 0.0) / eq for s in uni], dtype=float)
            gross_frac = 1.0
            if self._config.max_gross_fraction_of_equity is not None:
                gross_frac = min(gross_frac, float(self._config.max_gross_fraction_of_equity))
            if self._config.max_gross_book_usd is not None:
                capw = float(self._config.max_gross_book_usd) / eq
                gross_frac = min(gross_frac, capw)
            per_cap = np.full(len(uni), 1e6, dtype=float)
            if adv_usd is not None and self._config.max_adv_fraction is not None:
                part = float(self._config.max_adv_fraction)
                for i, s in enumerate(uni):
                    adv = float(adv_usd.get(s, 0.0))
                    per_cap[i] = (adv * part) / eq if adv > 0 else 1e6
            sector_mat: Optional[np.ndarray] = None
            if groups is not None and self._config.cvxpy_sector_net_abs is not None:
                sectors = sorted({str(groups.get(s, "Unknown")) for s in uni})
                sector_mat = np.zeros((len(sectors), len(uni)), dtype=float)
                for j, sec in enumerate(sectors):
                    for i, sym in enumerate(uni):
                        if str(groups.get(sym, "Unknown")) == sec:
                            sector_mat[j, i] = 1.0
            w_opt = _try_cvxpy_qp(
                w_raw,
                max_l1=gross_frac,
                per_asset_cap=per_cap,
                sector_matrix=sector_mat,
                sector_net_abs=self._config.cvxpy_sector_net_abs,
            )
            if w_opt is not None:
                for i, s in enumerate(uni):
                    t[s] = float(w_opt[i]) * eq
                flags.append("cvxpy_qp")
            else:
                messages.append("cvxpy_skip:unavailable_or_failed")

        # --- 4: sequential caps ---
        caps: List[float] = []
        if self._config.max_gross_book_usd is not None:
            caps.append(float(self._config.max_gross_book_usd))
        if self._config.max_gross_fraction_of_equity is not None and eq > 0:
            caps.append(float(self._config.max_gross_fraction_of_equity) * eq)
        if caps:
            gross_cap = min(caps)
            before = _gross(t)
            t = scale_signed_targets_to_gross_cap(t, gross_cap)
            if _gross(t) < before - 1e-9:
                flags.append("gross_cap")
                messages.append(f"gross_cap:{before:.2f}->{_gross(t):.2f}")

        if groups is not None and self._config.sector_gross_cap_fraction is not None:
            t2, br = apply_group_gross_cap(
                t,
                groups,
                max_group_gross_fraction=float(self._config.sector_gross_cap_fraction),
                on_breach=self._config.constraints_on_breach,
            )
            t = t2
            if br:
                flags.append("sector_gross")
                messages.append(f"sector_gross_breach:{br}")

        if groups is not None and self._config.sector_net_cap_fraction is not None:
            t2, br = apply_group_net_cap(
                t,
                groups,
                max_group_net_fraction=float(self._config.sector_net_cap_fraction),
                on_breach=self._config.constraints_on_breach,
            )
            t = t2
            if br:
                flags.append("sector_net")
                messages.append(f"sector_net_breach:{br}")

        if self._config.max_turnover_fraction is not None:
            t2, turn, lim = enforce_turnover_budget(
                t,
                current_usd,
                max_turnover_fraction=float(self._config.max_turnover_fraction),
                on_breach=self._config.constraints_on_breach,
            )
            if turn > lim + 1e-9:
                flags.append("turnover")
                messages.append(f"turnover_rescale:{turn:.2f}>{lim:.2f}")
            t = t2

        # --- 5: ADV on deltas ---
        if adv_usd is not None and self._config.max_adv_fraction is not None:
            d0 = broker_deltas(t, current_usd, uni)
            d1, br = cap_deltas_by_adv(
                d0,
                adv_usd,
                max_adv_fraction=float(self._config.max_adv_fraction),
                on_breach=self._config.constraints_on_breach,
            )
            if br:
                flags.append("adv")
                messages.append(f"adv_clip:{br}")
            for sym in uni:
                cur = float(current_usd.get(sym, 0.0))
                t[sym] = cur + float(d1.get(sym, 0.0))

        # --- 6: single-name ---
        if self._config.max_single_name_fraction is not None and eq > 0:
            applied, k = _single_name_scale(t, eq, float(self._config.max_single_name_fraction))
            if applied:
                flags.append("single_name")
                messages.append(f"single_name_scale:k={k:.6f}")

        # --- 7: shortability ---
        if shortable_by_symbol is not None:
            _apply_shortability(t, current_usd, shortable_by_symbol, self._config.shortability_mode, messages)
            if any(m.startswith("shortability_clip") for m in messages):
                flags.append("shortability")

        # --- 8: buying power ---
        if buying_power_usd is not None:
            eff = float(buying_power_usd) * float(self._config.buying_power_buffer) * float(
                self.state.buying_power_tighten
            )
            if eff > 0 and _buying_power_scale_positive_deltas(t, current_usd, uni, eff, messages):
                flags.append("buying_power")

        gross1 = _gross(t)
        return RiskVetResult(
            targets_vetted={k: float(v) for k, v in t.items()},
            targets_raw=dict(raw),
            flags=flags,
            messages=messages,
            gross_before=gross0,
            gross_after=gross1,
        )


AccountEquityFn = Callable[[], Union[float, Any]]
KillSwitchFn = Callable[[], Any]


class DrawdownSentinel:
    """
    Background equity monitor; on breach calls ``kill_switch`` (cancel / flatten).

    Does not start itself: the host coroutine should ``asyncio.create_task(sentinel.run())``.
    """

    def __init__(
        self,
        *,
        max_drawdown_pct: float,
        poll_interval_seconds: float = 10.0,
        account_equity: AccountEquityFn,
        kill_switch: KillSwitchFn,
        high_water_mark: Optional[float] = None,
    ) -> None:
        if not (0.0 < float(max_drawdown_pct) < 1.0):
            raise ValueError("max_drawdown_pct must be in (0, 1)")
        if float(poll_interval_seconds) <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._max_dd = float(max_drawdown_pct)
        self._interval = float(poll_interval_seconds)
        self._equity_fn = account_equity
        self._kill = kill_switch
        self._hwm = float(high_water_mark) if high_water_mark is not None else 0.0
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def high_water_mark(self) -> float:
        return self._hwm

    async def _resolve_equity(self) -> float:
        out = self._equity_fn()
        if inspect.isawaitable(out):
            return float(await out)
        return float(out)

    async def _invoke_kill(self) -> None:
        out = self._kill()
        if inspect.isawaitable(out):
            await out

    async def run(self) -> None:
        """Loop until drawdown breach or cancelled."""
        while True:
            try:
                eq = await self._resolve_equity()
                if math.isfinite(eq) and eq > self._hwm:
                    self._hwm = eq
                if self._hwm > 0 and math.isfinite(eq):
                    dd = (self._hwm - eq) / self._hwm
                    if dd > self._max_dd:
                        self._triggered = True
                        await self._invoke_kill()
                        return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("DrawdownSentinel tick error: %s", e)
            await asyncio.sleep(self._interval)
