"""Daily market-on-open runner: cron calls :func:`tick` every ~15 minutes; it decides what is due.

Timeline for a signal session ``T`` and the next session ``N`` (both from Alpaca's calendar,
so holidays, half-days and DST need no special casing):

* submit window: ``T.close + submit_after_close`` .. ``N.open - submit_before_open``
  (20:15 → 09:22 ET on a normal day; Alpaca accepts OPG orders from 19:00 ET until 09:28 ET).
* reconcile: from ``N.open + reconcile_after_open`` onwards.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from shunya.algorithm.portfolio_manager import allocate_proportional_by_request, sum_target_maps
from shunya.algorithm.reconciliation import required_delta_shares
from shunya.algorithm.risk_engine import PortfolioRiskEngine, RiskVetConfig
from shunya.data.fints import finTs
from shunya.live import signals
from shunya.live.broker import Broker, KillSwitchEngaged, Session
from shunya.live.control import engage_kill, read_control
from shunya.live.store import Book, OrderRow, PostgresLiveStore, Run, Sleeve

_log = logging.getLogger(__name__)

TERMINAL_ORDER_STATES = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced", "stopped"}


class AutoHalt(RuntimeError):
    """Safety trigger: the runner engages the kill switch (level ``cancel``) with this reason."""


@dataclass(frozen=True)
class RunnerConfig:
    """Timing and safety thresholds (book ``risk_config`` can override the safety ones)."""

    # 20:15 ET on a normal day: late prints / corrections have settled into the daily bar.
    submit_after_close: timedelta = timedelta(hours=4, minutes=15)
    submit_before_open: timedelta = timedelta(minutes=8)
    reconcile_after_open: timedelta = timedelta(minutes=15)
    max_rejections: int = 3
    position_tolerance_shares: float = 0.5
    max_equity_jump: float = 0.25
    max_single_name_fraction: float = 0.10


FinTsLoader = Callable[[Sequence[str], date, int], finTs]


def _default_loader(tickers: Sequence[str], signal_date: date, lookback_days: int) -> finTs:
    return signals.load_daily_fin_ts(tickers, signal_date, lookback_days)


@dataclass(frozen=True)
class SessionPlan:
    """Last completed session ``T`` and the session after it ``N``."""

    signal: Session
    next: Session

    def submit_window(self, cfg: RunnerConfig) -> Tuple[datetime, datetime]:
        """``[start, end)`` for OPG submission."""
        return self.signal.close + cfg.submit_after_close, self.next.open - cfg.submit_before_open


def plan_sessions(broker: Broker, now: datetime) -> SessionPlan:
    """Pick ``T`` = last session closed by ``now`` and ``N`` = the session right after it."""
    sessions = broker.sessions(now.date() - timedelta(days=10), now.date() + timedelta(days=10))
    closed = [s for s in sessions if s.close <= now]
    if not closed:
        raise RuntimeError("no completed session in the calendar window")
    t = closed[-1]
    later = [s for s in sessions if s.day > t.day]
    if not later:
        raise RuntimeError("no upcoming session in the calendar window")
    return SessionPlan(signal=t, next=later[0])


def client_order_id(book_id: str, session_day: date, symbol: str) -> str:
    """Deterministic id so a retried run cannot double-submit (Alpaca rejects duplicates)."""
    return f"shy-{book_id.replace('-', '')[:8]}-{session_day:%Y%m%d}-{symbol}"[:48]


@dataclass
class OrderIntent:
    """Whole-share OPG order for one symbol."""

    symbol: str
    side: str
    qty: int
    client_order_id: str


@dataclass
class BookPlan:
    """Everything decided for a book before any order goes out."""

    equity: float
    book_capital: float
    sleeve_rows: List[Dict[str, Any]]
    target_shares: Dict[str, int]
    orders: List[OrderIntent]
    flags: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """JSON-safe overview for run ``detail`` and the CLI dry-run."""
        return {
            "equity": self.equity,
            "book_capital": self.book_capital,
            "n_orders": len(self.orders),
            "orders": [o.__dict__ for o in self.orders],
            "target_shares": self.target_shares,
            "flags": self.flags,
            "messages": self.messages,
        }


def _risk_cfg(book: Book, cfg: RunnerConfig) -> Dict[str, Any]:
    rc = dict(book.risk_config or {})
    return {
        "max_drawdown": rc.get("max_drawdown"),
        "max_equity_jump": float(rc.get("max_equity_jump", cfg.max_equity_jump)),
        "max_rejections": int(rc.get("max_rejections", cfg.max_rejections)),
        "max_single_name_fraction": float(rc.get("max_single_name_fraction", cfg.max_single_name_fraction)),
    }


def check_equity_guards(book: Book, equity: float, cfg: RunnerConfig) -> float:
    """Raise :class:`AutoHalt` on drawdown / implausible equity jump; return the new high-water mark."""
    rc = _risk_cfg(book, cfg)
    if book.last_equity and book.last_equity > 0:
        jump = abs(equity / book.last_equity - 1.0)
        if jump > rc["max_equity_jump"]:
            raise AutoHalt(f"equity moved {jump:.1%} since last run (limit {rc['max_equity_jump']:.0%})")
    hwm = max(float(book.hwm_equity or 0.0), equity)
    max_dd = rc["max_drawdown"]
    if max_dd and hwm > 0 and (hwm - equity) / hwm > float(max_dd):
        raise AutoHalt(f"drawdown {(hwm - equity) / hwm:.1%} exceeds {float(max_dd):.0%}")
    return hwm


def plan_book(
    book: Book,
    sleeves: Sequence[Sleeve],
    ledger: Mapping[str, Mapping[str, float]],
    broker: Broker,
    plan: SessionPlan,
    *,
    cfg: RunnerConfig,
    load_fin_ts: FinTsLoader = _default_loader,
) -> BookPlan:
    """Compute sleeve targets → net → risk vet → whole-share deltas vs Alpaca positions."""
    acct = broker.account()
    if acct.trading_blocked:
        raise AutoHalt(f"Alpaca account trading_blocked (status={acct.status})")
    equity = acct.equity
    book_capital = equity * book.capital_fraction
    signal_day = plan.signal.day

    live = [s for s in sleeves if s.status in ("active", "paused", "draining")]
    held = {sym for s in live for sym, q in ledger.get(s.id, {}).items() if abs(q) > 1e-9}
    need_data = sorted({t for s in live if s.status == "active" for t in s.version.tickers} | held)
    positions = {p.symbol: p for p in broker.positions()}

    closes: Dict[str, float] = {}
    fts: Optional[finTs] = None
    if need_data:
        lookback = max([s.version.lookback_days for s in live if s.status == "active"] or [30])
        fts = load_fin_ts(need_data, signal_day, lookback)
        closes = signals.closes_on(fts, signal_day)
    for sym in held:  # fall back to broker mark for held names without a bar
        if sym not in closes and sym in positions and positions[sym].current_price > 0:
            closes[sym] = positions[sym].current_price

    sleeve_usd: Dict[str, Dict[str, float]] = {}
    for s in live:
        prev = ledger.get(s.id, {})
        if s.status == "active":
            assert fts is not None
            strat = signals.build_strategy(fts, s.version.alpha_source, s.version.finstrat_config)
            w = signals.sleeve_weights(strat, signal_day, universe=s.version.tickers, long_only=True)
            ramp = min(1.0, (s.runs_completed + 1) / max(1, s.ramp_days))
            sleeve_usd[s.id] = {k: v * s.weight * ramp * book_capital for k, v in w.items()}
        elif s.status == "paused":  # hold current recorded shares
            sleeve_usd[s.id] = {k: q * closes.get(k, 0.0) for k, q in prev.items()}
        else:  # draining → flat
            sleeve_usd[s.id] = {}

    proposed = sum_target_maps(list(sleeve_usd.values()))
    universe = sorted(set(proposed) | held)
    missing_px = [k for k in universe if closes.get(k, 0.0) <= 0]
    if missing_px:
        raise signals.StaleDataError(f"no price for {missing_px[:10]}")

    rc = _risk_cfg(book, cfg)
    engine = PortfolioRiskEngine(
        RiskVetConfig(
            max_gross_fraction_of_equity=book.capital_fraction,
            max_single_name_fraction=rc["max_single_name_fraction"],
        )
    )
    current_usd = {k: positions[k].market_value for k in universe if k in positions}
    vet = engine.vet(
        proposed,
        current_usd=current_usd,
        equity_usd=equity,
        prices=closes,
        universe=universe,
        buying_power_usd=acct.buying_power,
    )
    vetted = {k: max(0.0, float(v)) for k, v in vet.targets_vetted.items()}  # long-only

    sleeve_rows: List[Dict[str, Any]] = []
    net_float: Dict[str, float] = {}
    for s in live:
        prev = ledger.get(s.id, {})
        usd = sleeve_usd.get(s.id, {})
        for sym in sorted(set(usd) | set(prev)):
            raw = usd.get(sym, 0.0)
            scale = vetted.get(sym, 0.0) / proposed[sym] if proposed.get(sym, 0.0) > 0 else 0.0
            t_usd = raw * scale if s.status == "active" else raw
            t_sh = t_usd / closes[sym]
            net_float[sym] = net_float.get(sym, 0.0) + t_sh
            sleeve_rows.append(
                {
                    "sleeve_id": s.id,
                    "symbol": sym,
                    "target_usd": t_usd,
                    "target_shares": t_sh,
                    "prev_shares": float(prev.get(sym, 0.0)),
                }
            )

    # 1e-3 share tolerance: float32 alpha output can land a hair under a whole share
    target_shares = {k: int(math.floor(max(0.0, net_float.get(k, 0.0)) + 1e-3)) for k in universe}
    settled = {k: positions[k].qty for k in universe if k in positions}
    wb: Dict[str, float] = {}
    ws: Dict[str, float] = {}
    for o in broker.open_orders():
        if o.symbol in target_shares:
            rem = o.qty - o.filled_qty
            (wb if o.side == "buy" else ws)[o.symbol] = (wb if o.side == "buy" else ws).get(o.symbol, 0.0) + rem
    deltas = required_delta_shares(target_shares, settled, wb, ws, universe)

    orders: List[OrderIntent] = []
    for sym in universe:
        q = int(round(deltas[sym]))
        if q == 0:
            continue
        side = "buy" if q > 0 else "sell"
        if side == "sell":  # long-only: never sell through zero
            q = -min(-q, int(settled.get(sym, 0.0) - ws.get(sym, 0.0)))
            if q >= 0:
                continue
        orders.append(OrderIntent(sym, side, abs(q), client_order_id(book.id, plan.next.day, sym)))

    return BookPlan(
        equity=equity,
        book_capital=book_capital,
        sleeve_rows=sleeve_rows,
        target_shares=target_shares,
        orders=orders,
        flags=list(vet.flags),
        messages=list(vet.messages),
    )


def submit_book(
    store: PostgresLiveStore,
    broker: Broker,
    book: Book,
    plan: SessionPlan,
    run_id: str,
    *,
    cfg: RunnerConfig,
    load_fin_ts: FinTsLoader = _default_loader,
) -> Dict[str, Any]:
    """Plan, persist targets, submit OPG orders through the guarded broker; mark run ``submitted``."""
    acct = broker.account()
    hwm = check_equity_guards(book, acct.equity, cfg)
    store.update_book_equity(book.id, equity=acct.equity, hwm=hwm)

    sleeves = store.list_sleeves(book.id)
    ledger = store.get_sleeve_positions(book.id)
    bp = plan_book(book, sleeves, ledger, broker, plan, cfg=cfg, load_fin_ts=load_fin_ts)
    store.save_sleeve_targets(run_id, bp.sleeve_rows)

    rejections: List[str] = []
    for o in bp.orders:
        try:
            v = broker.submit_opg(symbol=o.symbol, side=o.side, qty=o.qty, client_order_id=o.client_order_id)
            store.upsert_order(
                OrderRow(run_id, o.client_order_id, o.symbol, o.side, o.qty, v.status, alpaca_id=v.alpaca_id)
            )
        except KillSwitchEngaged:
            raise
        except Exception as exc:  # noqa: BLE001 — record and keep going; halt below if too many
            rejections.append(f"{o.symbol}: {exc}")
            store.upsert_order(OrderRow(run_id, o.client_order_id, o.symbol, o.side, o.qty, "rejected", error=str(exc)))

    store.increment_sleeve_runs([s.id for s in sleeves if s.status == "active"])
    summary = {**bp.summary(), "rejections": rejections}
    store.update_run(run_id, status="submitted", equity=acct.equity, detail=summary)
    if len(rejections) >= _risk_cfg(book, cfg)["max_rejections"]:
        raise AutoHalt(f"{len(rejections)} order rejections in run {run_id}")
    return summary


def allocate_fills(
    targets: Sequence[Mapping[str, Any]], net_fill_by_symbol: Mapping[str, float]
) -> Dict[str, Dict[str, float]]:
    """New sleeve ledgers: ``prev + share of fill``.

    When the fill equals the sum of sleeve requests, each sleeve gets exactly its request
    (internal crosses between sleeves net out without a broker trade); otherwise the fill is
    split in proportion to requests.
    """
    by_sym: Dict[str, Dict[str, float]] = {}
    prev: Dict[str, Dict[str, float]] = {}
    for r in targets:
        sid, sym = str(r["sleeve_id"]), str(r["symbol"])
        by_sym.setdefault(sym, {})[sid] = float(r["target_shares"]) - float(r["prev_shares"])
        prev.setdefault(sid, {})[sym] = float(r["prev_shares"])
    out: Dict[str, Dict[str, float]] = {sid: dict(m) for sid, m in prev.items()}
    for sym, req in by_sym.items():
        fill = float(net_fill_by_symbol.get(sym, 0.0))
        alloc = req if abs(fill - sum(req.values())) < 1e-6 else allocate_proportional_by_request(fill, req)
        for sid, q in alloc.items():
            out[sid][sym] = out[sid].get(sym, 0.0) + q
    return out


def reconcile_book(
    store: PostgresLiveStore, broker: Broker, book: Book, run: Run, *, cfg: RunnerConfig
) -> Optional[Dict[str, Any]]:
    """After the open: record fills, update sleeve ledgers, verify they sum to Alpaca positions."""
    if not store.claim_reconcile(run.id):
        return None
    orders = store.list_orders(run.id)
    net_fill: Dict[str, float] = {}
    pending: List[str] = []
    for o in orders:
        if o.status == "rejected" and not o.alpaca_id:
            continue
        v = broker.order_by_client_id(o.client_order_id)
        if v is None:
            continue
        store.upsert_order(
            OrderRow(
                run.id, o.client_order_id, o.symbol, o.side, o.qty, v.status, alpaca_id=v.alpaca_id,
                filled_qty=v.filled_qty, filled_avg_price=v.filled_avg_price,
            )
        )
        if v.status not in TERMINAL_ORDER_STATES:
            pending.append(o.symbol)
        signed = v.filled_qty if o.side == "buy" else -v.filled_qty
        net_fill[o.symbol] = net_fill.get(o.symbol, 0.0) + signed
    if pending:
        store.update_run(run.id, status="submitted", detail={"pending_orders": pending})
        return None

    ledgers = allocate_fills(store.list_sleeve_targets(run.id), net_fill)
    positions = {p.symbol: p for p in broker.positions()}
    for sid, shares in ledgers.items():
        store.set_sleeve_positions(sid, shares)
        mv = sum(q * positions[sym].current_price for sym, q in shares.items() if sym in positions)
        store.save_snapshot(sid, run.session_date, mv)

    all_ledgers = store.get_sleeve_positions(book.id)
    tracked: Dict[str, float] = {}
    for m in all_ledgers.values():
        for sym, q in m.items():
            tracked[sym] = tracked.get(sym, 0.0) + q
    mismatches = {
        sym: {"ledger": q, "broker": positions[sym].qty if sym in positions else 0.0}
        for sym, q in tracked.items()
        if abs(q - (positions[sym].qty if sym in positions else 0.0)) > cfg.position_tolerance_shares
    }
    detail = {"fills": net_fill, "mismatches": mismatches}
    store.update_run(run.id, status="reconciled", detail=detail)
    if mismatches:
        raise AutoHalt(f"sleeve ledger != broker positions for {sorted(mismatches)[:10]}")
    return detail


def _ping_heartbeat() -> None:
    url = os.environ.get("SHUNYA_LIVE_HEARTBEAT_URL", "").strip()
    if not url:
        return
    try:
        urllib.request.urlopen(url, timeout=5).close()  # noqa: S310 — operator-configured URL
    except Exception as exc:  # noqa: BLE001
        _log.warning("heartbeat ping failed: %s", exc)


def tick(
    store: PostgresLiveStore,
    broker: Broker,
    *,
    now: Optional[datetime] = None,
    cfg: RunnerConfig = RunnerConfig(),
    load_fin_ts: FinTsLoader = _default_loader,
) -> Dict[str, Any]:
    """One scheduler firing: reconcile due runs, then submit if inside the window. Idempotent."""
    now = now or datetime.now(timezone.utc)
    result: Dict[str, Any] = {"now": now.isoformat(), "books": {}}
    try:
        ctl = read_control(store)
        if ctl.halted:
            result["status"] = "halted"
            return result
        plan = plan_sessions(broker, now)
        start, end = plan.submit_window(cfg)
        in_window = start <= now < end
        result.update(
            signal_date=plan.signal.day.isoformat(),
            next_session=plan.next.day.isoformat(),
            submit_window=[start.isoformat(), end.isoformat()],
            in_submit_window=in_window,
        )
        sessions = {s.day: s for s in broker.sessions(now.date() - timedelta(days=10), now.date())}
        for book in store.list_books(active_only=True):
            br: Dict[str, Any] = {}
            result["books"][book.name] = br
            try:
                for run in store.list_runs(book_id=book.id, status="submitted"):
                    sess = sessions.get(run.session_date)
                    if sess is not None and now >= sess.open + cfg.reconcile_after_open:
                        br["reconciled"] = reconcile_book(store, broker, book, run, cfg=cfg) is not None
                unreconciled = [
                    r.session_date.isoformat()
                    for st in ("submitted", "reconciling")
                    for r in store.list_runs(book_id=book.id, status=st)
                    if r.session_date != plan.next.day
                ]
                if in_window and unreconciled:  # ledger would be stale; wait for reconcile
                    br["blocked_by_unreconciled"] = unreconciled
                elif in_window:
                    run_id = store.claim_run(book_id=book.id, session_date=plan.next.day, signal_date=plan.signal.day)
                    if run_id is not None:
                        try:
                            br["submitted"] = submit_book(
                                store, broker, book, plan, run_id, cfg=cfg, load_fin_ts=load_fin_ts
                            )["n_orders"]
                        except Exception as exc:
                            store.update_run(run_id, status="failed", error=str(exc)[:2000])
                            raise
            except KillSwitchEngaged as exc:
                br["error"] = f"kill switch engaged mid-run: {exc}"
            except (AutoHalt, signals.StaleDataError) as exc:
                br["error"] = f"auto-halt: {exc}"
                engage_kill(store, broker, level="cancel", reason=f"auto: {exc}", actor="runner")
            except Exception as exc:  # noqa: BLE001 — one book must not block the others
                _log.exception("book %s failed", book.name)
                br["error"] = str(exc)
        result["status"] = "error" if any("error" in b for b in result["books"].values()) else "ok"
        return result
    except Exception as exc:  # noqa: BLE001
        _log.exception("tick failed")
        result["status"] = "error"
        result["error"] = str(exc)
        return result
    finally:
        try:
            store.write_heartbeat(result.get("status", "error"))
            if result.get("status") in ("ok", "halted"):
                _ping_heartbeat()
        except Exception as exc:  # noqa: BLE001
            _log.error("heartbeat write failed: %s", exc)
