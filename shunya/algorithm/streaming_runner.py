from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Optional, Sequence

import numpy as np

from ..streaming.events import MarketEvent
from ..streaming.metrics import StreamingMetrics
from ..streaming.snapshot import MicroBarAggregator, SnapshotBuilder, StreamingSnapshot
from ..streaming.state import StreamingState
from ..streaming.universe import UniverseCandidate, UniverseSelector
from .order_manager import ManagedOrderBatch, OrderManager
from .streaming_context import StreamingContextBuilder
from .finstrat import FinStrat


@dataclass
class StreamingDecision:
    snapshot: StreamingSnapshot
    targets_usd: dict[str, float]
    order_batch: ManagedOrderBatch


class StreamingRunner:
    """Tick-to-trade orchestration that reuses `FinStrat` and the execution adapters."""

    def __init__(
        self,
        fin_strat: FinStrat,
        order_manager: OrderManager,
        *,
        lookback: int,
        bar_interval: object = "1s",
        fill_policy: str = "ffill",
        max_concurrent_symbols: Optional[int] = None,
        metrics: Optional[StreamingMetrics] = None,
    ) -> None:
        self._fin_strat = fin_strat
        self._order_manager = order_manager
        self._state = StreamingState(lookback)
        self._aggregator = MicroBarAggregator(bar_interval=bar_interval, lookback=lookback)
        self._snapshot_builder = SnapshotBuilder(lookback=lookback, fill_policy=fill_policy)
        self._max_concurrent_symbols = max_concurrent_symbols
        self.metrics = metrics or StreamingMetrics()

    def set_target_universe(
        self,
        candidates: Sequence[str] | Sequence[UniverseCandidate],
        *,
        scores: Optional[Mapping[str, float]] = None,
    ) -> list[str]:
        if self._max_concurrent_symbols is None:
            chosen = [c.symbol if isinstance(c, UniverseCandidate) else str(c) for c in candidates]
        else:
            chosen = UniverseSelector.select(
                candidates,
                limit=self._max_concurrent_symbols,
                scores=scores,
            )
        return self._state.set_active_symbols(chosen)

    def active_symbols(self) -> list[str]:
        return self._state.active_symbols()

    def on_event(
        self,
        event: MarketEvent,
        *,
        capital: float,
        group_labels: Optional[Mapping[str, object]] = None,
        dry_run: bool = False,
        recompute_on: str = "bar_close",
    ) -> Optional[StreamingDecision]:
        if self._state.active_symbols() and event.symbol not in self._state.active_symbols():
            self.metrics.dropped_events += 1
            return None

        self.metrics.events_seen += 1
        self._state.ingest(event)
        buf = self._state.buffer(event.symbol)
        if buf is not None:
            self.metrics.mark_queue_depth(event.symbol, len(buf))
        completed = self._aggregator.observe(event)
        if recompute_on == "tick":
            return self.evaluate(capital=capital, group_labels=group_labels, dry_run=dry_run)
        if recompute_on == "bar_close" and completed is not None:
            return self.evaluate(capital=capital, group_labels=group_labels, dry_run=dry_run)
        return None

    def evaluate(
        self,
        *,
        capital: float,
        group_labels: Optional[Mapping[str, object]] = None,
        dry_run: bool = False,
    ) -> StreamingDecision:
        symbols = self._state.active_symbols()
        if not symbols:
            raise ValueError("active universe is empty")

        snapshot = self._snapshot_builder.build(self._aggregator, symbols)
        ctx = StreamingContextBuilder.build(snapshot)

        alpha_start = perf_counter()
        raw = self._fin_strat.scores_from_context(ctx)
        group_ids = None
        n = self._fin_strat.neutralization
        if n == "group":
            if group_labels is None:
                raise ValueError("group_labels are required for neutralization='group'")
            group_ids = [group_labels[symbol] for symbol in symbols]
        elif n in ("sector", "industry"):
            if group_labels is not None:
                raise ValueError(
                    "group_labels must be omitted for neutralization='sector' or 'industry' "
                    "(labels are read from fin_ts.df)."
                )
        notionals = self._fin_strat.process_raw_scores(
            raw,
            capital,
            tickers=snapshot.symbols,
            execution_date=snapshot.timestamps[-1],
            group_ids=group_ids,
        )
        self.metrics.alpha_runs += 1
        self.metrics.last_alpha_latency_ms = (perf_counter() - alpha_start) * 1000.0

        targets = {
            symbol: float(notionals[idx])
            for idx, symbol in enumerate(snapshot.symbols)
        }
        latest_close = np.asarray(snapshot.close[-1, :], dtype=float)
        prices = {
            symbol: float(latest_close[idx])
            for idx, symbol in enumerate(snapshot.symbols)
            if np.isfinite(latest_close[idx]) and float(latest_close[idx]) > 0
        }

        order_start = perf_counter()
        batch = self._order_manager.submit_targets(
            targets,
            prices=prices,
            correlation_id=f"stream-{int(snapshot.timestamps[-1].value)}",
            dry_run=dry_run,
        )
        self.metrics.orders_submitted += len(batch.order_attempts)
        self.metrics.last_order_latency_ms = (perf_counter() - order_start) * 1000.0
        return StreamingDecision(snapshot=snapshot, targets_usd=targets, order_batch=batch)
