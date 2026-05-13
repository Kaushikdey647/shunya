"""Institutional paper desk: PCS (or fixed targets) → risk → OMS → EMS + Alpaca stream."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from shunya.algorithm.execution import AlpacaExecutionAdapter
from shunya.algorithm.portfolio_manager import PortfolioConstructionService
from shunya.algorithm.risk_engine import PortfolioRiskEngine, RiskVetConfig
from shunya.ems.broker_gateway import AlpacaBrokerGateway
from shunya.ems.runner import EMSParentRunner
from shunya.ems.schedules import twap_slice_quantities
from shunya.integration.alpaca_settings import AlpacaRuntimeSettings
from shunya.live.prices import close_prices_at, fin_ts_from_portfolio_construction
from shunya.oms.alpaca_stream import AlpacaOMSTradeStream
from shunya.oms.service import InstitutionalOMS

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient


@dataclass
class PaperCycleResult:
    """Structured outcome of one paper / live EMS cycle."""

    correlation_id: str
    blend_mode: Optional[str]
    tickers: Tuple[str, ...]
    targets_constructed: Dict[str, float]
    targets_vetted: Dict[str, float]
    parent_intents: List[Dict[str, Any]]
    ems_parent_ids: List[str]
    messages: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "blend_mode": self.blend_mode,
            "tickers": list(self.tickers),
            "targets_constructed": dict(self.targets_constructed),
            "targets_vetted": dict(self.targets_vetted),
            "parent_intents": list(self.parent_intents),
            "ems_parent_ids": list(self.ems_parent_ids),
            "messages": list(self.messages),
        }


class InstitutionalPaperDesk:
    """
    One coordinator for: portfolio construction (optional), risk vet, OMS share
    reconciliation, Alpaca trade stream, and EMS TWAP-style child limits.

    **Prices:** when using :meth:`run_with_pcs`, per-symbol ``Close`` is taken from the
    book's shared ``finTs`` at ``execution_date`` via :func:`~shunya.live.prices.close_prices_at`.

    **OMS / EMS client_order_id:** children use ``{parent_id}:{slice}[u{n}]``; the stream
    extracts the parent root so fills attach to :class:`~shunya.oms.parent_fsm.ParentOrder`.
    """

    def __init__(
        self,
        trading_client: "TradingClient",
        data_client: "StockHistoricalDataClient",
        alpaca_settings: AlpacaRuntimeSettings,
        *,
        risk_engine: Optional[PortfolioRiskEngine] = None,
        twap_bins: int = 4,
        ems_interval_seconds: float = 2.0,
        ems_child_timeout_seconds: float = 15.0,
        stream_warmup_seconds: float = 0.75,
        post_ems_settle_seconds: float = 2.0,
        require_market_open: bool = False,
    ) -> None:
        self._tc = trading_client
        self._dc = data_client
        self._settings = alpaca_settings
        self._risk = risk_engine or PortfolioRiskEngine(RiskVetConfig(max_gross_fraction_of_equity=0.995))
        self._twap_bins = max(1, int(twap_bins))
        self._ems_interval = float(ems_interval_seconds)
        self._ems_child_timeout = float(ems_child_timeout_seconds)
        self._stream_warmup = float(stream_warmup_seconds)
        self._post_ems = float(post_ems_settle_seconds)
        self._require_open = bool(require_market_open)

    async def run_with_pcs(
        self,
        pcs: PortfolioConstructionService,
        *,
        capital: float,
        execution_date: pd.Timestamp,
        correlation_id: str,
        group_column: Optional[str] = "Sector",
        record_raw_history: bool = True,
    ) -> PaperCycleResult:
        """Run :meth:`PortfolioConstructionService.construct` then execution leg."""
        pcr = pcs.construct(
            capital=float(capital),
            execution_date=execution_date,
            group_column=group_column,
            record_raw_history=record_raw_history,
        )
        fts = fin_ts_from_portfolio_construction(pcs)
        prices = close_prices_at(fts, pcr.tickers, execution_date)
        return await self._run_execution_leg(
            targets_raw=dict(pcr.targets),
            tickers=tuple(str(t) for t in pcr.tickers),
            prices=prices,
            blend_mode=str(pcr.blend_mode),
            correlation_id=correlation_id,
        )

    async def run_with_targets(
        self,
        targets_usd: Mapping[str, float],
        *,
        universe: Sequence[str],
        prices: Mapping[str, float],
        correlation_id: str,
    ) -> PaperCycleResult:
        """Skip PCS; treat ``targets_usd`` as proposed USD notionals (e.g. HTTP API)."""
        uni = tuple(str(u) for u in universe)
        raw = {str(k): float(v) for k, v in targets_usd.items()}
        return await self._run_execution_leg(
            targets_raw=raw,
            tickers=uni,
            prices={str(k): float(v) for k, v in prices.items()},
            blend_mode=None,
            correlation_id=correlation_id,
        )

    async def _run_execution_leg(
        self,
        *,
        targets_raw: Dict[str, float],
        tickers: Tuple[str, ...],
        prices: Dict[str, float],
        blend_mode: Optional[str],
        correlation_id: str,
    ) -> PaperCycleResult:
        messages: List[str] = []
        adapter = AlpacaExecutionAdapter(self._tc)
        if self._require_open:
            adapter.assert_market_open()

        uni_list = list(tickers)
        current_raw = adapter.get_positions()
        current_usd = {s: float(current_raw.get(s, 0.0)) for s in uni_list}

        acct = self._tc.get_account()
        equity = float(getattr(acct, "equity", 0.0) or 0.0)
        bp = adapter.buying_power()

        vet = self._risk.vet(
            targets_raw,
            current_usd=current_usd,
            equity_usd=equity,
            prices=prices,
            universe=uni_list,
            buying_power_usd=bp,
        )
        if vet.messages:
            messages.extend(vet.messages)

        oms = InstitutionalOMS()
        oms.refresh_settled_shares_from_alpaca(self._tc)
        intents = oms.propose_parent_intents(vet.targets_vetted, prices, uni_list)

        intent_payload = [
            {"parent_id": i.parent_id, "symbol": i.symbol, "side": i.side, "quantity": i.quantity}
            for i in intents
        ]
        ems_ids: List[str] = []

        stream = AlpacaOMSTradeStream(settings=self._settings, oms=oms)
        loop = asyncio.get_running_loop()
        stream.start_background(loop)
        await asyncio.sleep(self._stream_warmup)

        gateway = AlpacaBrokerGateway(self._tc, self._dc)
        try:
            for intent in intents:
                parent = oms.create_parent_order(intent)
                ems_ids.append(parent.parent_id)
                slices = twap_slice_quantities(intent.quantity, self._twap_bins)
                runner = EMSParentRunner(
                    gateway=gateway,
                    parent_id=intent.parent_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    slice_quantities=slices,
                    interval_seconds=self._ems_interval,
                    child_timeout_seconds=self._ems_child_timeout,
                )
                await runner.run()
        finally:
            await asyncio.sleep(self._post_ems)
            await stream.stop()
            oms.refresh_settled_shares_from_alpaca(self._tc)

        return PaperCycleResult(
            correlation_id=correlation_id,
            blend_mode=blend_mode,
            tickers=tickers,
            targets_constructed=dict(targets_raw),
            targets_vetted=dict(vet.targets_vetted),
            parent_intents=intent_payload,
            ems_parent_ids=ems_ids,
            messages=messages,
        )


def new_correlation_id(prefix: str = "paper") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"
