"""Wire :class:`~shunya.algorithm.risk_engine.PortfolioRiskEngine` vet output into the OMS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from shunya.algorithm.risk_engine import RiskVetResult

from shunya.oms.service import InstitutionalOMS


def ingest_risk_vet_result_usd(
    vet: "RiskVetResult",
) -> Mapping[str, float]:
    """Return vetted USD targets map suitable for :meth:`InstitutionalOMS.propose_parent_intents`."""
    return dict(vet.targets_vetted)


def sync_oms_from_vet_and_prices(
    oms: InstitutionalOMS,
    vet: "RiskVetResult",
    prices: Mapping[str, float],
    universe: Sequence[str],
) -> list:
    """
    Convenience: ``vet`` → share reconciliation → :class:`~shunya.oms.service.ParentIntent` list.

    Does not create :class:`~shunya.oms.parent_fsm.ParentOrder` rows; callers still
    invoke :meth:`~shunya.oms.service.InstitutionalOMS.create_parent_order` and EMS.
    """
    return oms.propose_parent_intents(vet.targets_vetted, prices, universe)
