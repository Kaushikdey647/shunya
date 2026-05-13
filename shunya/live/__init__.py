"""Live / paper trading desk orchestration."""

from shunya.live.cli import main as cli_main
from shunya.live.demo import build_demo_target_blend_pcs, make_minimal_fints
from shunya.live.desk import InstitutionalPaperDesk, PaperCycleResult, new_correlation_id
from shunya.live.prices import close_prices_at, fin_ts_from_portfolio_construction

__all__ = [
    "InstitutionalPaperDesk",
    "PaperCycleResult",
    "build_demo_target_blend_pcs",
    "cli_main",
    "close_prices_at",
    "fin_ts_from_portfolio_construction",
    "make_minimal_fints",
    "new_correlation_id",
]
