from __future__ import annotations

from api.resolver import resolve_alpha


def test_resolve_sma_ratio_alpha() -> None:
    fn = resolve_alpha("examples.alphas.sma_ratio_20:alpha")
    assert callable(fn)
    # Smoke: would need real finTs context to call; module import is enough here.
