from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from shunya.algorithm.alpha_context import AlphaContext

from backtest_api.alpha_validation import validate_import_ref
from backtest_api.inline_alpha import resolve_alpha_from_source


def resolve_alpha(import_ref: str) -> Callable[[AlphaContext], Any]:
    validate_import_ref(import_ref)
    mod_name, _, attr = import_ref.partition(":")
    if not mod_name or not attr:
        raise ValueError("import_ref must be like module.path:attribute")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Attribute {attr!r} not found or not callable on {mod_name!r}")
    return fn


def resolve_alpha_for_backtest(
    import_ref: str | None,
    source_code: str | None,
) -> Callable[[AlphaContext], Any]:
    """
    If ``source_code`` is non-empty, load ``alpha`` from that string; else resolve ``import_ref``.
    """
    code = (source_code or "").strip()
    if code:
        return resolve_alpha_from_source(code)
    ref = (import_ref or "").strip()
    if not ref:
        raise ValueError("alpha has no import_ref and no source_code")
    return resolve_alpha(ref)
