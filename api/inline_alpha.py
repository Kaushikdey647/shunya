"""Compile trusted inline alpha source into a callable ``alpha(ctx)`` (implementation lives in the library)."""
from __future__ import annotations

from shunya.algorithm.inline_alpha import resolve_alpha_from_source

__all__ = ["resolve_alpha_from_source"]
