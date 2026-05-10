from __future__ import annotations

from shunya.algorithm.alpha_source_wrap import (
    HEADER_LINE_COUNT,
    LINT_HEADER_LINE_COUNT,
    lint_wrapped_line_to_body_line,
    unwrap_alpha_source,
    wrap_alpha_body,
)


def test_wrap_unwrap_round_trip() -> None:
    body = "return cs.rank(ctx.close)\nx = 1"
    full = wrap_alpha_body(body)
    assert "def alpha(ctx)" in full
    assert "ts = ctx.ts" in full
    assert unwrap_alpha_source(full) == body


def test_wrap_empty_body_uses_pass() -> None:
    full = wrap_alpha_body("   \n  ")
    assert "pass" in full
    assert unwrap_alpha_source(full).strip() == "pass"


def test_unwrap_legacy_full_module() -> None:
    legacy = """import jax.numpy as jnp

def alpha(ctx) -> jnp.ndarray:
    return ctx.cs.rank(ctx.close)
"""
    out = unwrap_alpha_source(legacy)
    assert "def alpha" not in out
    assert "return ctx.cs.rank" in out or "return" in out


def test_lint_line_mapping() -> None:
    assert lint_wrapped_line_to_body_line(LINT_HEADER_LINE_COUNT + 1) == 1
    assert lint_wrapped_line_to_body_line(1) is None
    assert HEADER_LINE_COUNT == 6
    assert LINT_HEADER_LINE_COUNT == 7
