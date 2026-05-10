"""Wrap and unwrap alpha ``source_code`` for body-only editors.

Persisted modules must define ``alpha(ctx)`` (see ``api.inline_alpha``). The
canonical wrapper injects ``ts``, ``cs``, and ``fun`` aliases inside the function.
"""

from __future__ import annotations

import ast
import textwrap

# Lines before the user-authored body (1-based: body starts at line 7).
HEADER_LINE_COUNT = 6

_CANONICAL_PREFIX = """import jax.numpy as jnp

def alpha(ctx) -> jnp.ndarray:
    ts = ctx.ts
    cs = ctx.cs
    fun = ctx.fun
"""


def wrap_alpha_body(body: str) -> str:
    """Return a full module string from editor-visible function body text."""
    raw = body.replace("\r\n", "\n").strip("\n")
    if not raw.strip():
        inner = "    pass\n"
    else:
        inner = textwrap.indent(raw, "    ") + "\n"
    return _CANONICAL_PREFIX + inner


def _is_ts_cs_fun_setup(stmts: list[ast.stmt]) -> bool:
    if len(stmts) < 3:
        return False
    names = ("ts", "cs", "fun")
    attrs = ("ts", "cs", "fun")
    for i, name, attr in zip(range(3), names, attrs, strict=True):
        s = stmts[i]
        if not isinstance(s, ast.Assign) or len(s.targets) != 1:
            return False
        t = s.targets[0]
        if not isinstance(t, ast.Name) or t.id != name:
            return False
        v = s.value
        if not isinstance(v, ast.Attribute):
            return False
        if not isinstance(v.value, ast.Name) or v.value.id != "ctx":
            return False
        if v.attr != attr:
            return False
    return True


def unwrap_alpha_source(source: str) -> str:
    """
    Extract editor-visible body from stored ``source_code``.

    - If the module matches the canonical wrapper, strip the prefix and dedent.
    - Else parse ``def alpha`` and return the function body (optionally skipping
      a leading ``ts``/``cs``/``fun`` setup block), dedented.
    - If there is no ``alpha`` function, treat the entire source as body text.
    """
    text = source.replace("\r\n", "\n")
    if text.startswith(_CANONICAL_PREFIX):
        rest = text[len(_CANONICAL_PREFIX) :]
        if rest.startswith("\n"):
            rest = rest[1:]
        return _dedent_body_block(rest)

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return textwrap.dedent(text).strip("\n")

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "alpha":
            body_stmts = node.body
            start_idx = 3 if _is_ts_cs_fun_setup(body_stmts[:3]) else 0
            inner = body_stmts[start_idx:]
            if not inner:
                return ""
            lines = text.splitlines()
            start_line = inner[0].lineno
            end_line = inner[-1].end_lineno or inner[-1].lineno
            chunk = "\n".join(lines[start_line - 1 : end_line])
            return textwrap.dedent(chunk).strip("\n")

    return textwrap.dedent(text).strip("\n")


def _dedent_body_block(block: str) -> str:
    """Remove one level of 4-space indent from each line."""
    lines = block.splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith("    "):
            out.append(line[4:])
        elif line == "":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def body_line_to_wrapped_line(body_line_1based: int) -> int:
    """Map a 1-based line number in the body editor to the wrapped module."""
    return HEADER_LINE_COUNT + body_line_1based


def wrapped_line_to_body_line(wrapped_line_1based: int) -> int | None:
    """Map a 1-based line in the persisted wrapped module to body line, or None if in header."""
    if wrapped_line_1based <= HEADER_LINE_COUNT:
        return None
    return wrapped_line_1based - HEADER_LINE_COUNT


# Lines before user body in the Pyright-only augmented file (canonical 6 + typing import).
LINT_HEADER_LINE_COUNT = HEADER_LINE_COUNT + 1


def lint_wrapped_line_to_body_line(wrapped_line_1based: int) -> int | None:
    """Map a line in the Pyright augmented module to editor body line."""
    if wrapped_line_1based <= LINT_HEADER_LINE_COUNT:
        return None
    return wrapped_line_1based - LINT_HEADER_LINE_COUNT
