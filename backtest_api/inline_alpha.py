from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

# Same trust model as import_ref: only trusted inputs (see README).
_MAX_SOURCE_BYTES = 512 * 1024


def resolve_alpha_from_source(source: str) -> Callable[[Any], Any]:
    """
    Load ``alpha(ctx)`` from a Python source string.
    Namespace exposes ``jnp`` (``jax.numpy``) for user code.
    """
    if not source or not str(source).strip():
        raise ValueError("source_code is empty")
    code = str(source)
    if len(code.encode("utf-8")) > _MAX_SOURCE_BYTES:
        raise ValueError(f"source_code exceeds max size ({_MAX_SOURCE_BYTES} bytes)")

    ns: dict[str, Any] = {"jnp": jnp, "__builtins__": __builtins__}
    try:
        compiled = compile(code, "<alpha_source>", "exec")
        exec(compiled, ns, ns)  # noqa: S102 — trusted service input; see module docstring
    except SyntaxError as exc:
        raise ValueError(f"source_code syntax error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"source_code could not be executed: {exc}") from exc

    alpha_fn = ns.get("alpha")
    if alpha_fn is None:
        raise ValueError("source_code must define a top-level function named alpha")
    if not callable(alpha_fn):
        raise ValueError("alpha must be callable")
    return alpha_fn  # type: ignore[return-value]
