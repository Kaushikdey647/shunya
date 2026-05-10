"""Shared markdown context for alpha AI endpoints (DSL + pipeline + appendix)."""

from __future__ import annotations

import json
from pathlib import Path

_API_DIR = Path(__file__).resolve().parent

_APPENDIX = _API_DIR / "alpha_assist_appendix.md"
_DSL = _API_DIR / "alpha_dsl_reference.md"
_PIPELINE = _API_DIR / "backtest_pipeline_summary.md"


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def build_shared_user_context(
    *,
    alpha_name: str | None,
    alpha_description: str | None,
    source_body: str,
    wrapped_source: str,
    extra_sections: dict[str, str] | None = None,
) -> str:
    """Assemble user-side prompt blocks (static docs + alpha state)."""
    parts: list[str] = [
        "### Static: product appendix\n" + _read(_APPENDIX),
        "### Static: alpha DSL reference\n" + _read(_DSL),
        "### Static: backtest pipeline\n" + _read(_PIPELINE),
        "### Alpha metadata\n"
        + json.dumps({"name": alpha_name, "description": alpha_description}, indent=2),
        "### Wrapped module\n```python\n" + wrapped_source.strip() + "\n```",
        "### Body only (line anchors refer to this)\n```python\n" + source_body.strip() + "\n```",
    ]
    if extra_sections:
        for title, body in extra_sections.items():
            parts.append(f"### {title}\n{body}")
    return "\n\n".join(parts)
