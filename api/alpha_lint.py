"""Run Pyright on a wrapped alpha module and return diagnostics mapped to body lines."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from shunya.algorithm.alpha_source_wrap import lint_wrapped_line_to_body_line

_log = logging.getLogger(__name__)

_API_DIR = Path(__file__).resolve().parent


def augment_wrapped_source_for_pyright(wrapped: str) -> str:
    """Insert stub import and annotate ``ctx`` for Pyright (not used at runtime)."""
    w = wrapped.replace("\r\n", "\n")
    if not w.endswith("\n"):
        w += "\n"
    if "from alpha_editor_stubs import AlphaContext" in w:
        out = w
    else:
        lines = w.split("\n", 1)
        first = lines[0]
        tail = lines[1] if len(lines) > 1 else ""
        out = first + "\nfrom alpha_editor_stubs import AlphaContext\n" + tail
    return out.replace(
        "def alpha(ctx) -> jnp.ndarray:",
        "def alpha(ctx: AlphaContext) -> jnp.ndarray:",
        1,
    )


def run_pyright_on_wrapped(wrapped_source: str) -> list[dict[str, Any]]:
    """
    Run Pyright on augmented source; return diagnostics with ``startLineNumber`` etc.
    in **body** coordinates (1-based), dropping issues that only live in the header.
    """
    augmented = augment_wrapped_source_for_pyright(wrapped_source)
    pyright = shutil.which("pyright")
    if pyright is None:
        _log.warning("pyright executable not found on PATH")
        return []

    extra = str(_API_DIR)
    cfg = {
        "pythonVersion": "3.12",
        "include": ["alpha_lint.py"],
        "extraPaths": [extra],
        "reportMissingImports": "none",
        "reportMissingModuleSource": "none",
        "typeCheckingMode": "basic",
    }

    diagnostics: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="shunya_alpha_lint_") as tmp:
        tdir = Path(tmp)
        (tdir / "pyrightconfig.json").write_text(json.dumps(cfg), encoding="utf-8")
        (tdir / "alpha_lint.py").write_text(augmented, encoding="utf-8")
        proc = subprocess.run(
            [pyright, "--outputjson", str(tdir / "alpha_lint.py")],
            cwd=str(tdir),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if not proc.stdout.strip():
            if proc.stderr:
                _log.warning("pyright stderr: %s", proc.stderr[:500])
            return []
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            _log.warning("pyright stdout not json: %s", proc.stdout[:500])
            return []

        for item in payload.get("generalDiagnostics", []):
            sev = item.get("severity")
            msg = item.get("message", "")
            rng = item.get("range") or {}
            start = rng.get("start") or {}
            line = int(start.get("line", 0)) + 1  # pyright 0-based line
            col = int(start.get("character", 0)) + 1
            end = rng.get("end") or start
            end_line = int(end.get("line", 0)) + 1
            end_col = int(end.get("character", 0)) + 1

            body_line = lint_wrapped_line_to_body_line(line)
            if body_line is None:
                continue
            body_end_line = lint_wrapped_line_to_body_line(end_line)
            if body_end_line is None:
                body_end_line = body_line

            diagnostics.append(
                {
                    "severity": sev,
                    "message": msg,
                    "startLineNumber": body_line,
                    "startColumn": col,
                    "endLineNumber": body_end_line,
                    "endColumn": end_col,
                }
            )

    return diagnostics
