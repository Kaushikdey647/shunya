"""Report public and architecture-critical symbols that lack docstrings."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOTS = ("shunya", "api")

CRITICAL_FILES = frozenset(
    {
        "api/runner.py",
        "api/worker.py",
        "api/worker_job.py",
        "api/fin_ts_factory.py",
        "api/resolver.py",
        "api/serializer.py",
        "api/universe_resolve.py",
        "api/backtest_resolve.py",
        "api/repositories/alphas.py",
        "api/repositories/universes.py",
        "api/repositories/backtests.py",
        "api/services/oms_event_hub.py",
        "api/services/oms_orders.py",
        "api/services/oms_runtime.py",
    }
)

CRITICAL_PREFIXES = (
    "shunya/adapters/",
    "shunya/ports/",
    "shunya/oms/",
    "shunya/ems/",
    "shunya/data/market_data/",
)


def is_critical(rel: str) -> bool:
    """Return whether private symbols in ``rel`` are part of the docs gate."""
    rel = rel.replace("\\", "/")
    if rel in CRITICAL_FILES:
        return True
    return any(rel.startswith(prefix) for prefix in CRITICAL_PREFIXES)


def in_scope(rel: str, name: str) -> bool:
    """Return whether ``name`` in ``rel`` must have a docstring."""
    if name.startswith("__") and name.endswith("__"):
        return is_critical(rel) and name == "__init__"
    if name.startswith("_"):
        return is_critical(rel)
    return True


def missing_in_source(source: str, rel: str) -> list[str]:
    """Return ``path:line name`` entries for undocumented in-scope symbols."""
    tree = ast.parse(source)
    nodes: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(node)
        if isinstance(node, ast.ClassDef):
            nodes.extend(
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    missing: list[str] = []
    for node in nodes:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not in_scope(rel, node.name):
            continue
        doc = ast.get_docstring(node)
        if doc and doc.strip():
            continue
        missing.append(f"{rel}:{node.lineno} {node.name}")
    return missing


def scan_tree(root: Path) -> list[str]:
    """Scan ``shunya`` and ``api`` under ``root`` for missing docstrings."""
    missing: list[str] = []
    for package in ROOTS:
        base = root / package
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            missing.extend(missing_in_source(path.read_text(encoding="utf-8"), rel))
    return missing


def main() -> int:
    """Print missing symbols and return a non-zero status when any remain."""
    root = Path(__file__).resolve().parents[1]
    missing = scan_tree(root)
    if missing:
        print(f"{len(missing)} undocumented symbols:")
        print("\n".join(missing))
        return 1
    print("docstring coverage ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
