"""Docstring coverage for the public library/API and critical internals."""

from pathlib import Path

from scripts.docstring_coverage import in_scope, missing_in_source, scan_tree

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_symbol_without_docstring_is_reported() -> None:
    source = "def public():\n    return 1\n"
    assert missing_in_source(source, "shunya/example.py") == ["shunya/example.py:1 public"]


def test_documented_public_symbol_is_accepted() -> None:
    source = 'def public():\n    """Return one."""\n    return 1\n'
    assert missing_in_source(source, "shunya/example.py") == []


def test_private_symbol_is_ignored_outside_critical_modules() -> None:
    source = "def _helper():\n    return 1\n"
    assert not in_scope("shunya/algorithm/example.py", "_helper")
    assert missing_in_source(source, "shunya/algorithm/example.py") == []


def test_critical_private_symbol_and_init_are_required() -> None:
    source = "class Engine:\n    def __init__(self):\n        self._x = 1\n    def _run(self):\n        return self._x\n"
    missing = missing_in_source(source, "api/runner.py")
    assert missing == [
        "api/runner.py:1 Engine",
        "api/runner.py:2 __init__",
        "api/runner.py:4 _run",
    ]


def test_other_dunders_stay_out_of_scope() -> None:
    assert not in_scope("api/runner.py", "__repr__")


def test_repository_docstring_coverage() -> None:
    missing = scan_tree(REPO_ROOT)
    assert missing == [], "\n".join(missing)
