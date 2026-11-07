"""Validate ``import_ref`` strings for packaged example alphas."""
from __future__ import annotations

import re

IMPORT_REF_PATTERN = re.compile(r"^examples\.alphas\.[a-z0-9_]+:alpha$")


def validate_import_ref(import_ref: str) -> None:
    """Raise ``ValueError`` when ``import_ref`` does not match ``examples.alphas.<module>:alpha``."""
    if not IMPORT_REF_PATTERN.match(import_ref.strip()):
        raise ValueError(
            "import_ref must match examples.alphas.<module_name>:alpha "
            "(lowercase module names and digits/underscores only)."
        )
