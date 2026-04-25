from __future__ import annotations

import pytest

from backtest_api.alpha_validation import validate_import_ref


def test_import_ref_ok() -> None:
    validate_import_ref("examples.alphas.sma_ratio_20:alpha")


@pytest.mark.parametrize(
    "ref",
    [
        "examples.alphas.SMA_bad:alpha",
        "other.alphas.foo:alpha",
        "examples.alphas.foo:beta",
        "",
    ],
)
def test_import_ref_rejected(ref: str) -> None:
    with pytest.raises(ValueError):
        validate_import_ref(ref)
