from __future__ import annotations

import numpy as np
import pytest

from src.algorithm import group_ops


def test_group_mean():
    x = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["a", "a", "b", "b"])
    out = np.asarray(group_ops.group_mean(x, g), dtype=float)
    assert np.allclose(out, np.array([2.0, 2.0, 15.0, 15.0]))


def test_group_neutralize_sums_to_zero_within_group():
    x = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["a", "a", "b", "b"])
    out = np.asarray(group_ops.group_neutralize(x, g), dtype=float)
    assert (out[0] + out[1]) == pytest.approx(0.0, abs=1e-6)
    assert (out[2] + out[3]) == pytest.approx(0.0, abs=1e-6)


def test_group_zscore():
    x = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["a", "a", "b", "b"])
    out = np.asarray(group_ops.group_zscore(x, g), dtype=float)
    assert out[0] == pytest.approx(-1.0)
    assert out[1] == pytest.approx(1.0)


def test_group_rank():
    x = np.array([1.0, 3.0, 10.0, 20.0])
    g = np.array(["a", "a", "b", "b"])
    out = np.asarray(group_ops.group_rank(x, g), dtype=float)
    assert np.allclose(out, np.array([0.0, 1.0, 0.0, 1.0]))
