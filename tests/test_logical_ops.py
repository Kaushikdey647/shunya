from __future__ import annotations

import numpy as np

from src.algorithm import logical


def test_if_else_and_logical_combinators():
    a = np.array([True, False, True])
    b = np.array([True, True, False])
    out = np.asarray(logical.if_else(a, np.array([1, 2, 3]), 0), dtype=float)
    assert np.allclose(out, np.array([1.0, 0.0, 3.0]))
    assert np.array_equal(np.asarray(logical.logical_and(a, b)), np.array([True, False, False]))
    assert np.array_equal(np.asarray(logical.logical_or(a, b)), np.array([True, True, True]))
    assert np.array_equal(np.asarray(logical.logical_not(a)), np.array([False, True, False]))


def test_trade_when_without_exit_condition():
    cond = np.array([True, False, True])
    alpha = np.array([1.0, 2.0, 3.0])
    out = np.asarray(logical.trade_when(cond, alpha, 0.0), dtype=float)
    assert np.allclose(out, np.array([1.0, 0.0, 3.0]))


def test_trade_when_entry_exit_state_machine():
    entry = np.array([False, True, False, False, False])
    exit_ = np.array([False, False, False, True, False])
    alpha = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = np.asarray(
        logical.trade_when(entry, alpha, -1.0, exit_condition=exit_),
        dtype=float,
    )
    assert np.allclose(out, np.array([-1.0, 2.0, 3.0, -1.0, -1.0]))
