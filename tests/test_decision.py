"""Tests for decision-time helpers."""

from __future__ import annotations

import pandas as pd

from src.algorithm.decision import DecisionContext, resolve_panel_timestamp


def test_resolve_precedence_decision_over_index_max():
    decision = DecisionContext(as_of=pd.Timestamp("2024-06-01"))
    idx_max = pd.Timestamp("2024-12-31")
    out = resolve_panel_timestamp(
        decision=decision,
        explicit_as_of=pd.Timestamp("2024-01-01"),
        index_max_date=idx_max,
    )
    assert out == pd.Timestamp("2024-06-01").normalize()


def test_resolve_explicit_as_of_when_no_decision():
    idx_max = pd.Timestamp("2024-12-31")
    out = resolve_panel_timestamp(
        decision=None,
        explicit_as_of="2024-03-15",
        index_max_date=idx_max,
    )
    assert out == pd.Timestamp("2024-03-15").normalize()


def test_resolve_fallback_index_max():
    idx_max = pd.Timestamp("2024-07-04")
    out = resolve_panel_timestamp(
        decision=None,
        explicit_as_of=None,
        index_max_date=idx_max,
    )
    assert out == idx_max.normalize()
