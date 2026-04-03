"""Tests for decision-time helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.algorithm.decision import (
    DecisionContext,
    resolve_panel_timestamp,
    validate_panel_timestamp,
)


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


def test_validate_panel_timestamp_rejects_future_and_weekend():
    with pytest.raises(ValueError, match="future"):
        validate_panel_timestamp(
            resolved_as_of=pd.Timestamp("2025-01-10"),
            index_max_date=pd.Timestamp("2025-01-10"),
            now_ts=pd.Timestamp("2025-01-09"),
        )
    with pytest.raises(ValueError, match="weekend"):
        validate_panel_timestamp(
            resolved_as_of=pd.Timestamp("2025-01-11"),  # Saturday
            index_max_date=pd.Timestamp("2025-01-11"),
            now_ts=pd.Timestamp("2025-01-11"),
        )


def test_validate_panel_timestamp_warnings_for_staleness():
    dt, warns = validate_panel_timestamp(
        resolved_as_of=pd.Timestamp("2025-01-06"),
        index_max_date=pd.Timestamp("2025-01-10"),
        now_ts=pd.Timestamp("2025-01-10"),
        max_staleness_days=2,
    )
    assert dt == pd.Timestamp("2025-01-06")
    assert any("as_of_older_than_panel_max_by_days=4" in w for w in warns)
    assert any("as_of_staleness_exceeds_limit" in w for w in warns)


def test_validate_panel_timestamp_strict_same_session():
    with pytest.raises(ValueError, match="strict_same_session"):
        validate_panel_timestamp(
            resolved_as_of=pd.Timestamp("2025-01-08"),
            index_max_date=pd.Timestamp("2025-01-08"),
            now_ts=pd.Timestamp("2025-01-09"),
            strict_same_session=True,
        )
