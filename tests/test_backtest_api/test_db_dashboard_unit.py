from __future__ import annotations

import pandas as pd

from backtest_api.db_dashboard import (
    completeness_histogram,
    longest_contiguous_run,
    resolve_bucket_granularity,
    _bucket_meta_for_step,
    _compress_bits_row,
    _compression_step,
)


def test_longest_contiguous_run() -> None:
    assert longest_contiguous_run([1, 1, 0, 1, 1, 1]) == 3
    assert longest_contiguous_run([]) == 0


def test_completeness_histogram() -> None:
    h = completeness_histogram([0, 15, 50, 99, 100], bins=10)
    assert sum(h) == 5
    assert h[0] >= 1
    assert h[-1] >= 1


def test_resolve_bucket_granularity_auto() -> None:
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    t1 = pd.Timestamp("2020-01-10", tz="UTC")
    assert resolve_bucket_granularity(t0, t1, "auto", max_buckets=200) == "day"


def test_compression_step_and_bits() -> None:
    bits = [1, 0, 1, 1, 0, 0, 1, 1]
    step = _compression_step(len(bits), max_buckets=4)
    assert step == 2
    out = _compress_bits_row(bits, step=step)
    assert out == [1, 1, 0, 1]


def test_bucket_meta_for_step_month() -> None:
    starts = [
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-02-01", tz="UTC"),
        pd.Timestamp("2020-03-01", tz="UTC"),
    ]
    meta, sub = _bucket_meta_for_step(starts, "month", step=2)
    assert sub is True
    assert len(meta) == 2
