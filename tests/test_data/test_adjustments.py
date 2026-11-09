"""Corporate-action adjustment factors and read-time adjustment of raw bars."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from shunya.data.adjustments import AdjustmentEvent, adjust_bars, cumulative_factors, event_factors


def _bars(dates: list[str], closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": volumes or [1_000.0] * n,
            "VWAP": closes,
            "TradeCount": [10] * n,
        },
        index=pd.DatetimeIndex(dates),
    )


def _split(ex: date, old: float, new: float) -> AdjustmentEvent:
    pf, vf = event_factors("forward_split" if new > old else "reverse_split", old_rate=old, new_rate=new)
    return AdjustmentEvent(ex, "forward_split" if new > old else "reverse_split", pf, vf)


def test_aapl_4_for_1_split_2020() -> None:
    raw = _bars(["2020-08-27", "2020-08-28", "2020-08-31"], [500.04, 499.23, 129.04], [38_888_096, 46_907_479, 225_702_700])
    adj = adjust_bars(raw, [_split(date(2020, 8, 31), 1, 4)])
    assert adj["Close"].tolist() == pytest.approx([125.01, 124.8075, 129.04])
    assert adj["Volume"].tolist() == pytest.approx([155_552_384, 187_629_916, 225_702_700])
    assert adj["TradeCount"].tolist() == [10, 10, 10]


def test_nvda_10_for_1_split_2024() -> None:
    raw = _bars(["2024-06-07", "2024-06-10"], [1208.88, 121.79])
    adj = adjust_bars(raw, [_split(date(2024, 6, 10), 1, 10)])
    assert adj["Close"].tolist() == pytest.approx([120.888, 121.79])
    assert adj["VWAP"].tolist() == pytest.approx([120.888, 121.79])


def test_reverse_split_factors() -> None:
    assert event_factors("reverse_split", old_rate=10, new_rate=1) == pytest.approx((10.0, 0.1))


def test_cash_dividend_uses_prior_raw_close_and_spares_ex_date() -> None:
    pf, vf = event_factors("cash_dividend", rate=1.0, prev_close=100.0)
    assert (pf, vf) == pytest.approx((0.99, 1.0))
    raw = _bars(["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 99.0])
    ev = AdjustmentEvent(date(2024, 1, 4), "cash_dividend", pf, vf)
    assert adjust_bars(raw, [ev], "all")["Close"].tolist() == pytest.approx([99.0, 99.0, 99.0])
    assert adjust_bars(raw, [ev], "split")["Close"].tolist() == [100.0, 100.0, 99.0]
    assert adjust_bars(raw, [ev], "raw")["Close"].tolist() == [100.0, 100.0, 99.0]


def test_events_compound_and_anchor_at_latest_action() -> None:
    dates = pd.DatetimeIndex(["2024-01-02", "2024-02-01", "2024-03-01"])
    evs = [
        _split(date(2024, 2, 1), 1, 2),
        AdjustmentEvent(date(2024, 3, 1), "cash_dividend", 0.98, 1.0),
        _split(date(2025, 1, 2), 1, 3),  # after the window: still scales it (anchored at today)
        _split(date(2023, 1, 3), 1, 5),  # before the window: no bar precedes it
    ]
    pf, vf = cumulative_factors(dates, evs, "all")
    assert pf == pytest.approx([0.5 * 0.98 / 3, 0.98 / 3, 1 / 3])
    assert vf == pytest.approx([6.0, 3.0, 3.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_type": "forward_split", "old_rate": 0, "new_rate": 4},
        {"action_type": "forward_split", "old_rate": 1, "new_rate": None},
        {"action_type": "cash_dividend", "rate": 1.0, "prev_close": None},
        {"action_type": "cash_dividend", "rate": 5.0, "prev_close": 5.0},
        {"action_type": "cash_dividend", "rate": 1.0, "prev_close": float("nan")},
        {"action_type": "stock_dividend", "rate": 0},
        {"action_type": "spin_off"},
        {"action_type": "name_change"},
    ],
)
def test_event_factors_unknown_or_no_effect(kwargs: dict) -> None:
    assert event_factors(**kwargs) is None


def test_stock_dividend() -> None:
    assert event_factors("stock_dividend", rate=0.05) == pytest.approx((1 / 1.05, 1.05))


def test_no_events_is_identity() -> None:
    raw = _bars(["2024-01-02"], [10.0])
    pd.testing.assert_frame_equal(adjust_bars(raw, []), raw.astype({c: float for c in ("Open", "High", "Low", "Close", "Volume", "VWAP")}))
    pf, vf = cumulative_factors(pd.DatetimeIndex([]), [], "all")
    assert pf.size == vf.size == 0 and np.all(pf == 1)
