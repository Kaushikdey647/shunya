from __future__ import annotations

import numpy as np
import pytest

from src.algorithm import time_series as ts


def test_tsdelay_and_tsdelta():
    x = np.array([1.0, 2.0, 4.0, 7.0], dtype=float)
    d1 = np.asarray(ts.tsdelay(x, 1), dtype=float)
    assert np.isnan(d1[0])
    assert np.allclose(d1[1:], np.array([1.0, 2.0, 4.0]))

    delta = np.asarray(ts.tsdelta(x, 1), dtype=float)
    assert np.isnan(delta[0])
    assert np.allclose(delta[1:], np.array([1.0, 2.0, 3.0]))


def test_tssum_tsmean_tsstddev():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    s = np.asarray(ts.tssum(x, 2), dtype=float)
    m = np.asarray(ts.tsmean(x, 2), dtype=float)
    sd = np.asarray(ts.tsstddev(x, 2), dtype=float)
    assert np.isnan(s[0]) and np.isnan(m[0]) and np.isnan(sd[0])
    assert np.allclose(s[1:], np.array([3.0, 5.0, 7.0]))
    assert np.allclose(m[1:], np.array([1.5, 2.5, 3.5]))
    assert np.allclose(sd[1:], np.array([0.5, 0.5, 0.5]))


def test_tsrank_and_tszscore():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    r = np.asarray(ts.tsrank(x, 3), dtype=float)
    z = np.asarray(ts.tszscore(x, 3), dtype=float)
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert np.allclose(r[2:], np.array([1.0, 1.0]))
    assert np.isnan(z[0]) and np.isnan(z[1])
    assert z[2] == pytest.approx((3.0 - 2.0) / np.std([1.0, 2.0, 3.0]))


def test_tsregression_all_retvals():
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    b = np.asarray(ts.tsregression(y, x, window=3, retval="b"), dtype=float)
    a = np.asarray(ts.tsregression(y, x, window=3, retval="a"), dtype=float)
    est = np.asarray(ts.tsregression(y, x, window=3, retval="estimate"), dtype=float)
    err = np.asarray(ts.tsregression(y, x, window=3, retval="error"), dtype=float)
    assert np.isnan(b[0]) and np.isnan(b[1])
    assert b[2] == pytest.approx(1.0)
    assert a[2] == pytest.approx(1.0)
    assert est[3] == pytest.approx(4.0)
    assert err[3] == pytest.approx(0.0)


def test_tsregression_with_lag():
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    b = np.asarray(ts.tsregression(y, x, window=2, lag=1, retval="b"), dtype=float)
    assert np.isnan(b[0]) and np.isnan(b[1])
    assert b[2] == pytest.approx(2.0)


def test_humpdecay():
    x = np.array([100.0, 100.1, 101.0], dtype=float)
    out = np.asarray(ts.humpdecay(x, hump=0.5), dtype=float)
    assert out[0] == pytest.approx(100.0)
    assert out[1] == pytest.approx(100.0)
    assert out[2] == pytest.approx(101.0)


def test_invalid_window_lag_raise():
    with pytest.raises(ValueError):
        ts.tssum(np.array([1.0, 2.0]), 0)
    with pytest.raises(ValueError):
        ts.tsdelay(np.array([1.0, 2.0]), -1)
