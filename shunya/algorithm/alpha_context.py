from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp

from . import cross_section, time_series


def _as_2d_float32(x: Any, *, name: str) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array shaped (time, n_tickers), got {arr.shape}")
    return arr


def _coerce_series_data(x: Any) -> jnp.ndarray:
    if isinstance(x, AlphaSeries):
        return x.data
    return jnp.asarray(x, dtype=jnp.float32)


@dataclass(frozen=True)
class AlphaSeries:
    """
    Lightweight wrapper for per-ticker history tensors.

    Internal shape convention is ``(time, n_tickers)``. Binary operators preserve
    full history, while ``latest`` exposes the cross-section at execution time.
    """

    data: jnp.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _as_2d_float32(self.data, name="AlphaSeries.data"))

    @property
    def latest(self) -> jnp.ndarray:
        return self.data[-1, :]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def _binary(self, other: Any, op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> AlphaSeries:
        return AlphaSeries(op(self.data, _coerce_series_data(other)))

    def _rbinary(self, other: Any, op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]) -> AlphaSeries:
        return AlphaSeries(op(_coerce_series_data(other), self.data))

    def __add__(self, other: Any) -> AlphaSeries:
        return self._binary(other, jnp.add)

    def __radd__(self, other: Any) -> AlphaSeries:
        return self._rbinary(other, jnp.add)

    def __sub__(self, other: Any) -> AlphaSeries:
        return self._binary(other, jnp.subtract)

    def __rsub__(self, other: Any) -> AlphaSeries:
        return self._rbinary(other, jnp.subtract)

    def __mul__(self, other: Any) -> AlphaSeries:
        return self._binary(other, jnp.multiply)

    def __rmul__(self, other: Any) -> AlphaSeries:
        return self._rbinary(other, jnp.multiply)

    def __truediv__(self, other: Any) -> AlphaSeries:
        return self._binary(other, jnp.divide)

    def __rtruediv__(self, other: Any) -> AlphaSeries:
        return self._rbinary(other, jnp.divide)

    def __pow__(self, other: Any) -> AlphaSeries:
        return self._binary(other, jnp.power)

    def __rpow__(self, other: Any) -> AlphaSeries:
        return self._rbinary(other, jnp.power)

    def __neg__(self) -> AlphaSeries:
        return AlphaSeries(-self.data)

    def __abs__(self) -> AlphaSeries:
        return AlphaSeries(jnp.abs(self.data))

    def __getitem__(self, item: Any) -> jnp.ndarray:
        return self.data[item]


class TimeSeriesOps:
    """Time-series operators over ``(time, n_tickers)`` history tensors."""

    @staticmethod
    def delay(x: AlphaSeries | jnp.ndarray, lag: int) -> AlphaSeries:
        return AlphaSeries(time_series.tsdelay(_coerce_series_data(x), lag))

    @staticmethod
    def delta(x: AlphaSeries | jnp.ndarray, lag: int) -> AlphaSeries:
        return AlphaSeries(time_series.tsdelta(_coerce_series_data(x), lag))

    @staticmethod
    def sum(x: AlphaSeries | jnp.ndarray, window: int) -> AlphaSeries:
        return AlphaSeries(time_series.tssum(_coerce_series_data(x), window))

    @staticmethod
    def mean(x: AlphaSeries | jnp.ndarray, window: int) -> AlphaSeries:
        return AlphaSeries(time_series.tsmean(_coerce_series_data(x), window))

    @staticmethod
    def std(x: AlphaSeries | jnp.ndarray, window: int) -> AlphaSeries:
        return AlphaSeries(time_series.tsstddev(_coerce_series_data(x), window))

    @staticmethod
    def zscore(x: AlphaSeries | jnp.ndarray, window: int) -> AlphaSeries:
        return AlphaSeries(time_series.tszscore(_coerce_series_data(x), window))

    @staticmethod
    def rank(x: AlphaSeries | jnp.ndarray, window: int) -> AlphaSeries:
        return AlphaSeries(time_series.tsrank(_coerce_series_data(x), window))

    @staticmethod
    def regression(
        y: AlphaSeries | jnp.ndarray,
        x: AlphaSeries | jnp.ndarray,
        window: int,
        lag: int = 0,
        retval: str = "b",
    ) -> AlphaSeries:
        return AlphaSeries(
            time_series.tsregression(
                _coerce_series_data(y),
                _coerce_series_data(x),
                window,
                lag=lag,
                retval=retval,
            )
        )

    @staticmethod
    def humpdecay(x: AlphaSeries | jnp.ndarray, hump: float) -> AlphaSeries:
        return AlphaSeries(time_series.humpdecay(_coerce_series_data(x), hump))


class CrossSectionOps:
    """Cross-sectional operators over the latest per-ticker snapshot."""

    @staticmethod
    def _latest(x: AlphaSeries | jnp.ndarray) -> jnp.ndarray:
        if isinstance(x, AlphaSeries):
            return x.latest
        arr = jnp.asarray(x, dtype=jnp.float32)
        if arr.ndim == 2:
            return arr[-1, :]
        if arr.ndim == 1:
            return arr
        raise ValueError(f"Cross-sectional input must be 1D or 2D, got shape {arr.shape}")

    def rank(self, x: AlphaSeries | jnp.ndarray) -> jnp.ndarray:
        return cross_section.rank(self._latest(x))

    def zscore(self, x: AlphaSeries | jnp.ndarray) -> jnp.ndarray:
        return cross_section.zscore(self._latest(x))

    def scale(self, x: AlphaSeries | jnp.ndarray, *, target: float = 1.0) -> jnp.ndarray:
        return cross_section.scale(self._latest(x), target=target)

    def sign(self, x: AlphaSeries | jnp.ndarray) -> jnp.ndarray:
        return cross_section.sign(self._latest(x))

    def winsorize(self, x: AlphaSeries | jnp.ndarray, tail: float) -> jnp.ndarray:
        return cross_section.winsorize(self._latest(x), tail)

    def neutralize_market(self, x: AlphaSeries | jnp.ndarray) -> jnp.ndarray:
        return cross_section.neutralize_market(self._latest(x))

    def neutralize_groups(
        self, x: AlphaSeries | jnp.ndarray, group_ids: jnp.ndarray
    ) -> jnp.ndarray:
        return cross_section.neutralize_groups(self._latest(x), group_ids)


class AlphaContext:
    """
    User-facing alpha context.

    Exposes base fields as ``AlphaSeries`` and operator namespaces as ``ctx.ts`` and
    ``ctx.cs`` so alphas can stay declarative and JAX internals remain hidden.
    """

    def __init__(
        self,
        *,
        open: jnp.ndarray,
        high: jnp.ndarray,
        low: jnp.ndarray,
        close: jnp.ndarray,
        adj_volume: jnp.ndarray,
        features: Mapping[str, Any] | None = None,
    ) -> None:
        self.open = AlphaSeries(open)
        self.high = AlphaSeries(high)
        self.low = AlphaSeries(low)
        self.close = AlphaSeries(close)
        self.adj_volume = AlphaSeries(adj_volume)
        raw_features = dict(features or {})
        overlap = sorted(set(raw_features).intersection({"open", "high", "low", "close", "adj_volume"}))
        if overlap:
            raise ValueError(f"feature names collide with built-in AlphaContext fields: {overlap}")
        self.features = {str(name): AlphaSeries(value) for name, value in raw_features.items()}
        self.ts = TimeSeriesOps()
        self.cs = CrossSectionOps()

    @property
    def n_tickers(self) -> int:
        return int(self.close.shape[1])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features.keys())

    def feature(self, name: str) -> AlphaSeries:
        if name not in self.features:
            raise KeyError(f"AlphaContext feature {name!r} is not available")
        return self.features[name]

