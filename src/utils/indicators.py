"""
Column names for :class:`src.data.fints.finTs` and integer indices for JAX feature vectors.

- **Dataframe columns:** ``df[COL.SMA_50]`` or legacy module aliases ``df[SMA_50]``.
- **JAX rows:** ``x[IX.RSI_14]`` with ``x`` shaped ``(len(STRATEGY_FEATURES),)``.
  For live / no-lookahead inputs use ``STRATEGY_FEATURES_LIVE`` width and ``IX_LIVE``.

``STRATEGY_FEATURES`` order matches ``finTs._add_features``. ``Future_1d_Ret`` is
**lookahead** — use ``IX_LIVE`` and 16-wide vectors for real-time ``FinStrat.pass_``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

# (Python attribute name, exact pandas column string) — single source of truth.
_OHLCV_SPEC: tuple[tuple[str, str], ...] = (
    ("OPEN", "Open"),
    ("HIGH", "High"),
    ("LOW", "Low"),
    ("CLOSE", "Close"),
    ("ADJ_CLOSE", "Adj Close"),
    ("VOLUME", "Volume"),
)

_ENGINEERED_SPEC: tuple[tuple[str, str], ...] = (
    ("SMA_50", "SMA_50"),
    ("SMA_200", "SMA_200"),
    ("RSI_14", "RSI_14"),
    ("MACD", "MACD"),
    ("MACD_SIGNAL", "MACD_Signal"),
    ("BB_UPPER", "BB_Upper"),
    ("BB_LOWER", "BB_Lower"),
    ("ATR_14", "ATR_14"),
    ("FUTURE_1D_RET", "Future_1d_Ret"),
    ("LOG_RET", "Log_Ret"),
    ("DIST_SMA50", "Dist_SMA50"),
    ("DIST_SMA200", "Dist_SMA200"),
    ("BB_WIDTH", "BB_Width"),
    ("BB_POSITION", "BB_Position"),
    ("MACD_HIST", "MACD_Hist"),
    ("ATR_NORM", "ATR_Norm"),
    ("VOL_CHANGE", "Vol_Change"),
)

# Full OHLCV + engineered column namespace for ``df[COL.OPEN]`` etc.
COL = SimpleNamespace(
    **{attr: col for attr, col in _OHLCV_SPEC + _ENGINEERED_SPEC},
)

# Integer positions into the 17-wide strategy vector (same names as COL for engineered).
IX = SimpleNamespace(
    **{attr: i for i, (attr, _) in enumerate(_ENGINEERED_SPEC)},
)

_LIVE_ENGINEERED_SPEC: tuple[tuple[str, str], ...] = tuple(
    (a, c) for a, c in _ENGINEERED_SPEC if c != "Future_1d_Ret"
)

IX_LIVE = SimpleNamespace(
    **{attr: i for i, (attr, _) in enumerate(_LIVE_ENGINEERED_SPEC)},
)

STRATEGY_FEATURES: tuple[str, ...] = tuple(c for _, c in _ENGINEERED_SPEC)
STRATEGY_FEATURES_LIVE: tuple[str, ...] = tuple(c for _, c in _LIVE_ENGINEERED_SPEC)

FEATURE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(STRATEGY_FEATURES)}
FEATURE_TO_IDX_LIVE: dict[str, int] = {
    name: i for i, name in enumerate(STRATEGY_FEATURES_LIVE)
}

# Module-level string aliases (backward compatible with ``df[SMA_50]``).
for _attr, _col in _OHLCV_SPEC + _ENGINEERED_SPEC:
    globals()[_attr] = _col

# Drift check: engineered columns added in finTs._add_features (order-independent).
_ENGINEERED_COLUMNS_FINTS: frozenset[str] = frozenset(STRATEGY_FEATURES)


def feature_index(name: str, *, live: bool = False) -> int:
    """
    Return index of ``name`` in ``STRATEGY_FEATURES`` (or ``STRATEGY_FEATURES_LIVE``).
    """
    m = FEATURE_TO_IDX_LIVE if live else FEATURE_TO_IDX
    if name not in m:
        subset = "STRATEGY_FEATURES_LIVE" if live else "STRATEGY_FEATURES"
        raise KeyError(f"{name!r} is not in {subset}")
    return m[name]


def strategy_feature_indices(columns: pd.Index, *, live: bool = False) -> list[int]:
    """
    Positions of strategy feature columns within ``columns``, in strategy order.

    Raises if any required feature name is missing from ``columns``.
    """
    names = STRATEGY_FEATURES_LIVE if live else STRATEGY_FEATURES
    col_list = list(columns)
    idx_map = {c: i for i, c in enumerate(col_list)}
    out: list[int] = []
    for n in names:
        if n not in idx_map:
            raise KeyError(f"Column {n!r} not in dataframe columns")
        out.append(idx_map[n])
    return out


def assert_engineered_columns_match_fints() -> None:
    """Assert engineered constant set matches ``finTs._add_features`` output columns."""
    expected = frozenset(
        [
            "SMA_50",
            "SMA_200",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Lower",
            "ATR_14",
            "Future_1d_Ret",
            "Log_Ret",
            "Dist_SMA50",
            "Dist_SMA200",
            "BB_Width",
            "BB_Position",
            "MACD_Hist",
            "ATR_Norm",
            "Vol_Change",
        ]
    )
    if _ENGINEERED_COLUMNS_FINTS != expected:
        raise AssertionError(
            f"indicators engineered set drift: {(_ENGINEERED_COLUMNS_FINTS ^ expected)!r}"
        )
