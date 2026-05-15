"""Universe return panel: correlations, cross-sectional vol, PCA, cap-weight concentration (Timescale OHLCV)."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from api.repositories import universes as universes_repo
from api.schemas.models import (
    UniverseConcentrationOut,
    UniversePcaLoadingsScatterPoint,
    UniversePcaScorePoint,
    UniverseReturnAnalyticsOut,
    UniverseTickerLoading,
    UniverseTickerWeight,
    UniverseXsVolPoint,
)
from shunya.data.timescale.ohlcv_window import period_to_utc_bounds
from shunya.errors import ErrorCode, ShunyaError

_log = logging.getLogger(__name__)

_MIN_RETURNS = 10
_DEFAULT_PCA_COMPONENTS = 5
# Require this fraction of universe members to have a non-null close on a calendar day (not 100%).
_MIN_FRAC_CLOSES_PER_DAY = 0.55
# After pct_change, keep rows with at least this fraction of finite returns (cross-section / corr).
_MIN_FRAC_RETURNS_PER_DAY = 0.45


def _iso_date(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
    return t.normalize().date().isoformat()


def _corr_matrix_to_lists(corr: pd.DataFrame, tickers: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    mat = corr.reindex(index=tickers, columns=tickers)
    vals = mat.values.astype(float)
    np.fill_diagonal(vals, 1.0)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    for i, _ in enumerate(tickers):
        row = [float(x) for x in vals[i].tolist()]
        row[i] = 1.0
        out.append(row)
    return out


def _fetch_latest_market_caps(dsn: str, universe_id: str) -> tuple[dict[str, float], bool]:
    """Return ticker -> positive market_cap and whether any member had missing/zero cap."""
    import psycopg

    caps: dict[str, float] = {}
    partial = False
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker, d.market_cap
                FROM api_universe_members m
                JOIN symbols s ON s.id = m.symbol_id
                LEFT JOIN LATERAL (
                    SELECT fd.market_cap
                    FROM fundamentals_daily fd
                    WHERE fd.symbol_id = s.id
                    ORDER BY fd.as_of_ts DESC
                    LIMIT 1
                ) d ON true
                WHERE m.universe_id = %s
                ORDER BY s.ticker
                """,
                (universe_id,),
            )
            for tk, mc in cur.fetchall():
                tks = str(tk)
                if mc is None or (isinstance(mc, float) and (np.isnan(mc) or mc <= 0)):
                    partial = True
                    continue
                try:
                    v = float(mc)
                except (TypeError, ValueError):
                    partial = True
                    continue
                if v > 0 and not np.isnan(v):
                    caps[tks] = v
                else:
                    partial = True
    return caps, partial


def _concentration_from_caps(tickers: Sequence[str], caps: dict[str, float], partial: bool) -> UniverseConcentrationOut:
    weights: list[tuple[str, float]] = []
    total = 0.0
    for t in tickers:
        c = caps.get(t)
        if c is not None and c > 0:
            weights.append((t, c))
            total += c
    if total <= 0 or not weights:
        w = 1.0 / max(len(tickers), 1)
        norm = [(t, w) for t in tickers]
        total = sum(w for _, w in norm)
        norm = [(t, c / total) for t, c in norm]
        norm.sort(key=lambda x: x[1], reverse=True)
        ws = np.array([x[1] for x in norm], dtype=float)
        hhi = float(np.sum(ws**2)) if len(ws) else 0.0
        cr5 = float(np.sum(ws[:5])) if len(ws) else 0.0
        cr10 = float(np.sum(ws[:10])) if len(ws) else 0.0
        top = [UniverseTickerWeight(ticker=t, weight=float(w)) for t, w in norm[:10]]
        return UniverseConcentrationOut(
            hhi=hhi,
            cr5=cr5,
            cr10=cr10,
            weight_mode="equal",
            mcap_weights_partial=True,
            top_holdings=top,
        )
    norm = [(t, c / total) for t, c in weights]
    norm.sort(key=lambda x: x[1], reverse=True)
    ws = np.array([x[1] for x in norm], dtype=float)
    hhi = float(np.sum(ws**2)) if len(ws) else 0.0
    cr5 = float(np.sum(ws[:5])) if len(ws) else 0.0
    cr10 = float(np.sum(ws[:10])) if len(ws) else 0.0
    top = [UniverseTickerWeight(ticker=t, weight=float(w)) for t, w in norm[:10]]
    return UniverseConcentrationOut(
        hhi=hhi,
        cr5=cr5,
        cr10=cr10,
        weight_mode="mcap",
        mcap_weights_partial=partial,
        top_holdings=top,
    )


def compute_universe_return_analytics(
    universe_id: str,
    *,
    period: str,
    interval: str,
    source: str,
    max_members: int,
    n_pca_components: int = _DEFAULT_PCA_COMPONENTS,
) -> UniverseReturnAnalyticsOut:
    """
    Build aligned simple/log return correlations, cross-sectional vol, PCA (SVD), and cap concentration.

    Raises ``ShunyaError`` for validation, 404 universe, or insufficient overlapping OHLCV.
    """
    if universes_repo.get_universe(universe_id) is None:
        raise ShunyaError("Universe not found.", code=ErrorCode.UNIVERSE_NOT_FOUND, http_status=404)

    tickers = universes_repo.constituent_tickers(universe_id)
    if len(tickers) < 2:
        raise ShunyaError(
            "Universe needs at least two members for return analytics.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        )
    if len(tickers) > max_members:
        raise ShunyaError(
            f"Universe has {len(tickers)} members; max supported is {max_members}.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
            context={"member_count": len(tickers), "max_members": max_members},
        )
    if interval != "1d":
        raise ShunyaError(
            "Only interval=1d is supported for universe return analytics.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        )

    try:
        from shunya.data.timescale.dbutil import get_database_url

        dsn = get_database_url()
    except ValueError as exc:
        raise ShunyaError(
            "Timescale database URL is not configured.",
            code=ErrorCode.FIN_TS_TIMESCALE_DSN_REQUIRED,
            http_status=503,
        ) from exc

    start_inclusive, end_exclusive = period_to_utc_bounds(period)

    import psycopg

    rows: list[tuple[object, str, float]] = []
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT b.ts, s.ticker, b.close::double precision AS close
                    FROM ohlcv_bars b
                    JOIN symbols s ON s.id = b.symbol_id
                    WHERE b.interval = %s
                      AND b.source = %s
                      AND b.ts >= %s
                      AND b.ts < %s
                      AND s.ticker = ANY(%s)
                    ORDER BY b.ts ASC, s.ticker ASC
                    """,
                    (interval, source, start_inclusive, end_exclusive, list(tickers)),
                )
                rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        _log.warning("universe_return_analytics query failed: %s", exc)
        raise ShunyaError(
            "Failed to read OHLCV from Timescale.",
            code=ErrorCode.FIN_TS_TIMESCALE_UNAVAILABLE,
            http_status=503,
        ) from exc

    if not rows:
        raise ShunyaError(
            "No OHLCV bars found for universe members in the selected window.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        )

    df = pd.DataFrame(rows, columns=["ts", "ticker", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset=["ts", "ticker"], keep="last")
    wide = df.pivot(index="ts", columns="ticker", values="close")
    wide = wide.reindex(columns=list(tickers))
    n_t = len(tickers)
    thresh_prices = max(2, min(n_t, int(np.ceil(_MIN_FRAC_CLOSES_PER_DAY * n_t))))
    wide = wide.dropna(axis=0, thresh=thresh_prices)
    if wide.shape[0] < 3:
        raise ShunyaError(
            "Too few calendar days with sufficient member OHLCV; widen the period or refresh OHLCV.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        )

    simple_ret = wide.pct_change().iloc[1:]
    simple_ret = simple_ret.dropna(axis=0, how="all")
    thresh_rets = max(2, min(n_t, int(np.ceil(_MIN_FRAC_RETURNS_PER_DAY * n_t))))
    simple_ret = simple_ret.dropna(axis=0, thresh=thresh_rets)

    log_wide = np.log(wide.replace(0, np.nan))
    log_ret = log_wide.diff().iloc[1:]
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="all").dropna(axis=0, thresh=thresh_rets)

    if len(simple_ret) < _MIN_RETURNS:
        raise ShunyaError(
            f"Need at least {_MIN_RETURNS} return rows after panel alignment; got {len(simple_ret)}. "
            "Try a longer period or ensure OHLCV coverage for most members.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        )

    min_periods_corr = max(5, min(len(simple_ret), max(8, int(0.15 * len(simple_ret)))))
    c_simple = simple_ret.corr(min_periods=min_periods_corr)
    c_log = log_ret.corr(min_periods=min_periods_corr)
    corr_simple = _corr_matrix_to_lists(c_simple, list(tickers))
    corr_log = _corr_matrix_to_lists(c_log, list(tickers))

    xs = simple_ret.std(axis=1, ddof=1, skipna=True)
    cross_sectional_vol = [
        UniverseXsVolPoint(date=_iso_date(idx), xs_vol=float(v))
        for idx, v in xs.items()
        if pd.notna(v)
    ]

    X = simple_ret.to_numpy(dtype=float)
    col_mean = np.nanmean(X, axis=0)
    Xc = X - col_mean
    col_std = np.nanstd(Xc, axis=0, ddof=1)
    col_std = np.where(col_std > 1e-12, col_std, 1.0)
    Xs = Xc / col_std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    total_var = float(np.sum(s**2)) if len(s) else 0.0
    evr = [(float(si * si) / total_var) if total_var > 0 else 0.0 for si in s]
    k = min(n_pca_components, len(evr))
    evr_out = evr[:k]

    vt0 = Vt[0] if Vt.shape[0] >= 1 else np.zeros(len(tickers))
    pca_pc1_loadings = [
        UniverseTickerLoading(ticker=tickers[j], loading=float(vt0[j]))
        for j in range(len(tickers))
    ]
    pca_pc2_loadings: list[UniverseTickerLoading] = []
    vt1 = Vt[1] if Vt.shape[0] >= 2 else None
    if vt1 is not None:
        pca_pc2_loadings = [
            UniverseTickerLoading(ticker=tickers[j], loading=float(vt1[j])) for j in range(len(tickers))
        ]
    scores1 = U[:, 0] * s[0] if len(s) else np.zeros(Xs.shape[0])
    pca_pc1_scores = [
        UniversePcaScorePoint(date=_iso_date(simple_ret.index[i]), score=float(scores1[i]))
        for i in range(len(simple_ret.index))
    ]
    scatter: list[UniversePcaLoadingsScatterPoint] = []
    if vt1 is not None:
        for j, tk in enumerate(tickers):
            scatter.append(
                UniversePcaLoadingsScatterPoint(
                    ticker=tk,
                    pc1_loading=float(vt0[j]),
                    pc2_loading=float(vt1[j]),
                )
            )

    caps, cap_partial = _fetch_latest_market_caps(dsn, universe_id)
    concentration = _concentration_from_caps(tickers, caps, cap_partial)

    align_note = (
        f"closes_per_day>={thresh_prices}/{n_t}; "
        f"finite_returns_per_day>={thresh_rets}/{n_t}; "
        f"corr_min_periods={min_periods_corr}"
    )

    return UniverseReturnAnalyticsOut(
        universe_id=universe_id,
        period=period,
        interval=interval,
        source=source,
        start_date=_iso_date(start_inclusive),
        end_date_exclusive=pd.Timestamp(end_exclusive).isoformat(),
        tickers=list(tickers),
        n_observations=int(len(simple_ret)),
        alignment=align_note,
        correlation_simple=corr_simple,
        correlation_log=corr_log,
        cross_sectional_vol=cross_sectional_vol,
        pca_explained_variance_ratio=evr_out,
        pca_pc1_loadings=pca_pc1_loadings,
        pca_pc2_loadings=pca_pc2_loadings,
        pca_pc1_scores=pca_pc1_scores,
        pca_loadings_scatter=scatter,
        concentration=concentration,
    )
