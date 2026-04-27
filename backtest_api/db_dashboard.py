"""Aggregate OHLCV coverage and risk metrics from Timescale for GET /data/dashboard."""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd

from backtest_api.risk_metrics import periods_per_year_from_bar_spec, per_bar_return_stats_with_ppy
from backtest_api.schemas.models import (
    ClassificationLabelCount,
    DashboardBucketMeta,
    DataDashboardResponse,
    TickerDashboardRow,
    TickerRiskRow,
)
from shunya.data.timescale.dbutil import get_database_url
from shunya.data.timescale.ohlcv_window import yfinance_interval_to_bar_spec

BucketLiteral = Literal["auto", "day", "week", "month"]

_MAX_BUCKETS_DEFAULT = 200


def _env_max_tickers() -> int | None:
    raw = os.environ.get("SHUNYA_DASHBOARD_MAX_TICKERS", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


def longest_contiguous_run(bits: list[int]) -> int:
    best = cur = 0
    for b in bits:
        if b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def completeness_histogram(completeness_values: list[float], bins: int = 10) -> list[int]:
    """Count tickers per completeness decile [0,10), …, [90,100]."""
    hist = [0] * bins
    for v in completeness_values:
        if not math.isfinite(v):
            continue
        x = max(0.0, min(100.0, float(v)))
        i = int(x // (100.0 / bins))
        if i >= bins:
            i = bins - 1
        hist[i] += 1
    return hist


def _trunc_unit_sql(gran: Literal["day", "week", "month"]) -> str:
    return {"day": "day", "week": "week", "month": "month"}[gran]


def _norm_utc(ts: pd.Timestamp | object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _bucket_starts(t0: pd.Timestamp, t1: pd.Timestamp, gran: Literal["day", "week", "month"]) -> list[pd.Timestamp]:
    """Aligned bucket starts in UTC for heatmap columns."""
    t0 = _norm_utc(t0)
    t1 = _norm_utc(t1)

    if gran == "day":
        return list(pd.date_range(t0.normalize(), t1.normalize(), freq="D", tz="UTC"))
    if gran == "week":
        dr = pd.date_range(t0, t1, freq="W-MON", tz="UTC")
        return list(dr) if dr.size else [t0]
    dr = pd.date_range(t0, t1, freq="MS", tz="UTC")
    return list(dr) if dr.size else [t0]


def _bucket_count_estimate(t0: pd.Timestamp, t1: pd.Timestamp, gran: Literal["day", "week", "month"]) -> int:
    return len(_bucket_starts(t0, t1, gran))


def resolve_bucket_granularity(
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    explicit: BucketLiteral,
    *,
    max_buckets: int = _MAX_BUCKETS_DEFAULT,
) -> Literal["day", "week", "month"]:
    if explicit in ("day", "week", "month"):
        return explicit
    for cand in ("day", "week", "month"):
        if _bucket_count_estimate(t0, t1, cand) <= max_buckets:
            return cand
    return "month"


def _bucket_key(ts: pd.Timestamp | object, gran: Literal["day", "week", "month"]) -> pd.Timestamp:
    ts = _norm_utc(ts)
    if gran == "day":
        return ts.normalize()
    if gran == "week":
        d = ts.normalize()
        return d - pd.Timedelta(days=int(d.weekday()))
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tzinfo=ts.tz)


def _compression_step(n_full: int, max_buckets: int) -> int:
    if n_full <= max_buckets:
        return 1
    return int(math.ceil(n_full / max_buckets))


def _compress_bits_row(bits: list[int], *, step: int) -> list[int]:
    if step <= 1:
        return list(bits)
    out: list[int] = []
    for j in range(0, len(bits), step):
        chunk = bits[j : j + step]
        out.append(1 if any(chunk) else 0)
    return out


def _bucket_meta_for_step(
    bucket_starts: list[pd.Timestamp],
    gran: Literal["day", "week", "month"],
    *,
    step: int,
) -> tuple[list[DashboardBucketMeta], bool]:
    """Build column metadata; ``subsampled`` True when step > 1."""
    if step <= 1:
        return _bucket_meta_from_starts(bucket_starts, gran), False

    meta: list[DashboardBucketMeta] = []
    for j in range(0, len(bucket_starts), step):
        bs_ts = _norm_utc(bucket_starts[j])
        end_j = min(j + step, len(bucket_starts)) - 1
        end_ts = _norm_utc(bucket_starts[end_j])
        next_end = _bucket_end_exclusive(end_ts, gran)
        meta.append(
            DashboardBucketMeta(
                index=len(meta),
                start=bs_ts.isoformat(),
                end=next_end.isoformat(),
            )
        )
    return meta, True


def _bucket_end_exclusive(bs_ts: pd.Timestamp, gran: Literal["day", "week", "month"]) -> pd.Timestamp:
    bs_ts = _norm_utc(bs_ts)
    if gran == "day":
        return bs_ts + pd.Timedelta(days=1)
    if gran == "week":
        return bs_ts + pd.Timedelta(weeks=1)
    return bs_ts + pd.DateOffset(months=1)


def _bucket_meta_from_starts(bucket_starts: list[pd.Timestamp], gran: Literal["day", "week", "month"]) -> list[DashboardBucketMeta]:
    meta: list[DashboardBucketMeta] = []
    for i, bs in enumerate(bucket_starts):
        bs_ts = _norm_utc(bs)
        meta.append(
            DashboardBucketMeta(
                index=i,
                start=bs_ts.isoformat(),
                end=_bucket_end_exclusive(bs_ts, gran).isoformat(),
            )
        )
    return meta


def compute_data_dashboard(
    *,
    interval: str,
    source: str,
    bucket: BucketLiteral = "auto",
    max_buckets: int = _MAX_BUCKETS_DEFAULT,
) -> DataDashboardResponse:
    """Load coverage matrix and risk metrics from ``ohlcv_bars`` for all symbols."""
    try:
        dsn = get_database_url()
    except ValueError as exc:
        raise RuntimeError("database not configured") from exc

    try:
        import psycopg
    except ModuleNotFoundError as exc:
        raise RuntimeError("timescale extra required for dashboard") from exc

    bar_spec = yfinance_interval_to_bar_spec(interval)
    ppy = periods_per_year_from_bar_spec(bar_spec)

    max_tickers = _env_max_tickers()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MIN(b.ts), MAX(b.ts), COUNT(*)::bigint
                FROM ohlcv_bars b
                WHERE b.interval = %s AND b.source = %s
                """,
                (interval, source),
            )
            row_bounds = cur.fetchone()
            if row_bounds is None or row_bounds[0] is None:
                raise ValueError("no OHLCV rows for this interval and source")

            ref_start, ref_end, _total_rows = row_bounds[0], row_bounds[1], row_bounds[2]
            t0 = pd.Timestamp(ref_start)
            t1 = pd.Timestamp(ref_end)

            gran = resolve_bucket_granularity(t0, t1, bucket, max_buckets=max_buckets)
            trunc = _trunc_unit_sql(gran)

            cur.execute(
                """
                SELECT s.ticker, MIN(b.ts), MAX(b.ts), COUNT(*)::bigint
                FROM ohlcv_bars b
                JOIN symbols s ON s.id = b.symbol_id
                WHERE b.interval = %s AND b.source = %s
                GROUP BY s.id, s.ticker
                ORDER BY s.ticker ASC
                """,
                (interval, source),
            )
            sym_rows = cur.fetchall()

    truncated = False
    if max_tickers is not None and len(sym_rows) > max_tickers:
        sym_rows = sym_rows[:max_tickers]
        truncated = True

    tickers_ordered = [str(r[0]) for r in sym_rows]
    sym_stats = {str(r[0]): (r[1], r[2], int(r[3])) for r in sym_rows}

    if not tickers_ordered:
        raise ValueError("no symbols with OHLCV for this interval and source")

    bucket_starts_full = _bucket_starts(t0, t1, gran)
    n_full = len(bucket_starts_full)

    col_by_key = {_norm_utc(bs): i for i, bs in enumerate(bucket_starts_full)}

    def _column_for_bucket(nk: pd.Timestamp) -> int | None:
        nk_u = _norm_utc(nk)
        if nk_u in col_by_key:
            return col_by_key[nk_u]
        for i, bs in enumerate(bucket_starts_full):
            bs_u = _norm_utc(bs)
            if gran == "day":
                if nk_u.normalize() == bs_u.normalize():
                    return i
            elif gran == "week":
                if bs_u <= nk_u < bs_u + pd.Timedelta(days=7):
                    return i
            elif nk_u.year == bs_u.year and nk_u.month == bs_u.month:
                return i
        return None

    step = _compression_step(n_full, max_buckets)
    bucket_meta_final, subsampled = _bucket_meta_for_step(bucket_starts_full, gran, step=step)

    presence_bits: dict[str, list[int]] = {t: [0] * n_full for t in tickers_ordered}

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT s.ticker, date_trunc(%s, b.ts) AS bk
                FROM ohlcv_bars b
                JOIN symbols s ON s.id = b.symbol_id
                WHERE b.interval = %s AND b.source = %s
                  AND b.ts >= %s AND b.ts <= %s
                  AND s.ticker = ANY(%s)
                """,
                (trunc, interval, source, t0, t1, tickers_ordered),
            )
            for tk, bk in cur.fetchall():
                tk_s = str(tk)
                if tk_s not in presence_bits:
                    continue
                nk = _bucket_key(bk, gran)
                nk = _norm_utc(nk)
                idx = _column_for_bucket(nk)
                if idx is not None:
                    presence_bits[tk_s][idx] = 1

    merged: list[TickerDashboardRow] = []
    completeness_vals: list[float] = []

    risk_by_ticker: dict[str, TickerRiskRow] = {
        t: TickerRiskRow(ticker=t, return_pct=None, risk_ann_pct=None, sharpe=None, sortino=None) for t in tickers_ordered
    }

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.ticker, array_agg(b.close ORDER BY b.ts)
                FROM ohlcv_bars b
                JOIN symbols s ON s.id = b.symbol_id
                WHERE b.interval = %s AND b.source = %s
                  AND b.ts >= %s AND b.ts <= %s
                  AND s.ticker = ANY(%s)
                GROUP BY s.id, s.ticker
                """,
                (interval, source, t0, t1, tickers_ordered),
            )
            for tk, closes in cur.fetchall():
                arr = np.asarray(closes, dtype=float)
                close_s = pd.Series(arr)
                ret_pct, risk_ann, sharpe, sortino = per_bar_return_stats_with_ppy(close_s, ppy)
                risk_by_ticker[str(tk)] = TickerRiskRow(
                    ticker=str(tk),
                    return_pct=ret_pct,
                    risk_ann_pct=risk_ann,
                    sharpe=sharpe,
                    sortino=sortino,
                )

    for tk in tickers_ordered:
        raw_bits = presence_bits[tk]
        bits = _compress_bits_row(raw_bits, step=step)
        filled = sum(bits)
        n_b = len(bits)
        pct = (100.0 * filled / n_b) if n_b else 0.0
        completeness_vals.append(pct)
        first_ts, last_ts, raw_cnt = sym_stats[tk]
        rk = risk_by_ticker[tk]
        merged.append(
            TickerDashboardRow(
                ticker=tk,
                first_ts=pd.Timestamp(first_ts).isoformat() if first_ts is not None else None,
                last_ts=pd.Timestamp(last_ts).isoformat() if last_ts is not None else None,
                raw_bar_count=raw_cnt,
                completeness_pct=round(pct, 4),
                longest_run_buckets=longest_contiguous_run(bits),
                coverage=bits,
                return_pct=rk.return_pct,
                risk_ann_pct=rk.risk_ann_pct,
                sharpe=rk.sharpe,
                sortino=rk.sortino,
            )
        )

    mean_c = float(np.mean(completeness_vals)) if completeness_vals else 0.0
    median_c = float(np.median(completeness_vals)) if completeness_vals else 0.0
    hist = completeness_histogram(completeness_vals, bins=10)

    per_risk = [
        TickerRiskRow(
            ticker=r.ticker,
            return_pct=r.return_pct,
            risk_ann_pct=r.risk_ann_pct,
            sharpe=r.sharpe,
            sortino=r.sortino,
        )
        for r in merged
    ]

    n_buck = len(bucket_meta_final)

    sector_counts: list[ClassificationLabelCount] = []
    industry_counts: list[ClassificationLabelCount] = []
    sub_industry_counts: list[ClassificationLabelCount] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH latest AS (
                    SELECT DISTINCT ON (sc.symbol_id)
                        sc.symbol_id,
                        sc.sector,
                        sc.industry,
                        sc.sub_industry
                    FROM symbol_classifications sc
                    WHERE sc.source = 'yfinance'
                    ORDER BY sc.symbol_id, sc.as_of DESC
                )
                SELECT
                    COALESCE(l.sector, 'Unknown'),
                    COALESCE(l.industry, 'Unknown'),
                    COALESCE(l.sub_industry, 'Unknown')
                FROM symbols s
                LEFT JOIN latest l ON l.symbol_id = s.id
                WHERE s.ticker = ANY(%s)
                """,
                (tickers_ordered,),
            )
            cls_rows = cur.fetchall()
    sec_c: Counter[str] = Counter()
    ind_c: Counter[str] = Counter()
    sub_c: Counter[str] = Counter()
    for sec, ind, sub in cls_rows:
        sec_c[str(sec)] += 1
        ind_c[str(ind)] += 1
        sub_c[str(sub)] += 1

    def _sorted_counts(c: Counter[str]) -> list[ClassificationLabelCount]:
        items = sorted(c.items(), key=lambda x: (-x[1], x[0]))
        return [ClassificationLabelCount(label=k, count=v) for k, v in items]

    sector_counts = _sorted_counts(sec_c)
    industry_counts = _sorted_counts(ind_c)
    sub_industry_counts = _sorted_counts(sub_c)

    return DataDashboardResponse(
        interval=interval,
        source=source,
        bucket_granularity=gran,
        bucket_auto_subsampled=subsampled,
        reference_start=_norm_utc(t0).isoformat(),
        reference_end=_norm_utc(t1).isoformat(),
        bucket_count=n_buck,
        ticker_count=len(merged),
        truncated=truncated,
        aggregate_mean_completeness_pct=round(mean_c, 4),
        aggregate_median_completeness_pct=round(median_c, 4),
        completeness_histogram=hist,
        buckets=bucket_meta_final,
        tickers=merged,
        per_ticker_metrics=per_risk,
        bar_unit=str(bar_spec.unit.value),
        bar_step=int(bar_spec.step),
        periods_per_year=float(ppy),
        max_buckets=max_buckets,
        sector_counts=sector_counts,
        industry_counts=industry_counts,
        sub_industry_counts=sub_industry_counts,
    )
