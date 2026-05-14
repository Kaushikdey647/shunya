"""Resolve saved universe_id on BacktestCreate into fin_ts ticker_list (Timescale-only)."""

from __future__ import annotations

import logging

from api.index_universe import tickers_with_ohlcv_in_window, validate_ohlcv_window
from api.repositories import universes as universes_repo
from api.schemas.models import BacktestCreate, BarSpecModel
from shunya.data.timeframes import BarSpec, BarUnit, default_bar_spec
from shunya.errors import ErrorCode, ShunyaError
from shunya.data.timescale.intervals import bar_spec_to_interval_key

_LOG = logging.getLogger(__name__)


def _interval_key_from_fin_ts_bar_spec(bar_spec: BarSpecModel | None) -> str:
    if bar_spec is None:
        return bar_spec_to_interval_key(default_bar_spec())
    spec = BarSpec(unit=BarUnit(bar_spec.unit), step=int(bar_spec.step))
    return bar_spec_to_interval_key(spec)


def resolve_universe_backtest_if_needed(body: BacktestCreate) -> BacktestCreate:
    """
    When ``universe_id`` is set: load members from ``api_universe_members``, require
    ``benchmark_ticker``, force Timescale-only ``fin_ts``, validate OHLCV like index jobs.
    """
    raw_uid = body.universe_id
    if raw_uid is None or not str(raw_uid).strip():
        return body

    uid = str(raw_uid).strip()
    if not universes_repo.universe_exists(uid):
        raise ShunyaError(
            f"Universe {uid!r} not found.",
            code=ErrorCode.BACKTEST_UNIVERSE_NOT_FOUND,
            http_status=404,
        )

    tickers = universes_repo.constituent_tickers(uid)
    if not tickers:
        raise ShunyaError(
            f"No members in universe {uid!r}. Add tickers via POST /universes/{{id}}/members.",
            code=ErrorCode.BACKTEST_UNIVERSE_NO_MEMBERS,
            http_status=400,
        )

    bench = (body.benchmark_ticker or "").strip()
    if not bench:
        raise ShunyaError(
            "benchmark_ticker is required for universe_id backtests.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        )

    interval = _interval_key_from_fin_ts_bar_spec(body.fin_ts.bar_spec)
    start_d = body.fin_ts.start_date
    end_d = body.fin_ts.end_date

    try:
        validate_ohlcv_window(
            [bench],
            start_date=start_d,
            end_date=end_d,
            interval=interval,
        )
    except ValueError as exc:
        raise ShunyaError(
            (
                f"Benchmark {bench!r} has no OHLCV in the backtest window ({start_d!r} .. {end_d!r}, "
                f"interval={interval!r}). Ingest it first. Original error: {exc}"
            ),
            code=ErrorCode.BACKTEST_UNIVERSE_OHLCV,
            http_status=400,
        ) from exc

    if body.omit_universe_members_missing_ohlcv:
        covered = tickers_with_ohlcv_in_window(
            tickers,
            start_date=start_d,
            end_date=end_d,
            interval=interval,
        )
        dropped = len(tickers) - len(covered)
        if not covered:
            raise ShunyaError(
                (
                    "No universe members have OHLCV in the backtest window after filtering; "
                    "ingest bars or disable omit_universe_members_missing_ohlcv."
                ),
                code=ErrorCode.BACKTEST_UNIVERSE_OHLCV,
                http_status=400,
            )
        if dropped:
            _LOG.info(
                "Universe %s: omitting %d/%d members missing OHLCV in [%s, %s)",
                uid,
                dropped,
                len(tickers),
                start_d,
                end_d,
            )
        tickers = covered
    else:
        universe_plus_bench = sorted(set(tickers + [bench]))
        try:
            validate_ohlcv_window(
                universe_plus_bench,
                start_date=start_d,
                end_date=end_d,
                interval=interval,
            )
        except ValueError as exc:
            raise ShunyaError(
                str(exc),
                code=ErrorCode.BACKTEST_UNIVERSE_OHLCV,
                http_status=400,
            ) from exc

    ft = body.fin_ts.model_copy(
        update={
            "ticker_list": tickers,
            "market_data_provider": "timescale",
            "attach_yfinance_classifications": False,
        }
    )
    return body.model_copy(
        update={
            "universe_id": uid,
            "fin_ts": ft,
            "benchmark_ticker": bench,
        }
    )
