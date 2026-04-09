from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from finta import TA

from .providers import (
    MarketDataProvider,
    YFinanceMarketDataProvider,
    fetch_yfinance_classifications,
)
from .fundamentals import (
    FinanceToolkitFundamentalDataProvider,
    FundamentalDataProvider,
    align_fundamental_panel_to_panel_index,
    validate_fundamental_fields,
)
from .timeframes import (
    BarIndexPolicy,
    BarSpec,
    BarUnit,
    build_trading_calendar,
    bar_spec_is_intraday,
    default_bar_index_policy,
    default_bar_spec,
    normalize_bar_timestamp,
    trading_time_distance,
)
from .validation import bounds_for_validation, validate_core_ohlcv_coverage
from ..utils import indicators as _indicators

FeatureMode = Literal["full", "ohlcv_only"]
TradingAxisMode = Literal["observed", "canonical"]

# Longest warm-up among default indicators in :meth:`finTs._add_features_full` (SMA_200).
_FULL_FEATURE_MIN_BARS = 200

def _series_for_plot_break_calendar_gaps(
    s: pd.Series,
    *,
    gap_threshold: pd.Timedelta,
) -> pd.Series:
    """
    Insert NaN points where consecutive index gaps exceed ``gap_threshold`` so line
    plots do not draw straight segments across missing sessions (e.g. overnight).
    """
    s = s.sort_index()
    if len(s) < 2:
        return s
    idx_list: List[pd.Timestamp] = []
    val_list: List[float] = []
    prev_ts: Optional[pd.Timestamp] = None
    for ts, val in s.items():
        ts = pd.Timestamp(ts)
        if prev_ts is not None and (ts - prev_ts) > gap_threshold:
            idx_list.append(prev_ts + (ts - prev_ts) / 2)
            val_list.append(float("nan"))
        idx_list.append(ts)
        v = float(val) if pd.notna(val) else float("nan")
        val_list.append(v)
        prev_ts = ts
    out = pd.Series(
        val_list,
        index=pd.DatetimeIndex(idx_list, name=getattr(s.index, "name", None)),
    )
    return out.sort_index()


_OHLCV_EXCLUDE_CORR = frozenset({"Open", "High", "Low", "Close", "Adj Close", "Volume"})
_CLASSIFICATION_COLUMNS = ("Sector", "Industry", "SubIndustry")
_CLASSIFICATION_FALLBACKS = {
    "Sector": "UnknownSector",
    "Industry": "UnknownIndustry",
    "SubIndustry": "UnknownSubIndustry",
}


@dataclass(frozen=True)
class PanelAlignReport:
    """
    Result of :meth:`finTs.align_universe`: common trading calendar and ticker survival.
    """

    calendar: pd.DatetimeIndex
    kept_tickers: Tuple[str, ...]
    dropped_tickers: Tuple[str, ...]
    drop_reasons: Tuple[Tuple[str, str], ...]  # (ticker, reason)

    def as_dict(self) -> Dict[str, Union[List[str], Dict[str, str], int, None]]:
        nb = int(len(self.calendar))
        return {
            "calendar_start": str(self.calendar.min()) if len(self.calendar) else None,
            "calendar_end": str(self.calendar.max()) if len(self.calendar) else None,
            "n_days": nb,
            "n_bars": nb,
            "kept_tickers": list(self.kept_tickers),
            "dropped_tickers": list(self.dropped_tickers),
            "drop_reasons": {t: r for t, r in self.drop_reasons},
        }


@dataclass(frozen=True)
class PanelQADiagnostics:
    duplicate_rows: int
    missing_ticker_dates: int
    stale_days_from_last_date: int
    invalid_ohlcv_rows: int
    warnings: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Union[int, List[str]]]:
        return {
            "duplicate_rows": self.duplicate_rows,
            "missing_ticker_dates": self.missing_ticker_dates,
            "stale_days_from_last_date": self.stale_days_from_last_date,
            "invalid_ohlcv_rows": self.invalid_ohlcv_rows,
            "warnings": list(self.warnings),
        }


class finTs:
    """
    Download OHLCV via a :class:`~shunya.data.providers.MarketDataProvider` (default: Yahoo)
    and attach technical / stationary features.

    ``bar_spec`` sets bar cadence for the provider (default: one trading day per bar); see
    :class:`~shunya.data.timeframes.BarSpec`.

    **Strict loading (defaults on):** ``strict_provider_universe`` fails if any requested
    ticker is missing from the provider frame; ``strict_ohlcv`` fails on non-finite core
    OHLCV or index outside ``[start_date, end_date]``; ``strict_empty`` fails on an empty
    download when symbols were requested. See :func:`~shunya.data.validation.validate_core_ohlcv_coverage`.

    **Features:** ``feature_mode='full'`` adds rolling indicators (needs **≥200** bars
    per series for ``SMA_200`` unless ``require_history_bars`` is set lower). Use
    ``feature_mode='ohlcv_only'`` for VWAP proxy + OHLCV only (e.g. short intraday samples).

    Raw ``Open``, ``High``, ``Low``, ``Close``, and ``Volume`` are kept on the frame and
    are the first columns in :obj:`shunya.utils.indicators.STRATEGY_FEATURES` / ``IX_LIVE``
    for :class:`shunya.algorithm.finstrat.FinStrat` panels.

    The panel index level named ``Date`` holds the **bar timestamp** (midnight for daily-like
    bars in ``bar_index_policy.timezone``, or that zone's wall clock for intraday after
    provider normalization).

    ``bar_index_policy`` defaults to NY session clock with a naive index; pass
    ``BarIndexPolicy(timezone="UTC")`` (and optionally ``daily_anchor="utc"`` for dailies)
    for legacy UTC-naive panels.
    """

    def __init__(
        self,
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        ticker_list: Union[str, List[str]],
        session: Optional[requests.Session] = None,
        market_data: Optional[MarketDataProvider] = None,
        classifications: Optional[Mapping[str, Mapping[str, str]]] = None,
        attach_yfinance_classifications: bool = True,
        fundamental_data: Optional[FundamentalDataProvider] = None,
        attach_fundamentals: bool = False,
        fundamental_fields: Optional[Sequence[str]] = None,
        fundamental_quarterly: bool = True,
        bar_spec: Optional[BarSpec] = None,
        bar_index_policy: Optional[BarIndexPolicy] = None,
        *,
        strict_provider_universe: bool = True,
        strict_ohlcv: bool = True,
        strict_empty: bool = True,
        feature_mode: FeatureMode = "full",
        require_history_bars: Optional[int] = None,
        trading_axis_mode: TradingAxisMode = "observed",
        strict_trading_grid: bool = False,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.session = session
        if isinstance(ticker_list, str):
            self.ticker_list = [ticker_list]
        else:
            self.ticker_list = list(ticker_list)

        self.bar_spec: BarSpec = bar_spec if bar_spec is not None else default_bar_spec()
        self._bar_index_policy: BarIndexPolicy = (
            bar_index_policy
            if bar_index_policy is not None
            else default_bar_index_policy()
        )
        self._aligned_calendar: Optional[pd.DatetimeIndex] = None
        self._fundamental_feature_columns: Tuple[str, ...] = tuple()
        self._strict_provider_universe = bool(strict_provider_universe)
        self._strict_ohlcv = bool(strict_ohlcv)
        self._strict_empty = bool(strict_empty)
        if trading_axis_mode not in ("observed", "canonical"):
            raise ValueError(
                f"trading_axis_mode must be 'observed' or 'canonical', got {trading_axis_mode!r}"
            )
        self._trading_axis_mode: TradingAxisMode = trading_axis_mode
        self._strict_trading_grid = bool(strict_trading_grid)
        if feature_mode not in ("full", "ohlcv_only"):
            raise ValueError(f"feature_mode must be 'full' or 'ohlcv_only', got {feature_mode!r}")
        self._feature_mode: FeatureMode = feature_mode
        self._require_history_bars_opt: Optional[int] = (
            int(require_history_bars) if require_history_bars is not None else None
        )
        if self._require_history_bars_opt is not None and self._require_history_bars_opt < 1:
            raise ValueError("require_history_bars must be >= 1 when set")

        provider = market_data or YFinanceMarketDataProvider(session=session)
        raw = provider.download(
            self.ticker_list,
            self.start_date,
            self.end_date,
            bar_spec=self.bar_spec,
            bar_index_policy=self._bar_index_policy,
        )
        self._validate_provider_output(raw)
        validate_core_ohlcv_coverage(
            raw,
            ticker_list=self.ticker_list,
            start=self.start_date,
            end=self.end_date,
            bar_spec=self.bar_spec,
            strict_provider_universe=self._strict_provider_universe,
            strict_ohlcv=self._strict_ohlcv,
            strict_empty=self._strict_empty,
            bar_index_policy=self._bar_index_policy,
            strict_trading_grid=self._strict_trading_grid,
        )
        self._ingest_raw_ohlcv(raw)
        self._validate_feature_history_bars()

        class_map: Dict[str, Dict[str, str]] = {
            str(t): dict(v) for t, v in (classifications or {}).items()
        }
        if attach_yfinance_classifications:
            fetched = fetch_yfinance_classifications(self.ticker_list, session=session)
            for t, fields in fetched.items():
                merged = dict(class_map.get(t, {}))
                merged.update(fields)
                class_map[t] = merged
        self._attach_classifications(class_map)
        if attach_fundamentals:
            provider = fundamental_data or FinanceToolkitFundamentalDataProvider()
            self._attach_fundamentals(
                provider,
                fields=fundamental_fields,
                quarterly=fundamental_quarterly,
            )

    @property
    def bar_index_policy(self) -> BarIndexPolicy:
        """Normalization used for the panel index (see :class:`~shunya.data.timeframes.BarIndexPolicy`)."""
        return self._bar_index_policy

    @property
    def trading_axis_mode(self) -> TradingAxisMode:
        """Default time axis mode used by calendar/lag helpers."""
        return getattr(self, "_trading_axis_mode", "observed")

    @property
    def fundamental_feature_columns(self) -> Tuple[str, ...]:
        """Attached time-varying fundamental columns that are safe to expose in ``AlphaContext``."""
        return getattr(self, "_fundamental_feature_columns", tuple())

    def _validate_provider_output(self, raw: pd.DataFrame) -> None:
        """
        Lightweight schema checks for provider outputs before feature engineering.
        """
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                f"market_data.download(...) must return pandas.DataFrame, got {type(raw)!r}"
            )
        if raw.empty:
            return

        # Provider contract: Date-like index per :class:`~shunya.data.timeframes.BarIndexPolicy`.
        try:
            pd.DatetimeIndex(pd.to_datetime(raw.index))
        except Exception as exc:
            raise ValueError(
                "Provider output index must be DatetimeIndex-compatible."
            ) from exc

        required = {"Open", "High", "Low", "Close", "Volume"}
        if isinstance(raw.columns, pd.MultiIndex):
            if raw.columns.nlevels < 2:
                raise ValueError(
                    "Provider multi-ticker output must use MultiIndex columns "
                    "with at least two levels: (Ticker, Field)."
                )
            tickers = raw.columns.get_level_values(0).unique().tolist()
            if not tickers:
                raise ValueError("Provider returned MultiIndex columns without tickers.")
            for t in tickers:
                cols = set(raw[t].columns.tolist())
                missing = sorted(required - cols)
                if missing:
                    raise ValueError(
                        f"Provider output for ticker {t!r} missing OHLCV columns: {missing}"
                    )
            return

        cols = set(raw.columns.tolist())
        missing = sorted(required - cols)
        if missing:
            raise ValueError(f"Provider output missing OHLCV columns: {missing}")

    def _featurize(self, part: pd.DataFrame) -> pd.DataFrame:
        if self._feature_mode == "ohlcv_only":
            return finTs._add_features_ohlcv_only(part)
        return finTs._add_features_full(part)

    def _ingest_raw_ohlcv(self, raw: pd.DataFrame) -> None:
        if raw.empty:
            self.df = raw
            return

        frames: List[pd.DataFrame] = []
        keys: List[str] = []

        if isinstance(raw.columns, pd.MultiIndex):
            available = set(raw.columns.get_level_values(0).unique().tolist())
            for t in self.ticker_list:
                if t not in available:
                    if self._strict_provider_universe:
                        raise ValueError(
                            f"strict_provider_universe: ticker {t!r} missing from provider frame "
                            "(should have been caught in validate_core_ohlcv_coverage)."
                        )
                    continue
                part = self._featurize(raw[t].copy())
                frames.append(part)
                keys.append(t)
        else:
            if len(self.ticker_list) != 1:
                raise ValueError(
                    "Multiple tickers were requested but the provider returned a "
                    "single-level column index; check ticker symbols and dates."
                )
            self.df = self._featurize(raw.copy())
            return

        if not frames:
            self.df = pd.DataFrame()
            return

        if len(frames) == 1 and len(self.ticker_list) == 1:
            self.df = frames[0]
        else:
            self.df = pd.concat(frames, keys=keys, names=["Ticker", None])
            self.df.index = self.df.index.set_names(["Ticker", "Date"])

    def _validate_feature_history_bars(self) -> None:
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return
        if self._feature_mode == "ohlcv_only":
            need: Optional[int] = self._require_history_bars_opt
        else:
            need = (
                self._require_history_bars_opt
                if self._require_history_bars_opt is not None
                else _FULL_FEATURE_MIN_BARS
            )
        if need is None:
            return
        if isinstance(self.df.index, pd.MultiIndex):
            counts = self.df.index.get_level_values(0).value_counts()
            m = int(counts.min())
        else:
            m = len(self.df.index)
        if m < need:
            raise ValueError(
                f"insufficient_bars_for_features: minimum bars per series is {m}, "
                f"but feature_mode={self._feature_mode!r} requires at least {need}. "
                "Widen the date range, use feature_mode='ohlcv_only', or set require_history_bars "
                "lower only if you accept meaningless rolling indicators."
            )

    @staticmethod
    def _add_features_ohlcv_only(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "VWAP" in out.columns:
            out["VWAP"] = pd.to_numeric(out["VWAP"], errors="coerce")
        else:
            out["VWAP"] = (out["High"] + out["Low"] + out["Close"]) / 3.0
        return out

    @staticmethod
    def _add_features_full(df: pd.DataFrame) -> pd.DataFrame:
        out = df

        if "VWAP" in out.columns:
            out["VWAP"] = pd.to_numeric(out["VWAP"], errors="coerce")
        else:
            # Daily fallback when provider VWAP is unavailable.
            out["VWAP"] = (out["High"] + out["Low"] + out["Close"]) / 3.0

        out["SMA_50"] = TA.SMA(out, 50)
        out["SMA_200"] = TA.SMA(out, 200)
        out["RSI_14"] = TA.RSI(out, 14)

        macd_df = TA.MACD(out)
        out["MACD"] = macd_df["MACD"]
        out["MACD_Signal"] = macd_df["SIGNAL"]

        bbands = TA.BBANDS(out)
        out["BB_Upper"] = bbands["BB_UPPER"]
        out["BB_Lower"] = bbands["BB_LOWER"]
        out["ATR_14"] = TA.ATR(out, 14)

        out["Future_1d_Ret"] = np.log(out["Close"].shift(-1) / out["Close"])

        out["Log_Ret"] = np.log(out["Close"] / out["Close"].shift(1))
        out["Dist_SMA50"] = (out["Close"] - out["SMA_50"]) / out["SMA_50"]
        out["Dist_SMA200"] = (out["Close"] - out["SMA_200"]) / out["SMA_200"]
        out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / out["SMA_50"]
        denom = (out["BB_Upper"] - out["BB_Lower"]).replace(0, np.nan)
        out["BB_Position"] = (out["Close"] - out["BB_Lower"]) / denom
        out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]
        out["ATR_Norm"] = out["ATR_14"] / out["Close"]
        out["Vol_Change"] = (
            out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        return out

    def _attach_classifications(
        self,
        class_map: Mapping[str, Mapping[str, str]],
    ) -> None:
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return

        if isinstance(self.df.index, pd.MultiIndex):
            tickers = self.df.index.get_level_values(0)
            for col in _CLASSIFICATION_COLUMNS:
                fallback = _CLASSIFICATION_FALLBACKS[col]
                self.df[col] = [
                    str(class_map.get(str(t), {}).get(col) or fallback) for t in tickers
                ]
            return

        ticker = self.ticker_list[0] if self.ticker_list else ""
        for col in _CLASSIFICATION_COLUMNS:
            fallback = _CLASSIFICATION_FALLBACKS[col]
            val = str(class_map.get(str(ticker), {}).get(col) or fallback)
            self.df[col] = val

    def _attach_fundamentals(
        self,
        provider: FundamentalDataProvider,
        *,
        fields: Optional[Sequence[str]] = None,
        quarterly: bool = True,
    ) -> None:
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            return
        specs = validate_fundamental_fields(fields)
        requested = [spec.column for spec in specs]
        periodic = provider.fetch(
            self.ticker_list,
            self.start_date,
            self.end_date,
            fields=requested,
            quarterly=quarterly,
            bar_spec=self.bar_spec,
        )
        if not isinstance(periodic, pd.DataFrame):
            raise TypeError(
                f"fundamental_data.fetch(...) must return pandas.DataFrame, got {type(periodic)!r}"
            )
        if periodic.empty:
            aligned = pd.DataFrame(index=self.df.index, columns=requested, dtype=float)
        else:
            aligned = align_fundamental_panel_to_panel_index(periodic, self.df.index)
        aligned = aligned.reindex(self.df.index)
        missing = [col for col in requested if col not in aligned.columns]
        for col in missing:
            aligned[col] = np.nan
        for col in requested:
            self.df[col] = pd.to_numeric(aligned[col], errors="coerce").to_numpy(dtype=float)
        self._fundamental_feature_columns = tuple(requested)

    def _require_nonempty_df(self) -> None:
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            raise ValueError("No data loaded; dataframe is empty.")

    def get_trading_calendar(
        self, *, mode: Optional[TradingAxisMode] = None
    ) -> pd.DatetimeIndex:
        """
        Ordered bar timestamps for the panel (calendar days for daily-like bars,
        full clock time for intraday :attr:`bar_spec`).

        After :meth:`align_universe`, returns the intersection calendar shared by all
        tickers. Before alignment, returns the sorted **union** of bar timestamps
        in the index.
        """
        self._require_nonempty_df()
        use_mode = mode if mode is not None else getattr(self, "_trading_axis_mode", "observed")
        if use_mode not in ("observed", "canonical"):
            raise ValueError(f"mode must be 'observed' or 'canonical', got {use_mode!r}")
        if use_mode == "canonical":
            return build_trading_calendar(
                self.start_date,
                self.end_date,
                self.bar_spec,
                policy=self._bar_index_policy,
            )
        if self._aligned_calendar is not None:
            return self._aligned_calendar
        df = self.df
        intraday = bar_spec_is_intraday(self.bar_spec)
        if not isinstance(df.index, pd.MultiIndex):
            raw = pd.DatetimeIndex(pd.to_datetime(df.index))
            if intraday:
                d = raw.unique()
            else:
                d = raw.normalize().unique()
            return pd.DatetimeIndex(d).sort_values()
        raw = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values("Date")))
        if intraday:
            dates = raw.unique()
        else:
            dates = raw.normalize().unique()
        return pd.DatetimeIndex(dates).sort_values()

    def trading_session_key(self, ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
        """
        Calendar anchor for intraday **session** grouping in :attr:`_bar_index_policy` timezone
        (midnight of the bar's local trading date).

        Used to forbid ``signal_delay`` lookback across overnight gaps when
        ``intraday_session_isolated_lag`` is enabled on :class:`~shunya.algorithm.finstrat.FinStrat`.
        """
        t = pd.Timestamp(ts)
        pol = self._bar_index_policy
        if t.tzinfo is None:
            loc = t.tz_localize(pol.timezone)
        else:
            loc = t.tz_convert(pol.timezone)
        day0 = loc.normalize()
        if pol.naive:
            return pd.Timestamp(day0.tz_localize(None))
        return pd.Timestamp(day0)

    def trading_distance(
        self,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        *,
        mode: TradingAxisMode = "canonical",
    ) -> tuple[int, float]:
        """
        Trading-time distance between two timestamps as ``(bar_steps, trading_seconds)``.

        ``mode='canonical'`` uses the canonical US-equities grid, while
        ``mode='observed'`` measures distance on the panel's observed calendar.
        """
        if mode == "canonical":
            return trading_time_distance(
                start, end, self.bar_spec, policy=self._bar_index_policy
            )
        if mode != "observed":
            raise ValueError(f"mode must be 'observed' or 'canonical', got {mode!r}")
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        sign = 1
        if t1 < t0:
            t0, t1 = t1, t0
            sign = -1
        cal = self.get_trading_calendar(mode="observed").sort_values()
        p0 = int(cal.searchsorted(t0, side="left"))
        p1 = int(cal.searchsorted(t1, side="left"))
        if p0 >= len(cal) or cal[p0] != t0:
            raise ValueError(f"start timestamp {t0!s} is not on observed trading calendar")
        if p1 >= len(cal) or cal[p1] != t1:
            raise ValueError(f"end timestamp {t1!s} is not on observed trading calendar")
        bars = (p1 - p0) * sign
        if bar_spec_is_intraday(self.bar_spec):
            if self.bar_spec.unit == BarUnit.SECONDS:
                step_seconds = float(self.bar_spec.step)
            elif self.bar_spec.unit == BarUnit.MINUTES:
                step_seconds = float(self.bar_spec.step * 60)
            elif self.bar_spec.unit == BarUnit.HOURS:
                step_seconds = float(self.bar_spec.step * 3600)
            else:
                step_seconds = 0.0
            seconds = abs(p1 - p0) * step_seconds
        else:
            seconds = abs(p1 - p0) * (6.5 * 60.0 * 60.0)
        return bars, float(sign * seconds)

    def align_universe(
        self,
        required_columns: Sequence[str],
        *,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        reference_ticker: Optional[str] = None,
        on_bad_ticker: Literal["drop", "raise"] = "drop",
    ) -> PanelAlignReport:
        """
        Restrict ``df`` and ``ticker_list`` to tickers that have **finite** values in
        ``required_columns`` on every date in a **common** calendar (intersection of
        dates per ticker in ``[start, end]``, optionally clipped to ``reference_ticker``'s
        dates).

        Mutates ``self.df``, ``self.ticker_list``, and sets ``_aligned_calendar``.
        """
        self._require_nonempty_df()
        df = self.df
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("align_universe requires MultiIndex (Ticker, Date) data.")
        if tuple(df.index.names) != ("Ticker", "Date"):
            raise ValueError(
                f"Expected index names ('Ticker', 'Date'), got {tuple(df.index.names)!r}"
            )

        req = list(required_columns)
        missing_cols = [c for c in req if c not in df.columns]
        if missing_cols:
            raise KeyError(f"required_columns missing from dataframe: {missing_cols}")

        intraday = bar_spec_is_intraday(self.bar_spec)
        raw_start = pd.Timestamp(start) if start is not None else pd.Timestamp(self.start_date)
        raw_end = pd.Timestamp(end) if end is not None else pd.Timestamp(self.end_date)
        t0, t1_incl, t1_open = bounds_for_validation(
            raw_start, raw_end, self.bar_spec, self._bar_index_policy
        )

        level_tickers = df.index.get_level_values(0).unique().tolist()
        ticker_candidates = [t for t in self.ticker_list if t in level_tickers]

        common_set: Optional[set] = None
        if reference_ticker is not None:
            if reference_ticker not in level_tickers:
                raise KeyError(f"reference_ticker {reference_ticker!r} not in panel index")
            ref_sub = df.xs(reference_ticker, level="Ticker").sort_index()
            ref_idx = pd.DatetimeIndex(pd.to_datetime(ref_sub.index))
            if intraday:
                common_set = set(ref_idx[(ref_idx >= t0) & (ref_idx < t1_open)])
            else:
                common_set = set(ref_idx[(ref_idx >= t0) & (ref_idx <= t1_incl)])

        date_sets: List[set] = []
        for t in ticker_candidates:
            sub = df.xs(t, level="Ticker").sort_index()
            idx = pd.DatetimeIndex(pd.to_datetime(sub.index))
            if intraday:
                mask = (idx >= t0) & (idx < t1_open)
            else:
                mask = (idx >= t0) & (idx <= t1_incl)
            part = set(idx[mask])
            if common_set is not None:
                part &= common_set
            date_sets.append(part)

        if not date_sets:
            self._aligned_calendar = pd.DatetimeIndex([])
            self.df = df.iloc[0:0]
            self.ticker_list = []
            return PanelAlignReport(
                calendar=self._aligned_calendar,
                kept_tickers=tuple(),
                dropped_tickers=tuple(ticker_candidates),
                drop_reasons=tuple(
                    (t, "no overlapping dates in range") for t in ticker_candidates
                ),
            )

        common_dates = set.intersection(*date_sets)
        calendar = pd.DatetimeIndex(sorted(common_dates))
        if len(calendar) == 0:
            raise ValueError(
                "Empty calendar after intersecting ticker date sets; widen dates or tickers."
            )

        kept: List[str] = []
        dropped: List[str] = []
        reasons: List[Tuple[str, str]] = []

        for t in ticker_candidates:
            bad = False
            reason = ""
            for d in calendar:
                key = (t, d)
                if key not in df.index:
                    bad = True
                    reason = f"missing row for date {d!s}"
                    break
                row = df.loc[key]
                if isinstance(row, pd.DataFrame):
                    bad = True
                    reason = "duplicate index row"
                    break
                vals = row[req].to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    bad = True
                    reason = "non-finite required column value"
                    break
            if bad:
                dropped.append(t)
                reasons.append((t, reason))
            else:
                kept.append(t)

        if on_bad_ticker == "raise" and dropped:
            msg = "; ".join(f"{t}: {r}" for t, r in reasons)
            raise ValueError(f"align_universe: bad tickers with on_bad_ticker='raise': {msg}")

        if len(kept) < 2:
            raise ValueError(
                f"Need at least two tickers after alignment; kept={kept}, dropped={dropped}"
            )

        new_idx = pd.MultiIndex.from_product([kept, calendar], names=["Ticker", "Date"])
        aligned_df = df.reindex(new_idx)
        na_block = aligned_df[req].isna().any(axis=1)
        if bool(na_block.any()):
            bad_sample = aligned_df.index[na_block].tolist()[:5]
            raise ValueError(f"align_universe: unexpected NaNs after reindex at {bad_sample!r}")

        self.df = aligned_df.sort_index()
        self.ticker_list = kept
        self._aligned_calendar = calendar

        return PanelAlignReport(
            calendar=calendar,
            kept_tickers=tuple(kept),
            dropped_tickers=tuple(dropped),
            drop_reasons=tuple(reasons),
        )

    def execution_lag_calendar_date(
        self,
        execution_date: Union[str, pd.Timestamp],
        *,
        lag: int,
        forbid_cross_session: bool = False,
        calendar_mode: Optional[TradingAxisMode] = None,
    ) -> pd.Timestamp:
        """
        Map execution **bar time** to the signal bar ``lag`` steps earlier on
        :meth:`get_trading_calendar`.

        ``lag`` counts **bars** on the panel calendar (for daily data, that is
        prior trading days; for intraday, prior bars of the same :attr:`bar_spec`).
        ``lag == 1`` means use the prior bar's features for this execution bar.

        When ``forbid_cross_session`` is True and :attr:`bar_spec` is intraday,
        ``lag > 0`` requires the signal bar to share the same :meth:`trading_session_key`
        as the execution bar (so the first bar of a session cannot use the prior session's last bar).
        """
        if lag < 0:
            raise ValueError("lag must be non-negative")
        dt = normalize_bar_timestamp(execution_date, self.bar_spec)
        cal = self.get_trading_calendar(mode=calendar_mode).sort_values()
        if lag == 0:
            pos = int(cal.searchsorted(dt, side="left"))
            if pos >= len(cal) or cal[pos] != dt:
                raise ValueError(f"execution_date {dt} not on trading calendar")
            return pd.Timestamp(cal[pos])
        pos = int(cal.searchsorted(dt, side="left"))
        if pos >= len(cal) or cal[pos] != dt:
            raise ValueError(f"execution_date {dt} not on trading calendar")
        sig_pos = pos - lag
        if sig_pos < 0:
            raise ValueError(
                f"lag={lag} would read before calendar start {cal[0]} for execution {dt}"
            )
        sig_ts = pd.Timestamp(cal[sig_pos])
        if forbid_cross_session and bar_spec_is_intraday(self.bar_spec) and lag > 0:
            if self.trading_session_key(sig_ts) != self.trading_session_key(dt):
                raise ValueError(
                    "cross_session_signal_lag: signal bar "
                    f"{sig_ts!s} and execution bar {dt!s} fall on different trading_session_key; "
                    "use FinStrat(signal_delay=0) for the session open, or disable intraday_session_isolated_lag."
                )
        return sig_ts

    def execution_lag_bars(
        self,
        execution_date: Union[str, pd.Timestamp],
        *,
        lag: int,
        forbid_cross_session: bool = False,
        calendar_mode: Optional[TradingAxisMode] = None,
    ) -> pd.Timestamp:
        """Alias of :meth:`execution_lag_calendar_date` (bar-based lag)."""
        return self.execution_lag_calendar_date(
            execution_date,
            lag=lag,
            forbid_cross_session=forbid_cross_session,
            calendar_mode=calendar_mode,
        )

    def qa_diagnostics(
        self,
        *,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        max_stale_days: int = 7,
    ) -> PanelQADiagnostics:
        """
        Basic panel QA checks for research and pre-trade validation.
        """
        self._require_nonempty_df()
        df = self.df
        warnings: List[str] = []

        intraday = bar_spec_is_intraday(self.bar_spec)
        if isinstance(df.index, pd.MultiIndex):
            duplicate_rows = int(df.index.duplicated().sum())
            raw_dates = pd.DatetimeIndex(pd.to_datetime(df.index.get_level_values("Date")))
            if intraday:
                uniq_dates = pd.DatetimeIndex(raw_dates.unique())
            else:
                uniq_dates = pd.DatetimeIndex(raw_dates.normalize().unique())
            uniq_tickers = list(df.index.get_level_values("Ticker").unique())
            expected = len(uniq_tickers) * len(uniq_dates)
            missing_ticker_dates = max(expected - int(len(df.index)), 0)
            last_ts = pd.Timestamp(raw_dates.max())
        else:
            duplicate_rows = int(df.index.duplicated().sum())
            raw_single = pd.DatetimeIndex(pd.to_datetime(df.index))
            missing_ticker_dates = 0
            last_ts = pd.Timestamp(raw_single.max())

        if as_of is not None:
            ref = pd.Timestamp(as_of)
        else:
            ref = pd.Timestamp.now(tz="America/New_York").tz_localize(None)

        stale_days = max(0, int((ref.normalize() - last_ts.normalize()).days))

        required = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        invalid_ohlcv_rows = 0
        if required:
            chk = df[required]
            invalid_ohlcv_rows = int(
                (
                    (~np.isfinite(chk.to_numpy(dtype=float))).any(axis=1)
                    | (chk["Volume"].to_numpy(dtype=float) < 0 if "Volume" in chk.columns else False)
                    | (
                        (chk["High"].to_numpy(dtype=float) < chk["Low"].to_numpy(dtype=float))
                        if "High" in chk.columns and "Low" in chk.columns
                        else False
                    )
                ).sum()
            )

        if duplicate_rows > 0:
            warnings.append(f"duplicate_rows={duplicate_rows}")
        if missing_ticker_dates > 0:
            warnings.append(f"missing_ticker_dates={missing_ticker_dates}")
        if (ref - last_ts) > pd.Timedelta(days=max_stale_days):
            warnings.append(
                f"stale_days_exceeds_limit ref_minus_last={ref - last_ts} limit_days={max_stale_days}"
            )
        if invalid_ohlcv_rows > 0:
            warnings.append(f"invalid_ohlcv_rows={invalid_ohlcv_rows}")

        ohlcv_strategy = {c for _, c in _indicators._STRATEGY_OHLCV_SPEC}
        eng_cols = [c for c in _indicators.STRATEGY_FEATURES if c not in ohlcv_strategy]
        max_eng_warnings = 8
        n_eng = 0
        for col in eng_cols:
            if n_eng >= max_eng_warnings:
                break
            if col not in df.columns:
                continue
            if isinstance(df.index, pd.MultiIndex):
                for t in df.index.get_level_values(0).unique():
                    if n_eng >= max_eng_warnings:
                        break
                    s = df.xs(t, level=0)[col]
                    if len(s) > 0 and bool(s.isna().all()):
                        warnings.append(f"engineered_all_nan ticker={t!r} column={col!r}")
                        n_eng += 1
            else:
                if len(df[col]) > 0 and bool(df[col].isna().all()):
                    warnings.append(f"engineered_all_nan column={col!r}")
                    n_eng += 1

        return PanelQADiagnostics(
            duplicate_rows=duplicate_rows,
            missing_ticker_dates=missing_ticker_dates,
            stale_days_from_last_date=stale_days,
            invalid_ohlcv_rows=invalid_ohlcv_rows,
            warnings=tuple(warnings),
        )

    def plot_correlation_heatmap(
        self,
        *,
        columns: Optional[Sequence[str]] = None,
        title: str = "S&P 500 Universe — Stationary Feature Correlation Matrix",
        save_path: Optional[str] = "chart_correlation_heatmap.png",
        figsize: Tuple[float, float] = (11, 9),
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Pearson correlation matrix as a masked seaborn heatmap; works in Jupyter."""
        self._require_nonempty_df()

        numeric = self.df.select_dtypes(include=[np.number])
        if columns is None:
            use_cols = [c for c in numeric.columns if c not in _OHLCV_EXCLUDE_CORR]
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                raise KeyError(f"Unknown columns: {missing}")
            use_cols = list(columns)

        if not use_cols:
            raise ValueError("No columns left for correlation.")

        sample = self.df.loc[:, use_cols].dropna(how="any")
        if sample.empty:
            raise ValueError("All rows dropped after NA removal; cannot compute correlation.")

        corr = sample.corr()
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            ax=ax,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            square=True,
            linewidths=0.4,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        )
        ax.set_title(title, pad=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    def plot_component_returns(
        self,
        *,
        title: str = "Normalized close by component (start = 1)",
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 6),
        show: bool = True,
        legend: bool = False,
        break_calendar_gaps: bool = True,
        max_bar_gap: Optional[pd.Timedelta] = None,
        concat_intraday_sessions: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot each ticker's Close / first Close on shared axes.

        For intraday panels, ``break_calendar_gaps`` inserts NaN across long index gaps
        (default: >1 hour) so lines do not connect overnight or through long halts.
        When ``concat_intraday_sessions`` is True (intraday MultiIndex only), the x-axis
        is a contiguous bar index across all sessions; light vertical lines mark session
        boundaries (see :meth:`trading_session_key`). Gap-breaking is disabled in that mode.

        Set ``legend=True`` to show per-ticker labels (many names clutter small figures).
        Pass ``ax`` to draw into an existing axes (e.g. subplot); ``figsize`` is ignored then.
        """
        self._require_nonempty_df()
        if "Close" not in self.df.columns:
            raise KeyError("Column 'Close' is required.")

        intraday = bar_spec_is_intraday(self.bar_spec)
        use_concat = bool(concat_intraday_sessions) and intraday and isinstance(
            self.df.index, pd.MultiIndex
        )

        gap_thr: Optional[pd.Timedelta] = None
        if (
            break_calendar_gaps
            and intraday
            and not use_concat
        ):
            gap_thr = max_bar_gap or pd.Timedelta(hours=1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if use_concat:
            cal = self.get_trading_calendar().sort_values()
            if len(cal) == 0:
                raise ValueError("Empty trading calendar for concat_intraday_sessions plot.")
            xcoord = np.arange(len(cal), dtype=float)
            boundary_idx = [
                i
                for i in range(1, len(cal))
                if self.trading_session_key(cal[i]) != self.trading_session_key(cal[i - 1])
            ]
            for bi in boundary_idx:
                ax.axvline(bi - 0.5, color="0.75", linewidth=0.8, linestyle=":", zorder=0)
            level_tickers = self.df.index.get_level_values(0).unique()
            for ticker in level_tickers:
                sub = self.df.xs(ticker, level=0, drop_level=True)
                if sub.empty:
                    continue
                close = sub["Close"].sort_index().reindex(cal)
                first = close.dropna().iloc[0] if close.notna().any() else np.nan
                if not np.isfinite(float(first)) or float(first) == 0.0:
                    continue
                y = (close / float(first)).to_numpy(dtype=float)
                ax.plot(
                    xcoord,
                    y,
                    label=str(ticker) if legend else None,
                )
            ax.set_xlabel("Bar index (sessions concatenated)")
            tick_pos = [0] + boundary_idx + ([len(cal) - 1] if len(cal) else [])
            tick_pos = sorted(set(int(p) for p in tick_pos if 0 <= p < len(cal)))
            if len(tick_pos) > 12:
                step = max(1, len(tick_pos) // 12)
                tick_pos = tick_pos[::step]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([str(cal[i]) for i in tick_pos], rotation=45, ha="right")
        elif isinstance(self.df.index, pd.MultiIndex):
            level_tickers = self.df.index.get_level_values(0).unique()
            for ticker in level_tickers:
                sub = self.df.xs(ticker, level=0, drop_level=True)
                if sub.empty:
                    continue
                close = sub["Close"].sort_index()
                denom = close.iloc[0]
                if not np.isfinite(float(denom)) or float(denom) == 0.0:
                    continue
                norm = close / float(denom)
                if gap_thr is not None:
                    norm = _series_for_plot_break_calendar_gaps(norm, gap_threshold=gap_thr)
                norm.plot(ax=ax, label=str(ticker) if legend else None, legend=False)
        else:
            label = self.ticker_list[0] if len(self.ticker_list) == 1 else "Close"
            close = self.df["Close"].sort_index()
            denom = close.iloc[0]
            if not np.isfinite(float(denom)) or float(denom) == 0.0:
                raise ValueError("First Close is missing or zero; cannot normalize.")
            norm = close / float(denom)
            if gap_thr is not None:
                norm = _series_for_plot_break_calendar_gaps(norm, gap_threshold=gap_thr)
            norm.plot(ax=ax, label=label if legend else None, legend=False)

        ax.set_title(title, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized close")
        if legend:
            ax.legend(loc="best", framealpha=0.9, ncol=2, fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax
