from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

_OHLCV_EXCLUDE_CORR = frozenset({"Open", "High", "Low", "Close", "Adj Close", "Volume"})
_CLASSIFICATION_COLUMNS = ("Sector", "Industry", "SubIndustry")
_CLASSIFICATION_FALLBACKS = {
    "Sector": "UnknownSector",
    "Industry": "UnknownIndustry",
    "SubIndustry": "UnknownSubIndustry",
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

    Raw ``Open``, ``High``, ``Low``, ``Close``, and ``Volume`` are kept on the frame and
    are the first columns in :obj:`shunya.utils.indicators.STRATEGY_FEATURES` / ``IX_LIVE``
    for :class:`shunya.algorithm.finstrat.FinStrat` panels.
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
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.session = session
        if isinstance(ticker_list, str):
            self.ticker_list = [ticker_list]
        else:
            self.ticker_list = list(ticker_list)

        provider = market_data or YFinanceMarketDataProvider(session=session)
        raw = provider.download(self.ticker_list, self.start_date, self.end_date)
        self._validate_provider_output(raw)
        self._ingest_raw_ohlcv(raw)

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

        # Provider contract: Date-like index (timezone handled by providers).
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

    def _ingest_raw_ohlcv(self, raw: pd.DataFrame) -> None:
        if raw.empty:
            self.df = raw
            return

        frames: List[pd.DataFrame] = []
        keys: List[str] = []

        if isinstance(raw.columns, pd.MultiIndex):
            available = raw.columns.get_level_values(0).unique().tolist()
            for t in self.ticker_list:
                if t not in available:
                    continue
                part = self._add_features(raw[t].copy())
                frames.append(part)
                keys.append(t)
        else:
            if len(self.ticker_list) != 1:
                raise ValueError(
                    "Multiple tickers were requested but the provider returned a "
                    "single-level column index; check ticker symbols and dates."
                )
            self.df = self._add_features(raw.copy())
            return

        if not frames:
            self.df = pd.DataFrame()
            return

        if len(frames) == 1 and len(self.ticker_list) == 1:
            self.df = frames[0]
        else:
            self.df = pd.concat(frames, keys=keys, names=["Ticker", None])
            self.df.index = self.df.index.set_names(["Ticker", "Date"])

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    def _require_nonempty_df(self) -> None:
        if not isinstance(self.df, pd.DataFrame) or self.df.empty:
            raise ValueError("No data loaded; dataframe is empty.")

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

        if isinstance(df.index, pd.MultiIndex):
            duplicate_rows = int(df.index.duplicated().sum())
            uniq_dates = pd.DatetimeIndex(
                pd.to_datetime(df.index.get_level_values("Date")).normalize().unique()
            )
            uniq_tickers = list(df.index.get_level_values("Ticker").unique())
            expected = len(uniq_tickers) * len(uniq_dates)
            missing_ticker_dates = max(expected - int(len(df.index)), 0)
            last_date = pd.Timestamp(uniq_dates.max()).normalize()
        else:
            duplicate_rows = int(df.index.duplicated().sum())
            idx_dates = pd.DatetimeIndex(pd.to_datetime(df.index).normalize().unique())
            missing_ticker_dates = 0
            last_date = pd.Timestamp(idx_dates.max()).normalize()

        ref = (
            pd.Timestamp(as_of).normalize()
            if as_of is not None
            else pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
        )
        stale_days = int((ref - last_date).days)

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
        if stale_days > max_stale_days:
            warnings.append(f"stale_days_exceeds_limit days={stale_days} limit={max_stale_days}")
        if invalid_ohlcv_rows > 0:
            warnings.append(f"invalid_ohlcv_rows={invalid_ohlcv_rows}")

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
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot each ticker's Close / first Close on shared axes with legend labels."""
        self._require_nonempty_df()
        if "Close" not in self.df.columns:
            raise KeyError("Column 'Close' is required.")

        fig, ax = plt.subplots(figsize=figsize)

        if isinstance(self.df.index, pd.MultiIndex):
            level_tickers = self.df.index.get_level_values(0).unique()
            for ticker in level_tickers:
                sub = self.df.xs(ticker, level=0, drop_level=True)
                if sub.empty:
                    continue
                close = sub["Close"]
                norm = close / close.iloc[0]
                norm.plot(ax=ax, label=str(ticker))
        else:
            label = self.ticker_list[0] if len(self.ticker_list) == 1 else "Close"
            close = self.df["Close"]
            norm = close / close.iloc[0]
            norm.plot(ax=ax, label=label)

        ax.set_title(title, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized close")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax
