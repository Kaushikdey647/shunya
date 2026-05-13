"""Merge env-backed :class:`api.settings.Settings` with DB ``api_runtime_config`` tunables."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field, field_validator

from api.repositories import runtime_config as runtime_repo
from api.settings import get_settings

_CACHE_TTL_SEC = 1.5
_cache_mono: float = 0.0
_cache_value: Optional["EffectiveTunables"] = None


class RuntimeOverlayPatch(BaseModel):
    """Partial update stored in ``api_runtime_config.payload`` (subset only)."""

    worker_poll_interval_seconds: float | None = Field(default=None, gt=0, le=3600.0)
    max_target_history_points: int | None = Field(default=None, ge=1, le=50_000)
    max_group_exposure_history_points: int | None = Field(default=None, ge=1, le=50_000)
    max_exposure_history_points: int | None = Field(default=None, ge=1, le=50_000)
    max_trade_events: int | None = Field(default=None, ge=1, le=100_000)
    index_ohlcv_backfill_batch_size: int | None = Field(default=None, ge=1, le=500)
    market_data_cache_ttl_days: int | None = Field(default=None, ge=1, le=3650)
    ollama_timeout_seconds: float | None = Field(default=None, gt=0, le=3600.0)
    ollama_model: str | None = Field(default=None, max_length=128)

    @field_validator("ollama_model", mode="before")
    @classmethod
    def _strip_model(cls, v: object) -> object:
        if v is None or not isinstance(v, str):
            return v
        s = v.strip()
        return s or None


@dataclass(frozen=True, slots=True)
class EffectiveTunables:
    worker_poll_interval_seconds: float
    max_target_history_points: int
    max_group_exposure_history_points: int
    max_exposure_history_points: int
    max_trade_events: int
    index_ohlcv_backfill_batch_size: int
    market_data_cache_ttl_days: int
    ollama_timeout_seconds: float
    ollama_model: str

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "worker_poll_interval_seconds": self.worker_poll_interval_seconds,
            "max_target_history_points": self.max_target_history_points,
            "max_group_exposure_history_points": self.max_group_exposure_history_points,
            "max_exposure_history_points": self.max_exposure_history_points,
            "max_trade_events": self.max_trade_events,
            "index_ohlcv_backfill_batch_size": self.index_ohlcv_backfill_batch_size,
            "market_data_cache_ttl_days": self.market_data_cache_ttl_days,
            "ollama_timeout_seconds": self.ollama_timeout_seconds,
            "ollama_model": self.ollama_model,
        }


_TUNABLE_KEYS = frozenset(RuntimeOverlayPatch.model_fields.keys())
TUNABLE_KEYS = _TUNABLE_KEYS


def clear_tunables_cache() -> None:
    global _cache_mono, _cache_value
    _cache_mono = 0.0
    _cache_value = None


def _overlay_value(overlay: dict[str, Any], key: str) -> Any | None:
    if key not in overlay:
        return None
    v = overlay[key]
    return v if v is not None else None


def get_effective_tunables(*, force_refresh: bool = False) -> EffectiveTunables:
    """Short-TTL cached merge of DB overlay and :func:`get_settings`."""
    global _cache_mono, _cache_value
    now = time.monotonic()
    if not force_refresh and _cache_value is not None and (now - _cache_mono) < _CACHE_TTL_SEC:
        return _cache_value

    s = get_settings()
    overlay = runtime_repo.fetch_runtime_payload() or {}
    o = overlay

    def fnum(key: str, default: float) -> float:
        raw = _overlay_value(o, key)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        return float(default)

    def fint(key: str, default: int) -> int:
        raw = _overlay_value(o, key)
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
        return int(default)

    om_raw = _overlay_value(o, "ollama_model")
    if om_raw is not None and str(om_raw).strip():
        ollama_model = str(om_raw).strip()
    else:
        ollama_model = str(s.ollama_model or "llama3.2").strip()

    eff = EffectiveTunables(
        worker_poll_interval_seconds=fnum("worker_poll_interval_seconds", s.worker_poll_interval_seconds),
        max_target_history_points=fint("max_target_history_points", s.max_target_history_points),
        max_group_exposure_history_points=fint(
            "max_group_exposure_history_points", s.max_group_exposure_history_points
        ),
        max_exposure_history_points=fint("max_exposure_history_points", s.max_exposure_history_points),
        max_trade_events=fint("max_trade_events", s.max_trade_events),
        index_ohlcv_backfill_batch_size=fint("index_ohlcv_backfill_batch_size", s.index_ohlcv_backfill_batch_size),
        market_data_cache_ttl_days=fint("market_data_cache_ttl_days", s.market_data_cache_ttl_days),
        ollama_timeout_seconds=fnum("ollama_timeout_seconds", s.ollama_timeout_seconds),
        ollama_model=ollama_model,
    )
    _cache_mono = now
    _cache_value = eff
    return eff


def tunable_sources() -> dict[str, str]:
    """Per-field ``database`` vs ``environment`` for UI (based on overlay presence)."""
    overlay = runtime_repo.fetch_runtime_payload() or {}
    out: dict[str, str] = {}
    for k in sorted(_TUNABLE_KEYS):
        out[k] = "database" if _overlay_value(overlay, k) is not None else "environment"
    return out


def merge_overlay_patch(existing: Mapping[str, Any], patch: RuntimeOverlayPatch) -> dict[str, Any]:
    data = dict(existing)
    dumped = patch.model_dump(exclude_none=True)
    for k, v in dumped.items():
        data[k] = v
    return data
