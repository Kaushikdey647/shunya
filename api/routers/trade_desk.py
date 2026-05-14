"""Paper trade desk HTTP surface (secured; requires lifespan Alpaca runtime)."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Annotated, Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import TypeAdapter

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import ActivityType, DTBPCheck, PDTCheck, TradeConfirmationEmail
from alpaca.trading.models import AccountConfiguration, PortfolioHistory
from alpaca.trading.requests import GetPortfolioHistoryRequest

from api.schemas.models import (
    AlpacaAccountActivitiesResponse,
    AlpacaAccountActivityOut,
    AlpacaAccountConfigurationsOut,
    AlpacaAccountConfigurationsPatch,
    AlpacaEquityAccountOut,
    AlpacaPortfolioHistoryOut,
    PaperCycleRequest,
    PaperCycleResponse,
)
from api.settings import get_settings
from api.trade_desk_runtime import TradeDeskRuntime
from shunya.live.demo import build_demo_target_blend_pcs
from shunya.live.desk import InstitutionalPaperDesk, new_correlation_id

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/trade", tags=["trade"])

_EQUITY_ACCOUNT_BLOCKLIST = frozenset(
    {
        "crypto_status",
        "options_buying_power",
        "options_approved_level",
        "options_trading_level",
        "crypto_tier",
        "admin_configurations",
        "user_configurations",
    }
)

_DEFAULT_EQUITY_ACTIVITY_TYPES: tuple[str, ...] = tuple(
    sorted(
        {
            t.value
            for t in ActivityType
        }
        - {
            "FXTRD",
            "OPTRD",
            "OPASN",
            "OPCSH",
            "OPEXC",
            "OPEXP",
        }
    )
)

_ALLOWED_PORTFOLIO_TIMEFRAMES = frozenset({"1Min", "5Min", "15Min", "1H", "1D"})
_ALLOWED_PORTFOLIO_PERIODS = frozenset(
    {
        "1D",
        "3D",
        "1W",
        "2W",
        "1M",
        "3M",
        "6M",
        "1A",
        "2A",
        "all",
    }
)


def _runtime(request: Request) -> TradeDeskRuntime:
    rt = getattr(request.app.state, "trade_desk_runtime", None)
    if rt is None:
        raise HTTPException(
            status_code=503,
            detail="Trade desk is disabled. Set SHUNYA_API_ALPACA_ENABLED=1 and APCA API keys.",
        )
    return rt


def require_trade_desk_token(
    x_token: Annotated[Optional[str], Header(alias="X-Shunya-Trade-Desk-Token")] = None,
) -> None:
    settings = get_settings()
    expected = settings.trade_desk_token
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Trade desk token is not configured. Set SHUNYA_API_TRADE_DESK_TOKEN on the API process.",
        )
    if not x_token or x_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Shunya-Trade-Desk-Token header.")


def _alpaca_http_error(exc: APIError) -> HTTPException:
    _log.warning("Alpaca API error: %s", exc)
    return HTTPException(status_code=502, detail="Broker request failed.")


def _equity_account_payload(raw: dict[str, Any]) -> AlpacaEquityAccountOut:
    filtered = {k: v for k, v in raw.items() if k not in _EQUITY_ACCOUNT_BLOCKLIST}
    return AlpacaEquityAccountOut.model_validate(filtered)


def _iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def _activity_to_out(row: dict[str, Any]) -> AlpacaAccountActivityOut:
    aid = row.get("id")
    at = row.get("activity_type")
    if not isinstance(aid, str) or not isinstance(at, str):
        raise ValueError("activity row missing id or activity_type")
    out: dict[str, Any] = {
        "id": aid,
        "activity_type": at,
        "symbol": row.get("symbol"),
        "qty": row.get("qty"),
        "price": row.get("price"),
        "net_amount": row.get("net_amount"),
        "description": row.get("description"),
    }
    if "transaction_time" in row:
        out["transaction_time"] = _iso(row.get("transaction_time"))
    if "date" in row:
        d = row.get("date")
        out["date"] = d.isoformat() if isinstance(d, date) else _iso(d)
    if "side" in row:
        s = row.get("side")
        out["side"] = s.value if hasattr(s, "value") else (str(s) if s is not None else None)
    if "type" in row:
        t = row.get("type")
        out["trade_activity_type"] = t.value if hasattr(t, "value") else (str(t) if t is not None else None)
    oid = row.get("order_id")
    if oid is not None:
        out["order_id"] = str(oid)
    return AlpacaAccountActivityOut.model_validate(out)


def _parse_activity_types_param(raw: Optional[str]) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(_DEFAULT_EQUITY_ACTIVITY_TYPES)
    allowed = {t.value for t in ActivityType}
    out: list[str] = []
    for part in str(raw).split(","):
        p = part.strip().upper()
        if not p:
            continue
        if p not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown activity_type: {p}")
        out.append(p)
    if not out:
        raise HTTPException(status_code=400, detail="activity_types must list at least one valid type.")
    return out


def _portfolio_history_to_out(ph: PortfolioHistory) -> AlpacaPortfolioHistoryOut:
    d = ph.model_dump(mode="json")
    cf = d.get("cashflow") or {}
    cashflow: dict[str, list[float]] = {}
    if isinstance(cf, dict):
        for k, v in cf.items():
            key = k.value if hasattr(k, "value") else str(k)
            if isinstance(v, list):
                cashflow[str(key)] = [float(x) for x in v]
    return AlpacaPortfolioHistoryOut(
        timestamp=list(d.get("timestamp") or []),
        equity=[float(x) for x in (d.get("equity") or [])],
        profit_loss=[float(x) for x in (d.get("profit_loss") or [])],
        profit_loss_pct=list(d.get("profit_loss_pct") or []),
        base_value=d.get("base_value"),
        timeframe=str(d.get("timeframe") or ""),
        cashflow=cashflow,
    )


def _config_to_out(cfg: AccountConfiguration) -> AlpacaAccountConfigurationsOut:
    d = cfg.model_dump(mode="json")
    return AlpacaAccountConfigurationsOut(
        dtbp_check=str(d["dtbp_check"]),
        fractional_trading=bool(d["fractional_trading"]),
        max_margin_multiplier=str(d["max_margin_multiplier"]),
        no_shorting=bool(d["no_shorting"]),
        pdt_check=str(d["pdt_check"]),
        suspend_trade=bool(d["suspend_trade"]),
        trade_confirm_email=str(d["trade_confirm_email"]),
        ptp_no_exception_entry=bool(d["ptp_no_exception_entry"]),
        max_options_trading_level=d.get("max_options_trading_level"),
    )


@router.post("/paper/cycle", response_model=PaperCycleResponse, dependencies=[Depends(require_trade_desk_token)])
async def paper_cycle(
    body: PaperCycleRequest,
    request: Request,
) -> PaperCycleResponse:
    rt = _runtime(request)
    desk = InstitutionalPaperDesk(
        rt.trading_client,
        rt.data_client,
        rt.settings,
        twap_bins=body.twap_bins,
    )
    cid = body.correlation_id or new_correlation_id("api")
    dt = pd.Timestamp(body.execution_date)
    if body.use_demo_pcs:
        pcs = build_demo_target_blend_pcs()
        res = await desk.run_with_pcs(pcs, capital=body.capital, execution_date=dt, correlation_id=cid)
    else:
        res = await desk.run_with_targets(
            body.targets_usd or {},
            universe=body.universe or [],
            prices=body.prices or {},
            correlation_id=cid,
        )
    out = res.as_dict()
    if body.universe_resolution_note and str(body.universe_resolution_note).strip():
        note = str(body.universe_resolution_note).strip()[:400]
        out["messages"] = [f"universe_resolution_note={note}"] + list(out.get("messages") or [])
    return PaperCycleResponse(**out)


@router.get(
    "/account/equity",
    response_model=AlpacaEquityAccountOut,
    dependencies=[Depends(require_trade_desk_token)],
)
def trade_account_equity(request: Request) -> AlpacaEquityAccountOut:
    rt = _runtime(request)
    try:
        raw = rt.trading_client.get("/account")
        if not isinstance(raw, dict):
            raise HTTPException(status_code=502, detail="Unexpected broker response.")
        return _equity_account_payload(raw)
    except APIError as exc:
        raise _alpaca_http_error(exc) from exc


@router.get(
    "/account/activities",
    response_model=AlpacaAccountActivitiesResponse,
    dependencies=[Depends(require_trade_desk_token)],
)
def trade_account_activities(
    request: Request,
    page_size: int = Query(default=50, ge=1, le=100),
    page_token: Optional[str] = Query(default=None),
    activity_types: Optional[str] = Query(
        default=None,
        description="Comma-separated Alpaca activity types; defaults to equity-oriented types (excludes FX/options).",
    ),
    after: Optional[str] = Query(default=None, description="RFC3339 or YYYY-MM-DD"),
    until: Optional[str] = Query(default=None, description="RFC3339 or YYYY-MM-DD"),
) -> AlpacaAccountActivitiesResponse:
    rt = _runtime(request)
    types_list = _parse_activity_types_param(activity_types)
    params: dict[str, Any] = {
        "page_size": page_size,
        "activity_types": ",".join(types_list),
    }
    if page_token:
        params["page_token"] = page_token
    if after:
        params["after"] = after
    if until:
        params["until"] = until
    try:
        raw = rt.trading_client.get("/account/activities", params)
    except APIError as exc:
        raise _alpaca_http_error(exc) from exc

    if not isinstance(raw, list):
        raise HTTPException(status_code=502, detail="Unexpected broker response.")

    activities: list[AlpacaAccountActivityOut] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            activities.append(_activity_to_out(item))
        except ValueError:
            continue

    next_token: Optional[str] = None
    if len(activities) == page_size:
        last = raw[-1] if raw else None
        if isinstance(last, dict) and isinstance(last.get("id"), str):
            next_token = last["id"]

    return AlpacaAccountActivitiesResponse(activities=activities, next_page_token=next_token)


@router.get(
    "/account/portfolio-history",
    response_model=AlpacaPortfolioHistoryOut,
    dependencies=[Depends(require_trade_desk_token)],
)
def trade_account_portfolio_history(
    request: Request,
    period: str = Query(default="1M"),
    timeframe: Optional[str] = Query(default=None),
    date_end: Optional[date] = Query(default=None),
    extended_hours: Optional[bool] = Query(default=None),
    intraday_reporting: Optional[str] = Query(default=None),
    pnl_reset: Optional[str] = Query(default=None),
) -> AlpacaPortfolioHistoryOut:
    if period not in _ALLOWED_PORTFOLIO_PERIODS:
        raise HTTPException(status_code=400, detail="Invalid period.")
    if timeframe is not None and timeframe not in _ALLOWED_PORTFOLIO_TIMEFRAMES:
        raise HTTPException(status_code=400, detail="Invalid timeframe.")
    if intraday_reporting is not None and len(intraday_reporting) > 32:
        raise HTTPException(status_code=400, detail="Invalid intraday_reporting.")
    if pnl_reset is not None and len(pnl_reset) > 32:
        raise HTTPException(status_code=400, detail="Invalid pnl_reset.")

    rt = _runtime(request)
    req = GetPortfolioHistoryRequest(
        period=period,
        timeframe=timeframe,
        date_end=date_end,
        extended_hours=extended_hours,
        intraday_reporting=intraday_reporting,
        pnl_reset=pnl_reset,
    )
    try:
        ph = rt.trading_client.get_portfolio_history(req)
    except APIError as exc:
        raise _alpaca_http_error(exc) from exc
    if not isinstance(ph, PortfolioHistory):
        raise HTTPException(status_code=502, detail="Unexpected broker response.")
    return _portfolio_history_to_out(ph)


@router.get(
    "/account/configurations",
    response_model=AlpacaAccountConfigurationsOut,
    dependencies=[Depends(require_trade_desk_token)],
)
def trade_account_configurations_get(request: Request) -> AlpacaAccountConfigurationsOut:
    rt = _runtime(request)
    try:
        cfg = rt.trading_client.get_account_configurations()
    except APIError as exc:
        raise _alpaca_http_error(exc) from exc
    if not isinstance(cfg, AccountConfiguration):
        raise HTTPException(status_code=502, detail="Unexpected broker response.")
    return _config_to_out(cfg)


@router.patch(
    "/account/configurations",
    response_model=AlpacaAccountConfigurationsOut,
    dependencies=[Depends(require_trade_desk_token)],
)
def trade_account_configurations_patch(
    request: Request,
    body: AlpacaAccountConfigurationsPatch,
) -> AlpacaAccountConfigurationsOut:
    rt = _runtime(request)
    try:
        current = rt.trading_client.get_account_configurations()
    except APIError as exc:
        raise _alpaca_http_error(exc) from exc
    if not isinstance(current, AccountConfiguration):
        raise HTTPException(status_code=502, detail="Unexpected broker response.")

    patch_data = body.model_dump(exclude_unset=True)
    if not patch_data:
        return _config_to_out(current)

    data = current.model_dump(mode="python")
    if "dtbp_check" in patch_data:
        data["dtbp_check"] = TypeAdapter(DTBPCheck).validate_python(patch_data["dtbp_check"])
    if "pdt_check" in patch_data:
        data["pdt_check"] = TypeAdapter(PDTCheck).validate_python(patch_data["pdt_check"])
    if "trade_confirm_email" in patch_data:
        data["trade_confirm_email"] = TypeAdapter(TradeConfirmationEmail).validate_python(
            patch_data["trade_confirm_email"]
        )
    for key in (
        "fractional_trading",
        "max_margin_multiplier",
        "no_shorting",
        "suspend_trade",
        "ptp_no_exception_entry",
        "max_options_trading_level",
    ):
        if key in patch_data:
            data[key] = patch_data[key]

    try:
        updated = AccountConfiguration.model_validate(data)
        out = rt.trading_client.set_account_configurations(updated)
    except APIError as exc:
        raise _alpaca_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid configuration values.") from exc

    if not isinstance(out, AccountConfiguration):
        raise HTTPException(status_code=502, detail="Unexpected broker response.")
    return _config_to_out(out)
