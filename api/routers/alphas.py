from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Query, status

from api.alpha_assist import run_alpha_assist, run_alpha_backtest_review
from api.alpha_lint import run_pyright_on_wrapped
from api.alpha_validation import validate_import_ref
from api.repositories import alphas as repo
from api.repositories import universes as universes_repo
from api.services.notify_background import schedule_notification
from api.schemas.models import (
    AlphaAssistBacktestReviewRequest,
    AlphaAssistBacktestReviewResponse,
    AlphaAssistBodyRequest,
    AlphaAssistBodyResponse,
    AlphaAssistIssue,
    AlphaAssistMarker,
    AlphaCreate,
    AlphaLintBodyRequest,
    AlphaLintBodyResponse,
    AlphaLintDiagnostic,
    AlphaOut,
    AlphaPatch,
)
from shunya.algorithm.alpha_source_wrap import wrap_alpha_body
from shunya.errors import ErrorCode, ShunyaError

router = APIRouter(prefix="/alphas", tags=["alphas"])


def _validate_default_universe_id(uid: str | None) -> None:
    if uid is None or not str(uid).strip():
        return
    s = str(uid).strip()
    try:
        UUID(s)
    except ValueError as exc:
        raise ShunyaError(
            "default_universe_id must be a UUID.",
            code=ErrorCode.VALIDATION_ERROR,
            http_status=400,
        ) from exc
    if not universes_repo.universe_exists(s):
        raise ShunyaError(
            "default_universe_id does not reference an existing universe.",
            code=ErrorCode.UNIVERSE_NOT_FOUND,
            http_status=404,
        )


@router.post("/lint-body", response_model=AlphaLintBodyResponse)
def lint_alpha_body(body: AlphaLintBodyRequest) -> AlphaLintBodyResponse:
    wrapped = wrap_alpha_body(body.source_body)
    raw = run_pyright_on_wrapped(wrapped)
    diagnostics = [AlphaLintDiagnostic(**d) for d in raw]
    return AlphaLintBodyResponse(diagnostics=diagnostics)


@router.post("/assist-body", response_model=AlphaAssistBodyResponse)
def assist_alpha_body(body: AlphaAssistBodyRequest) -> AlphaAssistBodyResponse:
    raw = run_alpha_assist(
        source_body=body.source_body,
        alpha_name=body.alpha_name,
        alpha_description=body.alpha_description,
    )
    issues = [AlphaAssistIssue(**r) for r in raw]
    markers = [
        AlphaAssistMarker(
            severity=i.severity,
            message=i.message,
            startLineNumber=i.startLineNumber,
            startColumn=i.startColumn,
            endLineNumber=i.endLineNumber,
            endColumn=i.endColumn,
        )
        for i in issues
    ]
    return AlphaAssistBodyResponse(issues=issues, markers=markers)


@router.post("/assist-backtest-review", response_model=AlphaAssistBacktestReviewResponse)
def assist_backtest_review(body: AlphaAssistBacktestReviewRequest) -> AlphaAssistBacktestReviewResponse:
    out = run_alpha_backtest_review(
        source_body=body.source_body,
        alpha_name=body.alpha_name,
        alpha_description=body.alpha_description,
        metrics=body.metrics,
        result_summary=body.result_summary,
    )
    return AlphaAssistBacktestReviewResponse(**out)


@router.post("", response_model=AlphaOut, status_code=status.HTTP_201_CREATED)
def create_alpha(body: AlphaCreate, background_tasks: BackgroundTasks) -> AlphaOut:
    _validate_default_universe_id(body.default_universe_id)
    try:
        out = repo.insert_alpha(body)
    except RuntimeError as exc:
        if str(exc) == "duplicate_alpha_name":
            raise ShunyaError(
                "Alpha name already exists.",
                code=ErrorCode.ALPHA_NAME_CONFLICT,
                http_status=409,
            ) from exc
        raise
    except Exception as exc:  # noqa: BLE001
        raise ShunyaError(str(exc), code=ErrorCode.VALIDATION_ERROR, http_status=400) from exc
    schedule_notification(
        background_tasks,
        level="info",
        title="Alpha created",
        message=f'Alpha "{out.name}" created.',
        code="alpha.created",
        context={"alpha_id": out.id},
    )
    return out


@router.get("", response_model=list[AlphaOut])
def list_alphas(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AlphaOut]:
    return repo.list_alphas(limit=limit, offset=offset)


@router.get("/{alpha_id}", response_model=AlphaOut)
def get_alpha(alpha_id: str) -> AlphaOut:
    row = repo.get_alpha(alpha_id)
    if row is None:
        raise ShunyaError("Alpha not found.", code=ErrorCode.ALPHA_NOT_FOUND, http_status=404)
    return row


@router.patch("/{alpha_id}", response_model=AlphaOut)
def patch_alpha(alpha_id: str, body: AlphaPatch, background_tasks: BackgroundTasks) -> AlphaOut:
    if body.import_ref:
        try:
            validate_import_ref(body.import_ref)
        except ValueError as exc:
            raise ShunyaError(str(exc), code=ErrorCode.VALIDATION_ERROR, http_status=400) from exc
    data = body.model_dump(exclude_unset=True)
    if "default_universe_id" in data:
        _validate_default_universe_id(data.get("default_universe_id"))
    row = repo.update_alpha(alpha_id, body)
    if row is None:
        raise ShunyaError("Alpha not found.", code=ErrorCode.ALPHA_NOT_FOUND, http_status=404)
    schedule_notification(
        background_tasks,
        level="info",
        title="Alpha updated",
        message=f'Alpha "{row.name}" updated.',
        code="alpha.updated",
        context={"alpha_id": alpha_id},
    )
    return row


@router.delete("/{alpha_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_alpha(alpha_id: str, background_tasks: BackgroundTasks) -> None:
    row = repo.get_alpha(alpha_id)
    if row is None:
        raise ShunyaError("Alpha not found.", code=ErrorCode.ALPHA_NOT_FOUND, http_status=404)
    ok = repo.delete_alpha(alpha_id)
    if not ok:
        raise ShunyaError("Alpha not found.", code=ErrorCode.ALPHA_NOT_FOUND, http_status=404)
    schedule_notification(
        background_tasks,
        level="info",
        title="Alpha deleted",
        message=f'Alpha "{row.name}" deleted.',
        code="alpha.deleted",
        context={"alpha_id": alpha_id},
    )
