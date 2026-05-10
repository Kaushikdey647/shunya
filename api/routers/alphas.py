from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from api.alpha_assist import run_alpha_assist, run_alpha_backtest_review
from api.alpha_lint import run_pyright_on_wrapped
from api.alpha_validation import validate_import_ref
from api.repositories import alphas as repo
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

router = APIRouter(prefix="/alphas", tags=["alphas"])


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
def create_alpha(body: AlphaCreate) -> AlphaOut:
    try:
        return repo.insert_alpha(body)
    except RuntimeError as exc:
        if str(exc) == "duplicate_alpha_name":
            raise HTTPException(status_code=409, detail="Alpha name already exists.") from exc
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        raise HTTPException(status_code=404, detail="Alpha not found.")
    return row


@router.patch("/{alpha_id}", response_model=AlphaOut)
def patch_alpha(alpha_id: str, body: AlphaPatch) -> AlphaOut:
    if body.import_ref:
        try:
            validate_import_ref(body.import_ref)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    row = repo.update_alpha(alpha_id, body)
    if row is None:
        raise HTTPException(status_code=404, detail="Alpha not found.")
    return row


@router.delete("/{alpha_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_alpha(alpha_id: str) -> None:
    ok = repo.delete_alpha(alpha_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Alpha not found.")
