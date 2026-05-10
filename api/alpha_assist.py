"""Ollama-backed alpha review (issues with optional corrected bodies) and backtest review.

Environment (see ``api.settings`` / ``get_settings``):
- ``SHUNYA_API_OLLAMA_HOST``: base URL for Ollama (e.g. ``http://127.0.0.1:11434``). When unset,
  ``/alphas/assist-body`` returns empty ``issues``/``markers``; ``/alphas/assist-backtest-review``
  returns a single-item ``summary_points`` list, empty ``risk_points``, a short ``summary_markdown`` stub,
  and ``suggested_body`` null.
- ``SHUNYA_API_OLLAMA_MODEL``: model id for chat completions.

HTTP response shapes (Pydantic in ``api.schemas.models``):
- ``POST /alphas/assist-body`` → ``AlphaAssistBodyResponse``: ``issues`` (``id``, ``severity``,
  ``message``, ``anchor``, ``corrected_body`` optional), ``markers`` (Monaco-compatible, derived).
- ``POST /alphas/assist-backtest-review`` → ``AlphaAssistBacktestReviewResponse``:
  ``summary_points``, ``risk_points`` (unordered-list items), ``summary_markdown`` (legacy / joined),
  optional ``suggested_body``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

import httpx

from api.alpha_assist_context import build_shared_user_context
from api.settings import get_settings
from shunya.algorithm.alpha_source_wrap import wrap_alpha_body

_log = logging.getLogger(__name__)

_SOURCE_MAX_BYTES = 512 * 1024
_METRICS_JSON_MAX = 48_000

_SEVERITY_MAP = {
    "warning": "warning",
    "hint": "information",
    "information": "information",
    "error": "error",
}

_ASSIST_SYSTEM = """You help improve a short **alpha body** (Python fragment: uses ts, cs, fun, ctx).
Goal: fix real bugs, unclear logic, or one concrete improvement. Do not lecture the platform.

Output ONLY valid JSON: a top-level array of issues, OR {"issues": [...]}. No markdown, no text outside JSON.

Each issue object MUST have:
- "id": string
- "message": string — must quote or name something **literally in the body** (identifier, op, or line content).
- "severity": "warning" | "hint" | "information"
- "anchor": {"kind":"line","line": <1-based line in BODY>} OR {"kind":"substring","text": "<exact substring from BODY>"}
- "corrected_body": full replacement BODY text if one fix is obvious and safe; else null.

Rules (follow strictly):
- At most **5** issues; **empty [] is correct** if the body looks coherent.
- Do **not** warn about fundamentals or lookahead unless the substring `fun.` appears in the body.
- Do **not** warn about jnp or vectorization unless there is a **Python for-loop** (or similar iteration) over tickers/symbols in the body.
- `cs` vs `ts`: warn only if usage likely mismatches intent (e.g. rolling per-stock vs cross-section at one bar). Do **not** flag every `cs.rank` or `cs.*` on returns/deltas as wrong.
- **Fundamentals (`fun.*`)**: cross-sectional ops like `cs.rank(fun.SomeField)` are often **correct** (e.g. value/quality vs peers at the same bar). Do **not** suggest swapping to `ts.rank` or "use ts instead of cs" on `fun.*` unless you cite a **specific** bug in the snippet (e.g. wrong window, obvious double-count, or clear misuse named in the code). Vague "ratios may not be appropriate for cs.rank" is forbidden.
- Prefer syntax errors, bad names, risky NaN handling, missing windows, then small refactors."""

_BACKTEST_SYSTEM = """You interpret **backtest numeric metrics** (JSON in the user message) plus the alpha **body**.
Respond with ONLY JSON (no markdown fences, no prose before or after):
{
  "summary_points": string[],
  "risk_points": string[],
  "suggested_body": string | null
}

Rules:
- Each bullet in summary_points and risk_points must tie to a **specific metric name or number** from the provided JSON (say which metric).
- Use 3–8 summary_points and 3–8 risk_points when data supports them; fewer is OK if sparse.
- risk_points: robustness / limitations suggested **by the numbers** (drawdown, turnover, Sharpe, etc.), not generic DSL advice.
- Do **not** repeat generic warnings about fun/jnp/cs/ts unless **suggested_body** fixes something implied by those metrics.
- suggested_body: full replacement alpha body only if metrics clearly motivate a code change; else null."""


def _ollama_chat(host: str, model: str, system: str, user: str, timeout: float) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            return str(msg.get("content", ""))
        if isinstance(msg, str):
            return msg
    return str(data)


def _extract_json_list(text: str) -> list[dict[str, Any]] | None:
    t = text.strip()
    if "```" in t:
        for block in t.split("```"):
            b = block.strip()
            if b.lower().startswith("json"):
                b = b[4:].lstrip()
            if b.startswith("["):
                t = b
                break
    m = re.search(r"\[[\s\S]*\]\s*$", t)
    if m:
        t = m.group(0)
    try:
        val = json.loads(t)
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, list) else None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    t = text.strip()
    if "```" in t:
        for block in t.split("```"):
            b = block.strip()
            if b.lower().startswith("json"):
                b = b[4:].lstrip()
            if b.startswith("{"):
                t = b
                break
    m = re.search(r"\{[\s\S]*\}\s*$", t)
    if m:
        t = m.group(0)
    try:
        val = json.loads(t)
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, dict) else None


def _body_line_bounds(body: str, line_1based: int) -> tuple[int, int, int, int]:
    lines = body.replace("\r\n", "\n").split("\n")
    idx = line_1based - 1
    if idx < 0 or idx >= len(lines):
        return line_1based, 1, line_1based, 1
    line = lines[idx]
    return line_1based, 1, line_1based, max(1, len(line) + 1)


def _substring_bounds(body: str, needle: str) -> tuple[int, int, int, int] | None:
    b = body.replace("\r\n", "\n")
    pos = b.find(needle)
    if pos < 0:
        return None
    before = b[:pos]
    start_line = before.count("\n") + 1
    last_nl = before.rfind("\n")
    start_col = pos - (last_nl + 1) + 1
    end_pos = pos + len(needle)
    before_end = b[:end_pos]
    end_line = before_end.count("\n") + 1
    last_nl_e = before_end.rfind("\n")
    end_col = end_pos - (last_nl_e + 1) + 1
    return start_line, start_col, end_line, end_col


def _stable_issue_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _validate_corrected_body(source_body: str, corrected: object) -> str | None:
    if corrected is None:
        return None
    if not isinstance(corrected, str):
        return None
    t = corrected.strip("\n")
    if not t or t == source_body.strip("\n"):
        return None
    if len(t.encode("utf-8")) > _SOURCE_MAX_BYTES:
        return None
    return t


def _normalize_issue(
    raw: dict[str, Any],
    *,
    source_body: str,
    n_lines: int,
) -> dict[str, Any] | None:
    msg = str(raw.get("message", "")).strip()
    if not msg:
        return None
    sev_raw = str(raw.get("severity", "hint")).lower()
    sev = _SEVERITY_MAP.get(sev_raw, "information")
    iid = str(raw.get("id") or "").strip() or _stable_issue_id(
        {"message": msg, "anchor": raw.get("anchor"), "severity": sev}
    )
    anchor = raw.get("anchor")
    bounds: tuple[int, int, int, int] | None = None
    if isinstance(anchor, dict):
        kind = str(anchor.get("kind", "")).lower()
        if kind == "line":
            try:
                ln = int(anchor.get("line", 1))
            except (TypeError, ValueError):
                ln = 1
            ln = max(1, min(n_lines, ln))
            bounds = _body_line_bounds(source_body, ln)
        elif kind == "substring":
            text = str(anchor.get("text", ""))
            if text:
                bounds = _substring_bounds(source_body, text)
    if bounds is None:
        bounds = (1, 1, 1, 1)
        msg = f"[Review] {msg}"
    sl, sc, el, ec = bounds
    corrected = _validate_corrected_body(source_body, raw.get("corrected_body"))
    return {
        "id": iid,
        "severity": sev,
        "message": msg,
        "startLineNumber": sl,
        "startColumn": sc,
        "endLineNumber": el,
        "endColumn": ec,
        "corrected_body": corrected,
    }


def run_alpha_assist(
    *,
    source_body: str,
    alpha_name: str | None = None,
    alpha_description: str | None = None,
) -> list[dict[str, Any]]:
    """Return assist **issues** (coords + optional corrected_body)."""
    settings = get_settings()
    host = (settings.ollama_host or "").strip().rstrip("/")
    if not host:
        return []
    model = (settings.ollama_model or "llama3.2").strip()
    wrapped = wrap_alpha_body(source_body)
    user = build_shared_user_context(
        alpha_name=alpha_name,
        alpha_description=alpha_description,
        source_body=source_body,
        wrapped_source=wrapped,
    )

    try:
        content = _ollama_chat(
            host,
            model,
            _ASSIST_SYSTEM,
            user,
            float(settings.ollama_timeout_seconds),
        )
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        _log.warning("ollama assist failed: %s", exc)
        return []

    obj = _extract_json_object(content)
    raw_list: list[Any] | None = None
    if isinstance(obj, dict) and isinstance(obj.get("issues"), list):
        raw_list = obj["issues"]
    if raw_list is None:
        raw_list = _extract_json_list(content)
    if raw_list is None:
        _log.warning("ollama assist could not parse JSON")
        return []

    body = source_body.replace("\r\n", "\n")
    n_lines = max(1, body.count("\n") + 1)
    issues: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        row = _normalize_issue(item, source_body=body, n_lines=n_lines)
        if row:
            issues.append(row)
    return issues


def sanitize_metrics_blob(metrics: dict[str, Any], summary: dict[str, Any] | None) -> str:
    """Flatten metrics + summary to a size-capped JSON string (numbers / small structures only)."""
    blob: dict[str, Any] = {"metrics": metrics, "result_summary": summary or {}}

    def _trim(o: Any, depth: int) -> Any:
        if depth <= 0:
            return "…"
        if isinstance(o, dict):
            return {str(k)[:64]: _trim(v, depth - 1) for k, v in list(o.items())[:80]}
        if isinstance(o, (list, tuple)):
            return [_trim(v, depth - 1) for v in o[:30]]
        if isinstance(o, (int, float, bool)) or o is None:
            return o
        if isinstance(o, str):
            s = o.strip()
            return s[:400] + ("…" if len(s) > 400 else "")
        return str(o)[:200]

    trimmed = _trim(blob, 5)
    s = json.dumps(trimmed, default=str, indent=2)
    if len(s.encode("utf-8")) > _METRICS_JSON_MAX:
        s = s.encode("utf-8")[:_METRICS_JSON_MAX].decode("utf-8", errors="ignore") + "\n…truncated"
    return s


def _extract_md_bullets(text: str) -> list[str]:
    out: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("- "):
            b = s[2:].strip()
        elif s.startswith("* "):
            b = s[2:].strip()
        else:
            continue
        if b:
            out.append(b[:2000])
    return out


def _split_markdown_summary_risks(md: str) -> tuple[list[str], list[str]]:
    t = md.replace("\r\n", "\n").strip()
    if not t:
        return [], []
    m = re.search(r"\n#{1,3}\s*risks\b[^\n]*\n", t, flags=re.IGNORECASE)
    if m:
        head = t[: m.start()].strip()
        tail = t[m.end() :].strip()
        return _extract_md_bullets(head), _extract_md_bullets(tail)
    return _extract_md_bullets(t), []


def _normalize_string_list(val: Any, cap: int = 24) -> list[str]:
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for x in val:
        if isinstance(x, (str, int, float, bool)) or x is None:
            s = str(x).strip()
            if s and s not in out:
                out.append(s[:2000])
        if len(out) >= cap:
            break
    return out


def _markdown_from_bullet_lists(summary: list[str], risks: list[str]) -> str:
    parts: list[str] = []
    if summary:
        parts.append("## Summary\n" + "\n".join(f"- {p}" for p in summary))
    if risks:
        parts.append("## Risks\n" + "\n".join(f"- {p}" for p in risks))
    return "\n\n".join(parts) if parts else "_Empty summary._"


def _normalize_backtest_review_payload(
    obj: dict[str, Any] | None,
    *,
    source_body: str,
    raw_text_fallback: str,
) -> dict[str, Any]:
    d = obj or {}
    sug = _validate_corrected_body(source_body, d.get("suggested_body"))

    sp = _normalize_string_list(d.get("summary_points"))
    rp = _normalize_string_list(d.get("risk_points"))
    legacy_md = str(d.get("summary_markdown", "")).strip() if isinstance(d.get("summary_markdown"), str) else ""

    had_model_lists = isinstance(obj, dict) and (
        (isinstance(obj.get("summary_points"), list) and len(obj["summary_points"]) > 0)
        or (isinstance(obj.get("risk_points"), list) and len(obj["risk_points"]) > 0)
    )

    if not sp and not rp and legacy_md:
        sp, rp = _split_markdown_summary_risks(legacy_md)

    if not sp and not rp and legacy_md:
        sp = [legacy_md[:4000]]

    if not sp and not rp and raw_text_fallback.strip():
        sp = [raw_text_fallback.strip()[:4000]]

    if not sp and not rp:
        sp = ["_Empty summary._"]

    if legacy_md and not had_model_lists:
        summary_markdown = legacy_md
    else:
        summary_markdown = _markdown_from_bullet_lists(sp, rp)

    return {
        "summary_points": sp,
        "risk_points": rp,
        "summary_markdown": summary_markdown,
        "suggested_body": sug,
    }


def run_alpha_backtest_review(
    *,
    source_body: str,
    alpha_name: str | None,
    alpha_description: str | None,
    metrics: dict[str, Any],
    result_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return summary_points, risk_points, summary_markdown, suggested_body."""
    settings = get_settings()
    host = (settings.ollama_host or "").strip().rstrip("/")
    if not host:
        msg = "Ollama is not configured (`SHUNYA_API_OLLAMA_HOST`)."
        return {
            "summary_points": [msg],
            "risk_points": [],
            "summary_markdown": f"_{msg}_",
            "suggested_body": None,
        }
    model = (settings.ollama_model or "llama3.2").strip()
    wrapped = wrap_alpha_body(source_body)
    metrics_block = sanitize_metrics_blob(metrics, result_summary)
    user = build_shared_user_context(
        alpha_name=alpha_name,
        alpha_description=alpha_description,
        source_body=source_body,
        wrapped_source=wrapped,
        extra_sections={"Backtest numbers (sanitized JSON)": "```json\n" + metrics_block + "\n```"},
    )

    try:
        content = _ollama_chat(
            host,
            model,
            _BACKTEST_SYSTEM,
            user,
            float(settings.ollama_timeout_seconds),
        )
    except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
        _log.warning("ollama backtest review failed: %s", exc)
        msg = f"Review failed: {exc}"
        return {
            "summary_points": [msg],
            "risk_points": [],
            "summary_markdown": f"_{msg}_",
            "suggested_body": None,
        }

    obj = _extract_json_object(content)
    if not isinstance(obj, dict):
        return _normalize_backtest_review_payload(
            None,
            source_body=source_body,
            raw_text_fallback=str(content)[:8000],
        )
    return _normalize_backtest_review_payload(obj, source_body=source_body, raw_text_fallback="")
