"""Helpers for multi-model comparison status, summaries, and selection rules."""

from __future__ import annotations

from typing import Any

from model_adapters import ModelInferenceResult
from schemas import InferenceResult, ModelConfig

COMPARE_MIN_MODELS = 2
COMPARE_MAX_MODELS = 3

# User-facing status labels (truthful; failures are never shown as Count: 0).
STATUS_SUCCESS_DETECTIONS = "Success with detections"
STATUS_SUCCESS_ZERO = "Success with zero detections"
STATUS_CONFIG = "Configuration failure"
STATUS_AUTH = "Authentication failure"
STATUS_UNAVAILABLE = "Workflow/model unavailable"
STATUS_TIMEOUT = "Timeout"
STATUS_NETWORK = "Network failure"
STATUS_PARSER = "Parser failure"
STATUS_ANNOTATION = "Annotation failure"
STATUS_FAILED = "Failed"


def is_compare_peer(model: ModelConfig) -> bool:
    """True if the model may participate in Compare Models.

    Includes enabled Roboflow workflow/model peers and confirmed local inference
    adapters. Excludes demo/mock fixtures.
    """
    if getattr(model, "demo_only", False) or model.is_demo_model_id():
        return False
    kind = (model.kind or "").lower()
    return kind in {"workflow", "model", "local"}


def compare_peer_models(selectable: list[ModelConfig]) -> list[ModelConfig]:
    return [m for m in selectable if is_compare_peer(m)]


def validate_compare_selection(selected_names: list[str], compare_names: list[str]) -> list[str]:
    """Return validation errors for a Compare Models selection."""
    errors: list[str] = []
    valid = [n for n in selected_names if n in compare_names]
    if len(valid) < COMPARE_MIN_MODELS:
        errors.append(
            f"Compare Models requires at least {COMPARE_MIN_MODELS} models."
        )
    if len(valid) > COMPARE_MAX_MODELS:
        errors.append(
            f"Compare Models allows at most {COMPARE_MAX_MODELS} models."
        )
    return errors


def sanitize_compare_selection(
    selected_names: list[str], compare_names: list[str]
) -> list[str]:
    """Keep only names still in the compare pool; preserve order; cap at max."""
    allowed = set(compare_names)
    out: list[str] = []
    for name in selected_names:
        if name in allowed and name not in out:
            out.append(name)
        if len(out) >= COMPARE_MAX_MODELS:
            break
    return out


def human_status(
    *,
    success: bool,
    final_count: int | None,
    error_type: str | None,
    request_completed: bool = True,
) -> str:
    """Map adapter/result fields to a truthful comparison status label."""
    if success and request_completed:
        if (final_count or 0) == 0:
            return STATUS_SUCCESS_ZERO
        return STATUS_SUCCESS_DETECTIONS

    et = (error_type or "").lower().replace(" ", "_")
    if et in {"auth", "authentication", "unauthorized", "forbidden", "api_key"}:
        return STATUS_AUTH
    if et in {"config", "configuration", "validation", "invalid_config"}:
        return STATUS_CONFIG
    if et in {
        "unavailable",
        "not_found",
        "workflow_unavailable",
        "model_unavailable",
        "empty_workflow_output",
    }:
        return STATUS_UNAVAILABLE
    if et in {"timeout", "timed_out", "deadline_exceeded"}:
        return STATUS_TIMEOUT
    if et in {"network", "connection", "connect", "http_error", "api_error"}:
        # api_error often means transport/HTTP; prefer network unless auth-like
        if "auth" in et or "401" in et or "403" in et:
            return STATUS_AUTH
        if et == "api_error":
            return STATUS_NETWORK
        return STATUS_NETWORK
    if et in {"parser", "parse", "parse_error", "schema"}:
        return STATUS_PARSER
    if et in {"annotation", "annotate", "annotation_error"}:
        return STATUS_ANNOTATION
    if "timeout" in et:
        return STATUS_TIMEOUT
    if "auth" in et or "unauthorized" in et:
        return STATUS_AUTH
    if "parse" in et:
        return STATUS_PARSER
    if "annotat" in et:
        return STATUS_ANNOTATION
    if "network" in et or "connect" in et:
        return STATUS_NETWORK
    if "config" in et:
        return STATUS_CONFIG
    return STATUS_FAILED


def summary_row_from_mir(
    mir: ModelInferenceResult,
    *,
    image_name: str,
    cached: bool = False,
) -> dict[str, Any]:
    """Build a comparison summary row. Failures never expose fake zero counts."""
    status = human_status(
        success=bool(mir.success),
        final_count=mir.final_count if mir.success else None,
        error_type=mir.error_type,
        request_completed=bool(mir.success),
    )
    if mir.success:
        raw = mir.raw_count
        final = mir.final_count
        avg = round(mir.avg_confidence, 4)
        mx = round(mir.max_confidence, 4)
        classes = ", ".join(mir.classes) or "(none)"
        warn_n = len(mir.warnings or [])
        warnings = "; ".join((mir.warnings or [])[:3])
    else:
        raw = None
        final = None
        avg = None
        mx = None
        classes = "—"
        warn_n = 0
        warnings = ""

    prompt_based = bool(mir.effective_prompt) and mir.model_source in {
        "foundation",
        "Roboflow",
        "roboflow",
    }
    # Prefer explicit prompt list for open-vocab; empty + returned classes ≈ fixed-class
    class_mode = "prompt-based" if mir.effective_prompt else "fixed-class"
    return {
        "model": mir.model_display_name,
        "model_key": mir.model_key,
        "status": status,
        "raw_count": raw,
        "final_count": final,
        "avg_confidence": avg,
        "max_confidence": mx,
        "classes": classes,
        "processing_time": round(mir.processing_time_seconds, 3),
        "warning_count": warn_n,
        "warnings": warnings,
        "source": mir.response_source,
        "error": mir.error_message or "",
        "error_type": mir.error_type or "",
        "image": image_name,
        "cached": cached,
        "success": bool(mir.success),
        "class_mode": class_mode,
        "prompt_based": class_mode == "prompt-based",
        "validation_status": "ready" if mir.success else (mir.error_type or "failed"),
    }


def summary_row_from_cached(
    result: InferenceResult,
    *,
    model_key: str = "",
) -> dict[str, Any]:
    status = human_status(
        success=True,
        final_count=result.final_count,
        error_type=None,
        request_completed=True,
    )
    return {
        "model": result.model_name,
        "model_key": model_key,
        "status": status,
        "raw_count": result.raw_prediction_count or result.raw_count,
        "final_count": result.final_count,
        "avg_confidence": round(result.avg_confidence, 4),
        "max_confidence": round(result.max_confidence, 4),
        "classes": ", ".join(sorted({d.class_name for d in result.detections}))
        or "(none)",
        "processing_time": round(result.processing_time_seconds, 3),
        "warning_count": len(result.warnings or []),
        "warnings": "; ".join((result.warnings or [])[:3]),
        "source": result.source or "",
        "error": "",
        "error_type": "",
        "image": result.image_name,
        "cached": True,
        "success": True,
    }


def format_count_display(value: Any) -> str:
    """UI helper: never show 0 for a failed run."""
    if value is None:
        return "—"
    return str(value)


def comparison_run_caption(n_photos: int, n_models: int) -> str:
    runs = max(0, n_photos * n_models)
    return f"{n_photos} photos × {n_models} models = {runs} analysis runs"


def progress_label(model_i: int, n_models: int, img_i: int, n_images: int) -> str:
    return f"Running model {model_i} of {n_models} on image {img_i} of {n_images}"
