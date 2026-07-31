"""Stakeholder-facing POC UX helpers: demo samples, errors, connection status."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sample_images import get_sample_by_id, list_enabled_samples, load_sample_library

# Verified built-in samples only — no fabricated assets.
DEMO_SAMPLE_SPECS: list[dict[str, str]] = [
    {
        "sample_id": "fence_picket_panel_01",
        "card_title": "Fence Panel",
        "inventory_key": "Fence Panel",
        "purpose": (
            "Demonstrates prompt-based counting of an individual fence panel segment."
        ),
        "difficulty": "standard",
    },
    {
        "sample_id": "fence_gate_driveway_01",
        "card_title": "Fence Gate",
        "inventory_key": "Gates",
        "purpose": (
            "Difficult example — structure-level detection or zero detections may occur."
        ),
        "difficulty": "difficult",
    },
]

POC_NOTICE = (
    "This proof of concept estimates visible objects in an image. Results depend on "
    "image quality, object visibility, prompt wording and model capability. Review "
    "detections before saving the final count."
)

POC_LIMITATIONS_DETAILS = [
    "Partially hidden or distant objects may be missed.",
    "Tightly stacked objects may be treated as one object.",
    "Confidence thresholds and prompt wording affect results.",
    "Custom-trained models may improve specialized inventory later.",
    "AI results require human review before operational use.",
]

PROGRESS_PHASES = [
    "Preparing image",
    "Applying inventory prompts",
    "Running selected model",
    "Processing detections",
    "Preparing review",
]

CONN_NOT_TESTED = "Not tested"
CONN_CONNECTED = "Connected"
CONN_AUTH_FAILED = "Authentication failed"
CONN_CONFIG_MISSING = "Configuration missing"
CONN_PROBE_ISSUE = "Connected (probe issue)"

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|token)\s*[:=]\s*\S+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+\S+")
_QUERY_SECRET_RE = re.compile(r"(?i)(api_key=)[^&\s]+")


@dataclass(frozen=True)
class UserFacingError:
    title: str
    message: str
    detail: str = ""


def sanitize_public_text(text: str | None, *, max_len: int = 500) -> str:
    cleaned = _QUERY_SECRET_RE.sub(r"\1***", str(text or ""))
    # Bearer tokens before generic Authorization=… so "Bearer <jwt>" is fully redacted.
    cleaned = _BEARER_RE.sub("Bearer ***", cleaned)
    cleaned = _SECRET_RE.sub(r"\1=***", cleaned)
    cleaned = cleaned.replace("\x00", "").strip()
    # Avoid dumping long Windows paths in user-facing copy
    cleaned = re.sub(r"[A-Za-z]:\\[^\s]{20,}", "[local-path]", cleaned)
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def escape_display(text: str | None) -> str:
    return html.escape(str(text or ""), quote=True)


def classify_user_error(
    *,
    error_type: str | None = None,
    message: str | None = None,
    api_configured: bool = True,
    dynamic_prompt_failed: bool = False,
    success_zero: bool = False,
) -> UserFacingError:
    et = (error_type or "").lower()
    msg = sanitize_public_text(message)

    if success_zero:
        return UserFacingError(
            title="Successful zero detections",
            message=(
                "The model ran successfully but did not find matching objects. "
                "Try adjusting the detection terms or confidence threshold."
            ),
            detail=msg,
        )
    if not api_configured or et in {"missing_api_key", "config_missing"}:
        return UserFacingError(
            title="Missing API configuration",
            message=(
                "Roboflow API configuration is missing. "
                "Open Settings → AI Configuration to review the connection."
            ),
            detail=msg,
        )
    if et in {"auth", "authentication", "unauthorized", "forbidden"} or "auth" in et:
        return UserFacingError(
            title="Authentication failure",
            message="Roboflow authentication failed. Verify the configured API key.",
            detail=msg,
        )
    if dynamic_prompt_failed or et in {
        "injection_failed",
        "workflow_not_dynamic",
        "dynamic_prompt",
    }:
        return UserFacingError(
            title="Dynamic prompt failure",
            message="The selected inventory terms could not be applied to YOLO-World.",
            detail=msg,
        )
    if et in {"timeout", "timed_out"}:
        return UserFacingError(
            title="Inference timeout",
            message="The model request timed out. Try again or use a smaller image.",
            detail=msg,
        )
    return UserFacingError(
        title="Inference failure",
        message=(
            "The model could not process this image. "
            "Try another image or open Technical Details."
        ),
        detail=msg,
    )


def list_demo_sample_cards() -> list[dict[str, Any]]:
    """Build Try-a-Sample cards from verified on-disk samples only."""
    cards: list[dict[str, Any]] = []
    for spec in DEMO_SAMPLE_SPECS:
        sample = get_sample_by_id(spec["sample_id"])
        if sample is None or not sample.decode_ok:
            continue
        cards.append(
            {
                **spec,
                "title": sample.title,
                "description": sample.description or spec["purpose"],
                "filename": sample.filename,
                "path": sample.path,
                "width": sample.width,
                "height": sample.height,
                "mime_type": sample.mime_type,
            }
        )
    return cards


def _auth_looks_successful(auth: str | None, *, auth_ok: bool | None = None) -> bool:
    if auth_ok is True:
        return True
    if auth_ok is False:
        return False
    text = (auth or "").strip().lower()
    if not text:
        return False
    if text in {"successful", "success", "ok", "demo mode", "authenticated"}:
        return True
    if "authenticated" in text or "demo mode" in text:
        return True
    return False


def _auth_looks_failed(auth: str | None, *, auth_ok: bool | None = None) -> bool:
    if auth_ok is False:
        return True
    text = (auth or "").strip().lower()
    if not text:
        return False
    if text in {"failed", "unauthorized", "forbidden", "missing"}:
        return True
    if "fail" in text or "unauthorized" in text or "401" in text or "403" in text:
        return True
    return False


def resolve_connection_label(
    *,
    api_configured: bool,
    last_probe: dict[str, Any] | None,
) -> str:
    """Map probe results to a connection label.

    Authentication success is independent of optional inference-probe success.
    A failed image probe must not be shown as Authentication failed.
    """
    if not api_configured:
        return CONN_CONFIG_MISSING
    if not last_probe:
        return CONN_NOT_TESTED
    auth = str(last_probe.get("auth") or "")
    auth_ok = last_probe.get("auth_ok")
    if isinstance(auth_ok, bool) or auth:
        if _auth_looks_successful(auth, auth_ok=auth_ok if isinstance(auth_ok, bool) else None):
            if last_probe.get("ok"):
                return CONN_CONNECTED
            # Auth worked; optional inference/details had an issue.
            return CONN_CONNECTED
        if _auth_looks_failed(auth, auth_ok=auth_ok if isinstance(auth_ok, bool) else None):
            return CONN_AUTH_FAILED
    if last_probe.get("ok"):
        return CONN_CONNECTED
    return CONN_NOT_TESTED


def connection_status_payload(
    *,
    api_configured: bool,
    workspace: str | None,
    workflow_available: bool,
    validated_model_count: int,
    last_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    label = resolve_connection_label(
        api_configured=api_configured, last_probe=last_probe
    )
    tested_at = None
    if last_probe:
        tested_at = last_probe.get("tested_at") or last_probe.get("finished_at")
    auth_ok = bool(last_probe and _auth_looks_successful(
        str(last_probe.get("auth") or ""),
        auth_ok=last_probe.get("auth_ok") if isinstance(last_probe.get("auth_ok"), bool) else None,
    ))
    return {
        "label": label,
        "workspace": workspace or "—",
        "workflow_available": bool(workflow_available),
        "validated_models": int(validated_model_count),
        "last_successful_test": tested_at if auth_ok or label == CONN_CONNECTED else None,
        "last_test_at": tested_at,
        "api_key_configured": bool(api_configured),
        "auth_ok": auth_ok,
    }


def stamp_connection_probe(result: dict[str, Any]) -> dict[str, Any]:
    """Sanitize and isolate a connection probe result (no wizard mutation)."""
    auth = sanitize_public_text(str(result.get("auth") or ""))
    explicit = result.get("auth_ok")
    if isinstance(explicit, bool):
        auth_ok = explicit
    else:
        auth_ok = _auth_looks_successful(auth)
    safe = {
        "ok": bool(result.get("ok")),
        "auth": auth,
        "auth_ok": auth_ok,
        "message": sanitize_public_text(str(result.get("message") or "")),
        "workflow": sanitize_public_text(str(result.get("workflow") or "")),
        "response_source": sanitize_public_text(str(result.get("response_source") or "")),
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "processing_time": float(result.get("processing_time") or 0.0),
    }
    # Never persist keys or probe image bytes here
    return safe


def progress_phase_label(phase_index: int) -> str:
    if phase_index < 0:
        return PROGRESS_PHASES[0]
    if phase_index >= len(PROGRESS_PHASES):
        return PROGRESS_PHASES[-1]
    return PROGRESS_PHASES[phase_index]


def compare_progress_caption(
    *,
    current_model: str,
    model_index: int,
    total_models: int,
    completed: int,
    failures: int,
    successes: int,
) -> str:
    return (
        f"Model {model_index} of {total_models}: {current_model} · "
        f"Completed {completed}/{total_models} · "
        f"Successful {successes} · Failed {failures}"
    )


def sample_library_has_demo_cards() -> bool:
    return bool(list_demo_sample_cards())


def featured_samples_for_inventory(inventory_key: str) -> list[Any]:
    return list_enabled_samples(inventory_key=inventory_key)
