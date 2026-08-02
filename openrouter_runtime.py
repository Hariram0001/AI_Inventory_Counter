"""Shared OpenRouter inference credential + Catalog Test preflight helpers.

Analyze and Catalog Test must obtain the deployment key through one accessor
(``get_openrouter_inference_key`` → ``openrouter_store.get_deployment_key``).
User-facing test/credential state is session-scoped and never stores the key
in the global catalog.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from openrouter import is_openrouter_model
from openrouter_store import (
    get_deployment_key,
    get_deployment_key_status,
    has_verified_deployment_key,
)
from security import redact_text

USER_TEST_STATE_KEY = "openrouter_user_model_test_state"
SESSION_REJECTED_KEY = "openrouter_session_key_rejected"
PAID_CONFIRM_KEY_PREFIX = "catalog_paid_confirm_"

# Offline/test fallback when Streamlit session_state is unavailable.
_FALLBACK_SESSION: dict[str, Any] = {}

CREDENTIAL_FAILURE_MARKERS = (
    "openrouter is not configured",
    "administrator must add",
    "connect openrouter",
    "rejected the current session key",
    "reconnect it in api connections",
    "no verified openrouter",
    "session key",
)

PUBLISH_PREDICTIONS_MESSAGE = (
    "The Roboflow Workflow editor contains detection outputs, but the "
    "published Serverless version does not yet expose predictions. "
    "Save and publish the Workflow, then refresh the catalog."
)

VISUALIZATION_ONLY_MESSAGE = (
    "Unsupported Workflow response: the published Workflow did not return "
    "predictions."
)


@dataclass(frozen=True)
class WorkflowSchemaReport:
    ok: bool
    workflow_id: str = ""
    workspace: str = ""
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    has_image_input: bool = False
    has_classes_input: bool = False
    has_model_api_key_input: bool = False
    has_predictions_output: bool = False
    has_error_status_output: bool = False
    has_label_visualization_output: bool = False
    message: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UserModelTestState:
    model_key: str
    credential_status: str = "unknown"  # missing|verified|rejected|unknown
    test_status: str = "not_tested"
    # not_tested|ready_to_test|successful|successful_zero_detections|failed|blocked
    message: str = ""
    available_for_analyze: bool = False
    last_tested_at: str = ""
    schema_predictions_present: bool | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogTestPreflight:
    ok: bool
    message: str = ""
    reason_code: str = ""
    schema: WorkflowSchemaReport | None = None
    classes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "reason_code": self.reason_code,
            "schema": self.schema.to_public_dict() if self.schema else None,
            "classes": list(self.classes),
            "details": dict(self.details),
        }


def get_openrouter_inference_key() -> str:
    """Single secure accessor used by Analyze and Catalog Test."""
    if _session_key_rejected():
        return ""
    return get_deployment_key()


def openrouter_credential_ready() -> bool:
    if _session_key_rejected():
        return False
    return has_verified_deployment_key()


def openrouter_credential_label() -> str:
    if _session_key_rejected():
        return "Invalid or expired"
    status = get_deployment_key_status()
    if status.configured and status.verified:
        return "Verified for this session"
    if status.configured:
        return "Connect OpenRouter key"
    return "Connect OpenRouter key"


def _session_bucket() -> dict[str, Any]:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is None:
            return _FALLBACK_SESSION
        import streamlit as st

        return st.session_state  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return _FALLBACK_SESSION


def mark_session_key_rejected(*, reason: str = "") -> None:
    """Block further OpenRouter use in this browser session after auth rejection."""
    bucket = _session_bucket()
    bucket[SESSION_REJECTED_KEY] = {
        "rejected": True,
        "at": datetime.now(timezone.utc).isoformat(),
        "reason": redact_text(reason or "", max_len=160),
    }
    # Drop any legacy session plaintext copy — never leave key material around.
    bucket.pop("openrouter_api_key", None)
    status = dict(bucket.get("openrouter_key_status") or {})
    status["verified"] = False
    status["rejected"] = True
    bucket["openrouter_key_status"] = status


def clear_session_key_rejection() -> None:
    bucket = _session_bucket()
    bucket.pop(SESSION_REJECTED_KEY, None)
    status = dict(bucket.get("openrouter_key_status") or {})
    if status:
        status["verified"] = True
        status.pop("rejected", None)
        bucket["openrouter_key_status"] = status


def _session_key_rejected() -> bool:
    flag = _session_bucket().get(SESSION_REJECTED_KEY) or {}
    return bool(isinstance(flag, dict) and flag.get("rejected"))


def _user_state_map() -> dict[str, dict[str, Any]]:
    bucket = _session_bucket()
    raw = bucket.get(USER_TEST_STATE_KEY)
    if not isinstance(raw, dict):
        bucket[USER_TEST_STATE_KEY] = {}
        return bucket[USER_TEST_STATE_KEY]
    return raw


def get_user_model_test_state(model_key: str) -> UserModelTestState:
    data = _user_state_map().get(str(model_key)) or {}
    return UserModelTestState(
        model_key=str(model_key),
        credential_status=str(data.get("credential_status") or "unknown"),
        test_status=str(data.get("test_status") or "not_tested"),
        message=str(data.get("message") or ""),
        available_for_analyze=bool(data.get("available_for_analyze")),
        last_tested_at=str(data.get("last_tested_at") or ""),
        schema_predictions_present=data.get("schema_predictions_present"),
    )


def set_user_model_test_state(state: UserModelTestState) -> None:
    mapping = _user_state_map()
    mapping[state.model_key] = state.to_public_dict()
    _session_bucket()[USER_TEST_STATE_KEY] = mapping


def clear_stale_credential_test_state(model_key: str | None = None) -> None:
    """After re-verifying a key: Ready to test — never auto Live validated."""
    clear_session_key_rejection()
    mapping = _user_state_map()
    keys = [str(model_key)] if model_key else list(mapping.keys())
    if not keys and model_key is None:
        # Also seed common OpenRouter catalog key when map empty.
        from config import OPENROUTER_WORKFLOW_ID

        keys = [f"workflow:hariram-s-mzhvc/{OPENROUTER_WORKFLOW_ID}"]
    for key in keys:
        prev = mapping.get(key) or {}
        mapping[key] = UserModelTestState(
            model_key=key,
            credential_status="verified" if openrouter_credential_ready() else "missing",
            test_status="ready_to_test",
            message="Ready to test",
            available_for_analyze=False,
            last_tested_at="",
            schema_predictions_present=prev.get("schema_predictions_present"),
        ).to_public_dict()
    _session_bucket()[USER_TEST_STATE_KEY] = mapping
    _clear_global_stale_credential_failure(model_key)


def _clear_global_stale_credential_failure(model_key: str | None) -> None:
    """Reset credential-related Failed stamps; do not mark live validated."""
    try:
        from model_catalog import (
            STATUS_READY,
            load_catalog_entries,
            save_catalog_entries,
        )
    except Exception:  # noqa: BLE001
        return

    entries = load_catalog_entries()
    changed = False
    for entry in entries:
        if model_key and entry.key != model_key:
            continue
        if not (
            is_openrouter_model(entry)
            or bool(getattr(entry, "requires_user_api_key", False))
        ):
            continue
        msg = str(entry.validation_message or entry.last_test_status or "").lower()
        credentialish = entry.last_test_status in {"Failed", "failed"} and (
            any(marker in msg for marker in CREDENTIAL_FAILURE_MARKERS) or not entry.validated
        )
        # Stale Failed before a verified key also qualifies when message empty/old.
        if entry.last_test_status in {"Failed", "failed"} and (
            credentialish or any(marker in msg for marker in CREDENTIAL_FAILURE_MARKERS)
            or "not configured" in msg
            or not (entry.validation_message or "").strip()
        ):
            entry.last_test_status = "Ready to test"
            entry.validated = False
            entry.status = STATUS_READY
            entry.validation_status = "ready_to_test"
            entry.validation_message = (
                "OpenRouter key is verified. Run Catalog Test to live-validate."
            )
            entry.normalize_schema_fields()
            changed = True
    if changed:
        save_catalog_entries(entries)


def is_credential_failure_message(message: str | None) -> bool:
    text = str(message or "").lower()
    return any(marker in text for marker in CREDENTIAL_FAILURE_MARKERS)


def inspect_published_workflow_schema(
    workspace_name: str,
    workflow_id: str,
    *,
    api_key: str | None = None,
) -> WorkflowSchemaReport:
    """Fetch published Workflow spec (no inference) and check OD outputs."""
    workspace = str(workspace_name or "").strip()
    wf = str(workflow_id or "").strip()
    if not workspace or not wf:
        return WorkflowSchemaReport(
            ok=False,
            workflow_id=wf,
            workspace=workspace,
            message="Workflow workspace/id is incomplete.",
        )
    try:
        from detector import RoboflowDetector

        detector = RoboflowDetector(api_key=api_key) if api_key is not None else RoboflowDetector()
        spec = detector._fetch_published_workflow_specification(workspace, wf)
    except Exception as exc:  # noqa: BLE001
        return WorkflowSchemaReport(
            ok=False,
            workflow_id=wf,
            workspace=workspace,
            message=redact_text(
                f"Could not load published Workflow specification: {type(exc).__name__}",
                max_len=200,
            ),
        )
    if not isinstance(spec, dict):
        return WorkflowSchemaReport(
            ok=False,
            workflow_id=wf,
            workspace=workspace,
            message="Published Workflow specification is unavailable.",
        )

    inputs = [i for i in (spec.get("inputs") or []) if isinstance(i, dict)]
    outputs = [o for o in (spec.get("outputs") or []) if isinstance(o, dict)]
    input_names = tuple(str(i.get("name") or "") for i in inputs)
    output_names = tuple(str(o.get("name") or "") for o in outputs)
    has_image = "image" in input_names
    has_classes = "classes" in input_names
    has_key = "model_api_key" in input_names
    has_preds = "predictions" in output_names
    has_err = "error_status" in output_names
    has_viz = "label_visualization" in output_names

    if not (has_image and has_classes and has_key):
        return WorkflowSchemaReport(
            ok=False,
            workflow_id=wf,
            workspace=workspace,
            input_names=input_names,
            output_names=output_names,
            has_image_input=has_image,
            has_classes_input=has_classes,
            has_model_api_key_input=has_key,
            has_predictions_output=has_preds,
            has_error_status_output=has_err,
            has_label_visualization_output=has_viz,
            message=(
                "Published Workflow is missing required inputs "
                "(image, classes, model_api_key)."
            ),
        )
    if not has_preds:
        return WorkflowSchemaReport(
            ok=False,
            workflow_id=wf,
            workspace=workspace,
            input_names=input_names,
            output_names=output_names,
            has_image_input=has_image,
            has_classes_input=has_classes,
            has_model_api_key_input=has_key,
            has_predictions_output=False,
            has_error_status_output=has_err,
            has_label_visualization_output=has_viz,
            message=PUBLISH_PREDICTIONS_MESSAGE,
        )
    return WorkflowSchemaReport(
        ok=True,
        workflow_id=wf,
        workspace=workspace,
        input_names=input_names,
        output_names=output_names,
        has_image_input=True,
        has_classes_input=True,
        has_model_api_key_input=True,
        has_predictions_output=True,
        has_error_status_output=has_err,
        has_label_visualization_output=has_viz,
        message="Published Workflow schema supports object detection.",
    )


def resolve_catalog_test_classes(
    model: Any, inventory_key: str = "Boxes"
) -> list[str]:
    import config
    from detector import prompt_to_class_names

    prompt = config.inventory_detection_prompt(inventory_key)
    classes = prompt_to_class_names(prompt)
    if classes:
        return classes
    # Explicit Boxes fallback used by the planned live probe.
    return ["cardboard box", "shipping box", "package box"]


def preflight_openrouter_catalog_test(
    model: Any,
    user: Any,
    *,
    has_test_image: bool,
    paid_confirmed: bool,
    inventory_key: str = "Boxes",
    fetch_schema: bool = True,
    db_path: str | None = None,
) -> CatalogTestPreflight:
    """Validate everything that must pass before a paid Catalog Test."""
    from model_access import evaluate_model_access, resolve_model_key
    from model_adapters import resolve_adapter_type

    model_key = resolve_model_key(model)
    if user is None:
        return CatalogTestPreflight(
            ok=False,
            message="Sign in to run Catalog Test.",
            reason_code="not_authenticated",
        )
    if not getattr(user, "is_active", True):
        return CatalogTestPreflight(
            ok=False,
            message="Your account is deactivated.",
            reason_code="inactive",
        )

    adapter_type = resolve_adapter_type(model)
    if adapter_type != "openrouter_vlm_detector" and not is_openrouter_model(model):
        return CatalogTestPreflight(
            ok=False,
            message="This Catalog Test path is only for the OpenRouter VLM adapter.",
            reason_code="wrong_adapter",
            details={"adapter_type": adapter_type},
        )

    if not openrouter_credential_ready():
        return CatalogTestPreflight(
            ok=False,
            message=(
                "OpenRouter rejected the current session key. Reconnect it in "
                "API Connections."
                if _session_key_rejected()
                else "Connect and verify an OpenRouter key in API Connections before testing."
            ),
            reason_code="missing_key",
        )

    decision = evaluate_model_access(
        model,
        user,
        inventory_key=inventory_key,
        has_verified_key=True,
        db_path=db_path,
    )
    if not decision.allowed:
        return CatalogTestPreflight(
            ok=False,
            message=decision.reason or "Model access blocked.",
            reason_code=decision.action or "access_blocked",
            details={
                "quota_used": decision.quota_used,
                "quota_limit": decision.quota_limit,
            },
        )

    if not has_test_image:
        return CatalogTestPreflight(
            ok=False,
            message="Upload a probe image or add data/ai_config_test_image.jpg.",
            reason_code="missing_image",
        )

    classes = resolve_catalog_test_classes(model, inventory_key=inventory_key)
    if not classes:
        return CatalogTestPreflight(
            ok=False,
            message="Effective detection classes are empty.",
            reason_code="empty_classes",
        )

    schema = None
    # Direct OpenRouter chat/completions path does not depend on Roboflow
    # Workflow published outputs. Keep schema inspection informational only.
    if fetch_schema:
        try:
            schema = inspect_published_workflow_schema(
                getattr(model, "workspace_name", "") or "",
                getattr(model, "workflow_id", "") or "",
            )
        except Exception:  # noqa: BLE001
            schema = None

    if not paid_confirmed:
        return CatalogTestPreflight(
            ok=False,
            message="Confirm that this Catalog Test may use paid OpenRouter inference.",
            reason_code="confirmation_required",
            schema=schema,
            classes=classes,
        )

    if decision.quota_limit is not None and (decision.quota_used or 0) >= decision.quota_limit:
        return CatalogTestPreflight(
            ok=False,
            message="You have reached today's run limit for this model.",
            reason_code="wait_quota",
            schema=schema,
            classes=classes,
            details={
                "quota_used": decision.quota_used,
                "quota_limit": decision.quota_limit,
            },
        )

    return CatalogTestPreflight(
        ok=True,
        message="Preflight OK",
        reason_code="ok",
        schema=schema,
        classes=classes,
        details={"model_key": model_key, "credential": openrouter_credential_label()},
    )


def redacted_workflow_parameters(
    *,
    image_name: str,
    classes: list[str],
    key_param: str = "model_api_key",
) -> dict[str, Any]:
    return {
        "image": f"<{image_name or 'test-image'}>",
        "classes": list(classes),
        key_param: "[REDACTED]",
    }


def payload_has_predictions(payload: Any) -> bool:
    if isinstance(payload, dict):
        if "predictions" in payload:
            return True
        return any(payload_has_predictions(v) for v in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(payload_has_predictions(item) for item in payload)
    return False


def payload_has_label_visualization(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() == "label_visualization" and value not in (None, "", [], {}):
                return True
            if payload_has_label_visualization(value):
                return True
        return False
    if isinstance(payload, (list, tuple)):
        return any(payload_has_label_visualization(item) for item in payload)
    return False


def is_auth_rejection_error(message: str | None) -> bool:
    text = str(message or "").lower()
    return any(
        token in text
        for token in (
            "401",
            "unauthorized",
            "invalid api key",
            "rejected your api key",
            "rejected the current session key",
            "no auth",
        )
    )


__all__ = [
    "CREDENTIAL_FAILURE_MARKERS",
    "CatalogTestPreflight",
    "PUBLISH_PREDICTIONS_MESSAGE",
    "USER_TEST_STATE_KEY",
    "UserModelTestState",
    "VISUALIZATION_ONLY_MESSAGE",
    "WorkflowSchemaReport",
    "clear_session_key_rejection",
    "clear_stale_credential_test_state",
    "get_openrouter_inference_key",
    "get_user_model_test_state",
    "inspect_published_workflow_schema",
    "is_auth_rejection_error",
    "is_credential_failure_message",
    "mark_session_key_rejected",
    "openrouter_credential_label",
    "openrouter_credential_ready",
    "payload_has_label_visualization",
    "payload_has_predictions",
    "preflight_openrouter_catalog_test",
    "redacted_workflow_parameters",
    "resolve_catalog_test_classes",
    "set_user_model_test_state",
]
