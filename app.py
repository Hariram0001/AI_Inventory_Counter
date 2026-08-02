"""AI Inventory Counter — redesigned Streamlit wizard UI."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Avoid interactive matplotlib backends if supervision pulls it in via inference-sdk.
os.environ.setdefault("MPLBACKEND", "Agg")

# Constants first — zero Streamlit / UI dependencies (safe under Streamlit re-entry)
from app_constants import (
    ADMIN_ONLY_VIEWS,
    PANEL_CAPTIONS,
    PANEL_TITLES,
    PHOTO_REL_INTERNAL_TO_DISPLAY,
    STAGES,
)

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import admin_console
import api_connections_ui
import auth_session
import auth_ui
import config
import model_access
from config import (
    DEFAULT_PROMPTS,
    DEDUP_STRATEGY_EXPLAINER,
    FENCE_PANEL_PROMPT_PRESETS,
    INVENTORY_TYPES,
    TILE_OVERLAPS,
    TILE_SIZES,
    YARD_OPTIONS,
    api_key_configured,
    ensure_data_dir,
    masked_api_key_status,
)
from database import (
    DatabaseError,
    compute_percentage_error,
    compute_reviewed_count,
    get_inventory_history,
    initialize_database,
    insert_inventory_count,
)
from detector import (
    DetectorError,
    RoboflowDetector,
    run_inference_on_prepared_image,
)


def _verify_dynamic_prompt_propagation(*args, **kwargs):
    """Load diagnostic helper fresh — Streamlit can keep a stale detector module."""
    import importlib

    import detector as _detector

    if not hasattr(_detector, "verify_dynamic_prompt_propagation"):
        _detector = importlib.reload(_detector)
    return _detector.verify_dynamic_prompt_propagation(*args, **kwargs)


from detection_viz import (
    assign_marker_numbers,
    color_for_detection,
    css_rgb,
)
from image_processing import (
    ImageProcessingError,
    annotate_image,
    image_to_png_bytes,
    load_image_from_bytes,
    preview_resize,
    validate_upload,
)
from inventory_config import (
    FIXED_PHOTO_RELATIONSHIP,
    PHOTO_RELATIONSHIP_NOTE,
    SELECTABLE_INVENTORY_KEY,
    form_updates_from_recommendation,
    inventory_display_name,
    is_custom_inventory,
    is_inventory_selectable,
    resolve_recommended_model,
)
from inventory_profiles import (
    MAX_PROMPTS,
    AnalysisRunContext,
    build_run_context,
    canonicalize_detection_class,
    counting_unit_for,
    counts_by_item_type,
    effective_prompts_for_inventory,
    enabled_profiles,
    load_inventory_profiles,
    prompts_to_csv,
)
from comparison_helpers import (
    COMPARE_MAX_MODELS,
    COMPARE_MIN_MODELS,
    compare_peer_models,
    comparison_run_caption,
    format_count_display,
    progress_label,
    sanitize_compare_selection,
    summary_row_from_cached,
    summary_row_from_mir,
    validate_compare_selection,
)
from model_adapters import (
    InferenceOptions,
    get_adapter,
    model_key,
    provider_for,
)
from sample_images import (
    get_sample_by_id,
    list_enabled_samples,
    load_sample_library,
    read_sample_bytes,
    sample_library_diagnostics_warnings,
)
from confidence_ui import (
    CONFIDENCE_HELP,
    CONFIDENCE_LABEL,
    CONFIDENCE_LABEL_SHORT,
    confidence_band,
    format_confidence_percent,
    is_low_confidence_warning,
)
from model_registry import (
    get_default_model,
    get_enabled_valid_models,
    get_selectable_analysis_models,
    load_models_from_file,
    merge_session_models,
    normalize_model_name,
    sanitize_selected_model_names,
    save_models_to_file,
    summarize_models,
)
from review_navigation import (
    ITEM_TYPE_ALL,
    PAGE_SIZE,
    available_item_types,
    filter_detections,
    format_detection_option,
    index_of_detection,
    next_detection_id_after_toggle,
    paginate,
    step_detection_id,
)
from overlap import build_consensus_detections
from schemas import ConsensusResult, Detection, InferenceResult, ModelConfig
from ui_helpers import (
    default_form,
    inject_css,
    navigate_to,
    normalize_stage,
    normalize_view,
    open_settings,
    render_empty_state,
    render_nav_buttons,
    render_page_hero,
    render_stage_header,
    render_status_badge,
    render_stepper,
    reset_active_analysis,
)


def _init_session() -> None:
    defaults: dict[str, Any] = {
        "app_view": "welcome",
        "wizard_stage": "setup",
        "settings_section": "ai_configuration",
        "previous_page": None,
        "previous_wizard_step": None,
        "form": default_form(),
        "uploaded_images": [],
        "uploader_nonce": 0,
        "pending_camera": None,
        "annotation_style": "both",
        "review_edits": {
            "excluded_ids": [],
            "manual_detections": [],
            "class_overrides": {},
        },
        "comparison_summaries": [],
        "model_test_results": {},
        "model_trial_rows": [],
        "model_trial_suggestion": {},
        "session_models": [],
        "inference_cache": {},
        "analysis_status": "idle",
        "analysis_results": [],
        "analysis_failures": [],
        "analysis_meta": {},
        "consensus_result": None,
        "accepted_result_key": None,
        "review_state": {
            "use_direct": False,
            "direct_count": None,
            "false_positives": 0,
            "missed_items": 0,
            "notes": "",
        },
        "save_status": "idle",
        "saved_record": None,
        "analyze_running": False,
        "analysis_run_id": None,
        "connection_probe": None,
        "pending_review_payload": None,
        "selected_photo_index": 0,
        "selected_detection_id": None,
        "review_active_image": None,
        "review_active_model": None,
        "open_advanced_settings": False,
        "config_refresh_nonce": 0,
        "ai_config_test_result": None,
        "last_diag_error": None,
        "sample_selected_ids": [],
        "sample_preview_id": None,
        "sample_gallery_page": 0,
        "selected_photos_page": 0,
        "photo_source_mode": "Upload Images",
        "compare_side_by_side": False,
        "run_context": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
        elif key == "form" and isinstance(st.session_state.form, dict):
            for fk, fv in value.items():
                st.session_state.form.setdefault(fk, fv)
        elif key == "review_state" and isinstance(st.session_state.review_state, dict):
            for rk, rv in value.items():
                st.session_state.review_state.setdefault(rk, rv)

    # Migrate legacy stage / view names
    st.session_state.wizard_stage = normalize_stage(
        st.session_state.get("wizard_stage") or "setup"
    )
    view = normalize_view(st.session_state.get("app_view") or "welcome")
    # Legacy Settings shell → dedicated panel views
    raw_view = st.session_state.get("app_view")
    if raw_view == "settings":
        section = st.session_state.get("settings_section") or "ai_configuration"
        view = normalize_view(str(section))
    elif raw_view == "setup":
        view = "diagnostics"
    st.session_state.app_view = view


def _form_get(key: str, default: Any = None) -> Any:
    return st.session_state.form.get(key, default)


def _form_set(**kwargs: Any) -> None:
    st.session_state.form.update(kwargs)


def _resolved_yard() -> str:
    choice = _form_get("yard_choice", "LA Yard")
    if choice == "Other":
        return (_form_get("yard_custom") or "").strip()
    return choice


def _resolved_inventory() -> str:
    choice = _form_get("inventory_choice", "")
    if not choice:
        return ""
    if choice == "Other":
        return (_form_get("inventory_custom") or "").strip()
    return choice


def _custom_item_name() -> str:
    return str(_form_get("custom_item_name") or "").strip()


def _custom_item_alternatives() -> str:
    return str(_form_get("custom_item_alternatives") or "").strip()


def _apply_recommended_setup(
    *,
    inventory_key: str | None = None,
    apply_selection: bool = True,
) -> dict[str, Any]:
    """Resolve and persist Recommended AI Setup for the selected inventory."""
    key = inventory_key if inventory_key is not None else _resolved_inventory()
    _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)
    if not key or not is_inventory_selectable(key):
        clears: dict[str, Any] = {
            "recommended_setup_resolved": False,
            "recommended_model_name": "",
            "recommended_setup_error": "",
        }
        if apply_selection:
            clears["selected_models"] = []
        _form_set(**clears)
        return {"ok": False, "error": "Select an inventory type to continue."}
    resolved = resolve_recommended_model(
        key,
        _all_models(),
        getattr(config, "INVENTORY_MODEL_RECOMMENDATIONS", {}),
        allow_demo=bool(config.DEMO_MODE),
        custom_item_name=_custom_item_name() if is_custom_inventory(key) else None,
        custom_alternatives=_custom_item_alternatives()
        if is_custom_inventory(key)
        else None,
    )
    _form_set(
        **form_updates_from_recommendation(resolved, apply_selection=apply_selection)
    )
    return resolved


def _build_current_run_context(
    *,
    selected_models: list[ModelConfig] | None = None,
    images: list[dict[str, Any]] | None = None,
) -> tuple[AnalysisRunContext | None, list[str]]:
    key = _resolved_inventory()
    models = selected_models or []
    model = models[0] if models else None
    img_ids = [
        str(img.get("id") or img.get("content_hash") or img.get("name") or "")
        for img in (images or st.session_state.get("uploaded_images") or [])
    ]
    return build_run_context(
        inventory_key=key or "",
        custom_item_name=_custom_item_name() if is_custom_inventory(key) else None,
        custom_alternatives=_custom_item_alternatives()
        if is_custom_inventory(key)
        else None,
        selected_model_key=(model_key(model) if model else "") or "",
        selected_model_display_name=(model.name if model else "YOLO-World"),
        confidence_threshold=float(_form_get("confidence_threshold", 0.25)),
        uploaded_image_ids=[i for i in img_ids if i],
        prompt_override=None,
    )


def _all_models() -> list[ModelConfig]:
    return merge_session_models(load_models_from_file(), st.session_state.session_models)


def _enabled_models() -> list[ModelConfig]:
    """Enabled/valid models for Settings and internal use (may include demo when DEMO_MODE)."""
    return get_enabled_valid_models(_all_models())


def _get_selectable_analysis_models():
    """Return current registry selector, reloading if Streamlit cached a stale module."""
    import importlib
    import inspect

    import model_registry as mr

    fn = mr.get_selectable_analysis_models
    if "custom_item" not in inspect.signature(fn).parameters:
        mr = importlib.reload(mr)
        try:
            import model_catalog as mc

            importlib.reload(mc)
        except Exception:  # noqa: BLE001
            pass
        fn = mr.get_selectable_analysis_models
    # Keep app module globals in sync after reload.
    global get_selectable_analysis_models
    get_selectable_analysis_models = fn
    return fn


def _analysis_models() -> list[ModelConfig]:
    """Models the signed-in user may select, after catalog and policy filtering."""
    return _analysis_models_with_blocked()[0]


def _analysis_models_with_blocked():
    """Return (selectable models, [(model, decision)]) for the current user."""
    inv = _resolved_inventory() or SELECTABLE_INVENTORY_KEY
    catalog_models = _get_selectable_analysis_models()(
        _all_models(),
        inv,
        allow_demo=bool(config.DEMO_MODE),
        custom_item=(inv == "Custom Item"),
    )
    from openrouter_runtime import openrouter_credential_ready

    return model_access.partition_models(
        catalog_models,
        auth_session.current_user(),
        inventory_key=inv,
        has_verified_key=openrouter_credential_ready(),
        cost_notice_accepted=True,
    )


def _render_blocked_models_notice(blocked) -> None:
    """Explain why a model the user might expect is not selectable."""
    if not blocked:
        return
    with st.expander(f"{len(blocked)} model(s) unavailable to you", expanded=False):
        for model, decision in blocked:
            st.markdown(f"**{model.name}** — {decision.reason}")
            if decision.quota_limit is not None:
                st.caption(
                    f"Daily limit {decision.quota_used} of {decision.quota_limit} used."
                )
        viewer = auth_session.current_user()
        if viewer and viewer.is_admin:
            actions = {d.action for _, d in blocked}
            if any("OpenRouter" in (d.reason or "") for _, d in blocked) or (
                actions & {"contact_admin"}
            ):
                if st.button(
                    "Configure OpenRouter (admin)",
                    key="analyze_open_api_connections",
                ):
                    open_settings(section="api_keys")


def _record_model_run(model: ModelConfig) -> None:
    """Count a completed run against the user's daily quota for that model."""
    user = auth_session.current_user()
    if user is None:
        return
    try:
        model_access.register_run(user, model)
    except Exception:  # noqa: BLE001 — quota accounting must not break a run
        pass


def _primary_workflow_model() -> ModelConfig | None:
    enabled = _enabled_models()
    if not enabled:
        return None
    selected = _form_get("selected_models") or []
    for name in selected:
        for m in enabled:
            if m.name == name:
                return m
    return enabled[0]


def _dynamic_prompt_verify_models() -> list[ModelConfig]:
    """Workflow models offered in Diagnostics → Dynamic Prompt Verification."""
    try:
        models = list(_all_models())
    except Exception:  # noqa: BLE001 — Diagnostics must work outside wizard state
        models = load_models_from_file()
    out: list[ModelConfig] = []
    seen: set[str] = set()
    for model in models:
        if not model.enabled:
            continue
        if (model.kind or "").lower() != "workflow":
            continue
        if not (model.workflow_id or "").strip():
            continue
        # Local / classical detectors have nothing to inject into.
        if model.is_demo_model_id():
            continue
        key = model.key or model.name
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    # Prefer YOLO-World first in the picker when present.
    out.sort(
        key=lambda m: (
            0 if (m.workflow_id or "").strip().lower() == "custom-workflow" else 1,
            m.name.lower(),
        )
    )
    return out


def _mode_api_name(ui_mode: str) -> str:
    mapping = {
        "Single Model": "Single model",
        "Compare Models": "Compare models",
        "Experimental Consensus": "Experimental consensus",
    }
    return mapping.get(ui_mode, ui_mode)


def _inference_api_name(ui_mode: str) -> str:
    mapping = {
        "Whole Image": "Whole-image inference",
        "Tiled": "Tiled inference",
        "Thorough Multi-Scale": "Thorough multi-scale analysis",
    }
    return mapping.get(ui_mode, ui_mode)


def _error_box(message: str, detail: str | None = None) -> None:
    from poc_ux import sanitize_public_text

    st.error(message)
    if detail:
        with st.expander("Technical Details", expanded=False):
            st.code(sanitize_public_text(detail, max_len=1200))


def _show_user_facing_error(
    *,
    error_type: str | None = None,
    message: str | None = None,
    dynamic_prompt_failed: bool = False,
    success_zero: bool = False,
) -> None:
    from poc_ux import classify_user_error

    err = classify_user_error(
        error_type=error_type,
        message=message,
        api_configured=bool(api_key_configured() or config.DEMO_MODE),
        dynamic_prompt_failed=dynamic_prompt_failed,
        success_zero=success_zero,
    )
    st.error(f"**{err.title}** — {err.message}")
    if err.detail:
        with st.expander("Technical Details", expanded=False):
            st.code(err.detail)


def _make_demo_image() -> bytes:
    img = Image.new("RGB", (800, 600), color=(210, 215, 220))
    draw = ImageDraw.Draw(img)
    panels = [
        (100, 80, 220, 360),
        (221, 88, 339, 362),
        (339, 91, 461, 369),
        (461, 92, 579, 364),
        (580, 94, 700, 370),
        (145, 80, 255, 340),
        (285, 260, 415, 460),
        (408, 250, 532, 460),
        (76, 90, 104, 390),
        (5, 70, 75, 290),
        (695, 130, 785, 370),
    ]
    for i, box in enumerate(panels):
        shade = 70 + (i * 12) % 80
        draw.rectangle(box, outline=(shade, shade + 20, shade), width=3)
        draw.rectangle(
            [box[0] + 8, box[1] + 8, box[2] - 8, box[3] - 8],
            fill=(180 - i * 5, 185, 190),
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _image_meta(
    name: str,
    data: bytes,
    *,
    source: str = "upload",
    mime_type: str = "image/jpeg",
    sample_id: str | None = None,
) -> dict[str, Any]:
    with Image.open(io.BytesIO(data)) as img:
        w, h = img.size
    content_hash = hashlib.sha256(data).hexdigest()
    meta: dict[str, Any] = {
        "id": content_hash[:16],
        "name": name,
        "source": source,
        "mime_type": mime_type,
        "data": data,
        "bytes": data,
        "width": w,
        "height": h,
        "size_bytes": len(data),
        "content_hash": content_hash,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    if sample_id:
        meta["sample_id"] = sample_id
    return meta


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _result_key(result: InferenceResult) -> str:
    return f"{result.model_name}||{result.image_name}"


def _photo_rel_display() -> str:
    internal = _form_get("photo_relationship", "Separate inventory areas")
    return PHOTO_REL_INTERNAL_TO_DISPLAY.get(internal, internal)


def _ensure_selected_models() -> list[str]:
    """Ensure Analysis has a valid selected model; clear stale/deleted names."""
    names = [m.name for m in _analysis_models()]
    compare_names = [m.name for m in compare_peer_models(_analysis_models())]
    raw = [normalize_model_name(n) for n in (_form_get("selected_models") or [])]
    selected = sanitize_selected_model_names(raw, names)
    mode = _form_get("selected_mode", "Single Model")
    if mode == "Compare Models" and len(compare_names) < COMPARE_MIN_MODELS:
        mode = "Single Model"
        _form_set(selected_mode=mode)
    if mode == "Compare Models":
        selected = sanitize_compare_selection(selected, compare_names)
        _form_set(selected_models=selected)
        return selected
    if not selected:
        inv = _resolved_inventory()
        if inv and is_inventory_selectable(inv):
            resolved = _apply_recommended_setup(inventory_key=inv)
            name = normalize_model_name(resolved.get("model_name"))
            if resolved.get("ok") and name in names:
                selected = [name]
        if not selected and names:
            recommended = normalize_model_name(_form_get("recommended_model_name") or "")
            selected = [recommended] if recommended in names else names[:1]
    _form_set(selected_models=selected)
    return selected


def _config_snapshot() -> dict[str, Any]:
    """Build read-only AI configuration summary from local project sources."""
    from poc_ux import connection_status_payload

    config.reload_settings()
    model = _primary_workflow_model()
    api_ok = bool(api_key_configured())
    connected = bool(config.DEMO_MODE or api_ok)
    detection_mode = "Demo" if config.DEMO_MODE else "Live Workflow"
    response_source = "demo source" if config.DEMO_MODE else "live Roboflow"
    probe = st.session_state.get("connection_probe")
    try:
        from model_catalog import get_all_catalog_models, STATUS_READY

        validated_n = sum(
            1
            for e in get_all_catalog_models()
            if e.validated and e.status == STATUS_READY and not e.stale
        )
    except Exception:  # noqa: BLE001
        validated_n = len(_enabled_models())
    conn = connection_status_payload(
        api_configured=api_ok or bool(config.DEMO_MODE),
        workspace=(model.workspace_name if model and model.workspace_name else None),
        workflow_available=bool(model and model.workflow_id),
        validated_model_count=validated_n,
        last_probe=probe if isinstance(probe, dict) else None,
    )
    return {
        "connected": connected,
        "connection_label": conn["label"],
        "provider": "Roboflow",
        "workspace": conn["workspace"],
        "workflow_name": (model.name if model else "—"),
        "workflow_id": (model.workflow_id if model and model.workflow_id else "—"),
        "model_id": (model.model_id if model and model.model_id else "—"),
        "kind": (model.kind if model else "—"),
        "detection_mode": detection_mode,
        "response_source": response_source,
        "api_key": "Configured" if api_ok else "Missing",
        "source_label": "Local project settings (.env / Streamlit secrets + models.json)",
        "models_path": str(config.MODELS_JSON_PATH.name),
        "validated_models": validated_n,
        "workflow_available": conn["workflow_available"],
        "last_successful_test": conn.get("last_successful_test"),
        "last_test_at": conn.get("last_test_at"),
        "connection": conn,
    }


def render_configuration_summary(*, show_actions: bool = True) -> dict[str, Any]:
    from poc_ux import escape_display, render_connection_light_html, stamp_connection_probe
    from roboflow_status import ensure_roboflow_probe
    from roboflow_status import _run_lightweight_auth_probe

    # Auto-test once when this panel opens and nothing fresh is cached.
    ensure_roboflow_probe(force=False)
    snap = _config_snapshot()
    conn = snap.get("connection") or {}
    light_html = render_connection_light_html(
        str(snap.get("connection_label") or "Not tested"),
        auth_ok=conn.get("auth_ok"),
        detail=str(
            (st.session_state.get("connection_probe") or {}).get("message") or ""
        )[:120],
    )
    st.markdown(
        f"""
        <div class="aic-panel aic-panel-g">
          <div class="aic-panel-title">Roboflow connection</div>
          <div style="margin:0.35rem 0 0.65rem 0;">{light_html}</div>
          <div class="aic-chip-grid aic-chip-grid-4">
            <div class="aic-chip aic-chip-b">
              <span class="aic-chip-label">Mode</span>
              <span class="aic-chip-value">{escape_display(snap["detection_mode"])}</span>
            </div>
            <div class="aic-chip aic-chip-g">
              <span class="aic-chip-label">Validated models</span>
              <span class="aic-chip-value">{int(snap.get("validated_models") or 0)}</span>
            </div>
            <div class="aic-chip aic-chip-b">
              <span class="aic-chip-label">Workflow</span>
              <span class="aic-chip-value">{"Available" if snap.get("workflow_available") else "Missing"}</span>
            </div>
            <div class="aic-chip aic-chip-g">
              <span class="aic-chip-label">API key</span>
              <span class="aic-chip-value">{escape_display(snap["api_key"])}</span>
            </div>
          </div>
          <div class="aic-kv-grid" style="margin-top:0.45rem;">
            <div class="aic-kv"><b>Workspace</b><br/>{escape_display(snap["workspace"])}</div>
            <div class="aic-kv"><b>Primary model</b><br/>{escape_display(snap["workflow_name"])}</div>
            <div class="aic-kv"><b>Last check</b><br/>{escape_display(snap.get("last_test_at") or snap.get("last_successful_test") or "—")}</div>
            <div class="aic-kv"><b>Provider</b><br/>Roboflow</div>
          </div>
          <p class="aic-muted" style="margin:0.55rem 0 0 0;">
            Connection is checked automatically when you open this page.
            The API key is never displayed.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_actions:
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("Retest", width="stretch", key="cfg_test_connection"):
                with st.spinner("Retesting Roboflow authentication…"):
                    stamped = stamp_connection_probe(_run_lightweight_auth_probe())
                st.session_state.connection_probe = stamped
                st.session_state.connection_probe_flash = stamped
                st.rerun()
        with a2:
            if st.button("Refresh Configuration", width="stretch", key="cfg_refresh"):
                config.reload_settings()
                st.session_state.config_refresh_nonce = (
                    int(st.session_state.get("config_refresh_nonce", 0)) + 1
                )
                st.rerun()
        with a3:
            if st.button("Advanced Settings", width="stretch", key="cfg_adv_toggle"):
                st.session_state.open_advanced_settings = not bool(
                    st.session_state.get("open_advanced_settings")
                )
                st.rerun()
        probe_flash = st.session_state.pop("connection_probe_flash", None)
        if isinstance(probe_flash, dict):
            if probe_flash.get("auth_ok"):
                st.success(
                    probe_flash.get("message")
                    or "Roboflow authentication succeeded."
                )
            else:
                st.error(
                    probe_flash.get("message")
                    or "Roboflow authentication failed. Verify ROBOFLOW_API_KEY in .env "
                    "or Streamlit secrets."
                )
    return snap


# ---------------------------------------------------------------------------
# Welcome / Settings sections
# ---------------------------------------------------------------------------


def _start_demo_sample(sample_id: str) -> None:
    """Load a verified sample into the wizard without running paid inference."""
    from inventory_profiles import get_profile
    from poc_ux import list_demo_sample_cards
    from sample_images import clear_sample_library_cache, get_sample_by_id

    clear_sample_library_cache()
    cards = {c["sample_id"]: c for c in list_demo_sample_cards()}
    card = cards.get(sample_id)
    sample = get_sample_by_id(sample_id)
    if card is None or sample is None:
        st.warning("That sample is not available.")
        return

    reset_active_analysis(go_home=False, start_wizard=False, rerun=False)
    inv_key = card["inventory_key"] or sample.app_inventory_key
    st.session_state.form = dict(st.session_state.get("form") or default_form())
    st.session_state.form["inventory_choice"] = inv_key
    profile = get_profile(inv_key) or {}
    if profile.get("prompt_terms"):
        from inventory_profiles import prompts_to_csv

        st.session_state.form["prompt"] = prompts_to_csv(list(profile["prompt_terms"]))
        st.session_state.form["effective_prompts"] = list(profile["prompt_terms"])
        st.session_state.form["counting_unit"] = profile.get("counting_unit") or ""
    err = _add_sample_by_id(sample_id)
    if err:
        st.warning(err)
        return
    st.session_state.demo_sample_id = sample_id
    # Photos stage — user explicitly runs analysis (no auto-inference).
    navigate_to("wizard", stage="photos")


def view_welcome(user=None) -> None:
    from poc_ux import (
        POC_LIMITATIONS_DETAILS,
        POC_NOTICE,
        escape_display,
    )

    user = user or auth_session.current_user()

    try:
        initialize_database()
        hist_rows = _visible_history_rows(user)
    except Exception:  # noqa: BLE001
        hist_rows = []

    greeting = f"Welcome back, {escape_display(user.label)}. " if user else ""
    render_page_hero(
        "AI Inventory Counter",
        f"{greeting}Count visible inventory items from photos using AI-powered "
        "object detection.",
    )

    st.info(POC_NOTICE)
    with st.expander("POC limitations", expanded=False):
        for line in POC_LIMITATIONS_DETAILS:
            st.markdown(f"- {line}")

    st.markdown(
        """
        <div class="aic-dash-tile aic-dash-tile-r">
          <h4>Get Started</h4>
          <p>Choose inventory, add photos, run detection, then review and save.
          Use the left panel icons for History, AI Configuration, and more.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Get Started", type="primary", width="stretch", key="get_started"):
        reset_active_analysis(go_home=False, start_wizard=True)

    # Experimental Shape Detection — local OpenCV; never auto-starts.
    try:
        from shape_detection_storage import shape_detection_allowed

        shape_ok, _shape_msg = shape_detection_allowed(user)
    except Exception:  # noqa: BLE001
        shape_ok, _shape_msg = False, ""
    if shape_ok:
        if st.button(
            "Shape Detection",
            width="stretch",
            key="shape_detection_home",
        ):
            navigate_to("shape_detection")
            st.rerun()
        st.caption(
            "Testing Phase · Local computer vision · No API key required"
        )

    st.markdown("#### Capabilities")
    st.markdown(
        """
        - Preset or custom inventory  
        - Upload or camera  
        - AI-generated numbered detections  
        - Manual review and correction  
        - Model comparison where available  
        - Saved inventory history  
        """
    )

    if hist_rows:
        st.markdown("#### Recent saves")
        for row in hist_rows[:3]:
            inv = escape_display(row.get("inventory_type") or "—")
            reviewed = row.get("reviewed_count")
            when = escape_display(row.get("created_at") or "—")
            st.caption(f"{inv} · Reviewed {reviewed} · {when}")
    else:
        st.caption("You have not saved any inventory counts yet.")


def _visible_history_rows(user, *, limit: int = 200) -> list[dict[str, Any]]:
    """History is strictly private: each signed-in user sees only their own rows.

    Unowned / pre-authentication rows are never shared into another account's
    history. Administrators do not get a combined feed here either — every
    account keeps its own log.
    """
    if user is None:
        return []
    return get_inventory_history(
        limit=limit, user_id=user.user_id, include_legacy=False
    )


def _render_history_section() -> None:
    st.caption(
        "Opening history does not rerun inference or change the active wizard. "
        "Photo bytes are not stored with history rows; missing images show as text-only records. "
        "Each account only sees the counts it saved — history is never shared between users."
    )

    viewer = auth_session.current_user()
    if viewer is None:
        return

    try:
        initialize_database()
        rows = _visible_history_rows(viewer)
    except DatabaseError as exc:
        _error_box("Could not load history.", str(exc))
        return

    st.caption(
        f"Private to {viewer.label}: only inventory counts saved while signed "
        "in as this account."
    )

    if not rows:
        render_empty_state(
            "No inventory counts have been saved yet.",
            "Complete a review and choose Save Result to populate Inventory History.",
        )
        return

    df = pd.DataFrame(rows)
    display_cols = [
        c
        for c in [
            "created_at",
            "inventory_type",
            "yard",
            "number_of_photos",
            "ai_count",
            "reviewed_count",
            "accepted_model",
        ]
        if c in df.columns
    ]
    rename = {
        "created_at": "Saved date",
        "inventory_type": "Inventory type",
        "yard": "Location",
        "number_of_photos": "Photos",
        "ai_count": "Detected count",
        "reviewed_count": "Reviewed count",
        "accepted_model": "Model",
    }

    total_photos = int(pd.to_numeric(df.get("number_of_photos"), errors="coerce").fillna(0).sum())
    total_reviewed = int(pd.to_numeric(df.get("reviewed_count"), errors="coerce").fillna(0).sum())
    latest = str(df["created_at"].iloc[0]) if "created_at" in df.columns and len(df) else "—"
    st.markdown(
        f"""
        <div class="aic-chip-grid">
          <div class="aic-chip aic-chip-r">
            <span class="aic-chip-label">Records</span>
            <span class="aic-chip-value">{len(df)}</span>
          </div>
          <div class="aic-chip aic-chip-b">
            <span class="aic-chip-label">Photos saved</span>
            <span class="aic-chip-value">{total_photos}</span>
          </div>
          <div class="aic-chip aic-chip-g">
            <span class="aic-chip-label">Reviewed total</span>
            <span class="aic-chip-value">{total_reviewed}</span>
          </div>
        </div>
        <div class="aic-panel aic-panel-b" style="margin-top:0.35rem;">
          <div class="aic-kv"><b>Latest save</b><br/>{latest}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="aic-panel aic-panel-r"><div class="aic-panel-title">Filters</div></div>',
        unsafe_allow_html=True,
    )
    filter_cols = st.columns(2)
    yards = ["All"] + sorted(
        {str(v) for v in df.get("yard", pd.Series(dtype=str)).dropna().unique()}
    )
    types = ["All"] + sorted(
        {str(v) for v in df.get("inventory_type", pd.Series(dtype=str)).dropna().unique()}
    )
    with filter_cols[0]:
        yard_f = st.selectbox("Location", yards, key="hist_yard")
    with filter_cols[1]:
        type_f = st.selectbox("Inventory type", types, key="hist_type")

    filtered = df
    if yard_f != "All":
        filtered = filtered[filtered["yard"] == yard_f]
    if type_f != "All":
        filtered = filtered[filtered["inventory_type"] == type_f]

    if filtered.empty:
        st.info("No records match the selected filters.")
        return

    shown = filtered[display_cols].rename(columns=rename)
    shown.insert(len(shown.columns), "Status", "Saved")
    if "Model" in shown.columns or "accepted_model" in filtered.columns:
        registry_names = {m.name for m in _all_models()}
        model_col = "Model" if "Model" in shown.columns else None
        if model_col:
            shown[model_col] = shown[model_col].apply(
                lambda n: (
                    n
                    if normalize_model_name(str(n)) in registry_names
                    or str(n) in registry_names
                    else f"{n} (Model no longer configured)"
                    if n and str(n) != "nan"
                    else n
                )
            )

    # Compact preview cards for the most recent matches
    preview = filtered.head(3)
    st.markdown(
        '<div class="aic-panel-title" style="margin:0.35rem 0 0.25rem 0;">Recent matches</div>',
        unsafe_allow_html=True,
    )
    for _, row in preview.iterrows():
        inv = row.get("inventory_type") or "—"
        yard = row.get("yard") or "—"
        reviewed = row.get("reviewed_count")
        ai_count = row.get("ai_count")
        model = row.get("accepted_model") or "—"
        when = row.get("created_at") or "—"
        st.markdown(
            f"""
            <div class="aic-hist-card">
              <div class="aic-hist-card-top">
                <b>{inv}</b>
                <span class="aic-pill-rgb">Saved</span>
              </div>
              <div class="aic-hist-meta">
                {yard} · Reviewed {reviewed} · AI {ai_count}<br/>
                {model}<br/>
                {when}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_table, tab_export = st.tabs(["Full table", "Export & details"])
    with tab_table:
        st.dataframe(shown, hide_index=True, width="stretch", height=320)
    with tab_export:
        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="inventory_count_history.csv",
            mime="text/csv",
            key="hist_csv",
            width="stretch",
        )
        with st.expander("View full record details", expanded=False):
            st.dataframe(filtered, hide_index=True, width="stretch")


def _ai_config_test_image_bytes() -> tuple[bytes | None, str]:
    """Prefer Settings upload, then data/ai_config_test_image.*, never inventory uploads."""
    uploaded = st.session_state.get("ai_config_test_image_bytes")
    name = st.session_state.get("ai_config_test_image_name") or "settings_test.jpg"
    if isinstance(uploaded, (bytes, bytearray)) and uploaded:
        return bytes(uploaded), str(name)

    for candidate in (
        config.DATA_DIR / "ai_config_test_image.jpg",
        config.DATA_DIR / "ai_config_test_image.png",
        config.PROJECT_ROOT / "data" / "ai_config_test_image.jpg",
    ):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate.read_bytes(), candidate.name
    return None, ""


def _run_ai_configuration_test() -> dict[str, Any]:
    """Live lightweight validation — does not touch inventory analysis uploads."""
    import time

    started = time.perf_counter()
    result: dict[str, Any] = {
        "ok": False,
        "auth": "Unknown",
        "workflow": "Unknown",
        "response_source": "demo source" if config.DEMO_MODE else "live Roboflow",
        "demo_mode": config.DEMO_MODE,
        "api_key_configured": bool(config.ROBOFLOW_API_KEY) or config.DEMO_MODE,
        "processing_time": 0.0,
        "message": "",
        "test_image": None,
        "raw_prediction_count": None,
        "normalized_prediction_count": None,
        "detected_classes": [],
        "parser_status": "not_run",
        "details": {},
    }
    try:
        detector = RoboflowDetector()
        ok, msg = detector.test_connectivity()
        result["auth"] = "Successful" if ok else "Failed"
        result["auth_ok"] = bool(ok)
        result["message"] = msg
        result["details"]["connectivity"] = msg

        model = _primary_workflow_model()
        if model is None:
            result["workflow"] = "Missing"
            # Auth may still be fine — do not mark authentication failed.
            result["ok"] = bool(ok)
            result["message"] = (
                msg
                if not ok
                else "Authentication OK. No enabled workflow/model is configured in models.json."
            )
            result["processing_time"] = time.perf_counter() - started
            return result

        result["details"]["workspace"] = model.workspace_name
        result["details"]["workflow_id"] = model.workflow_id
        result["details"]["workflow_name"] = model.name
        result["details"]["image_input_name"] = model.image_input_name or "image"
        result["workflow"] = model.name

        if config.DEMO_MODE:
            result["ok"] = True
            result["auth"] = "Demo Mode"
            result["auth_ok"] = True
            result["response_source"] = "demo_mock"
            result["message"] = "Demo Mode active — live API not required."
            result["parser_status"] = "skipped_demo"
            result["processing_time"] = time.perf_counter() - started
            return result

        if not ok:
            result["workflow"] = "Unavailable"
            result["ok"] = False
            result["auth_ok"] = False
            result["processing_time"] = time.perf_counter() - started
            return result

        image_bytes, image_name = _ai_config_test_image_bytes()
        if not image_bytes:
            result["ok"] = True
            result["auth_ok"] = True
            result["message"] = (
                "Authentication OK. Upload a dedicated test image below "
                "(or place data/ai_config_test_image.jpg) to run a live inference probe."
            )
            result["parser_status"] = "skipped_no_test_image"
            result["details"]["inference"] = "skipped_no_test_image"
            result["processing_time"] = time.perf_counter() - started
            return result

        prepared = load_image_from_bytes(image_bytes, image_name)
        result["test_image"] = {
            "name": image_name,
            "dimensions": f"{prepared.original_width}x{prepared.original_height}",
        }
        conf = float(_form_get("confidence_threshold", 0.25))
        inference = run_inference_on_prepared_image(
            detector,
            prepared,
            model,
            confidence_threshold=conf,
            iou_threshold=float(_form_get("iou_threshold", 0.5)),
            inference_mode="Whole-image inference",
            deduplication_strategy="Conservative",
        )
        result["response_source"] = inference.source or detector.last_source
        result["raw_prediction_count"] = inference.raw_prediction_count
        result["normalized_prediction_count"] = inference.normalized_prediction_count
        result["detected_classes"] = sorted({d.class_name for d in inference.detections})
        result["parser_status"] = (
            "ok"
            if inference.request_completed and inference.error_type != "empty_workflow_output"
            else (inference.error_type or "error")
        )
        result["details"]["invocation_mode"] = inference.invocation_mode
        result["details"]["final_count"] = inference.final_count
        result["details"]["warnings"] = list(inference.warnings)
        result["details"]["error_type"] = inference.error_type
        result["details"]["shape"] = {
            "source": inference.source,
            "predictions_found": inference.predictions_found,
            "success": inference.success,
        }
        result["ok"] = bool(inference.request_completed and inference.source == "live_roboflow")
        result["auth_ok"] = True  # connectivity already succeeded
        result["message"] = (
            f"Authentication OK. Live probe: raw={inference.raw_prediction_count}, "
            f"normalized={inference.normalized_prediction_count}, "
            f"final={inference.final_count}."
        )
        result["details"]["annotated"] = bool(inference.annotated_image_bytes)
    except DetectorError as exc:
        from poc_ux import sanitize_public_text

        detail = sanitize_public_text(str(exc), max_len=400)
        result["ok"] = False
        result["message"] = detail
        result["parser_status"] = "api_error"
        # Preserve successful auth unless the error is clearly unauthorized.
        low = detail.lower()
        if result.get("auth_ok") and not (
            "401" in low or "unauthorized" in low or "forbidden" in low
        ):
            result["auth"] = "Successful"
            result["auth_ok"] = True
            result["message"] = (
                "Authentication OK, but the follow-up probe failed: " + detail
            )
        else:
            result["auth"] = result.get("auth") if result.get("auth_ok") else "Failed"
            result["auth_ok"] = bool(result.get("auth_ok"))
        st.session_state.last_diag_error = detail
    except Exception as exc:  # noqa: BLE001
        from poc_ux import sanitize_public_text

        detail = sanitize_public_text(f"{type(exc).__name__}: {exc}", max_len=400)
        result["ok"] = False
        result["message"] = detail
        result["parser_status"] = "unexpected_error"
        if result.get("auth_ok"):
            result["auth"] = "Successful"
            result["message"] = (
                "Authentication OK, but the follow-up probe failed: " + detail
            )
        else:
            result["auth"] = "Failed"
            result["auth_ok"] = False
        st.session_state.last_diag_error = detail
    result["processing_time"] = time.perf_counter() - started
    return result


def _render_advanced_settings() -> None:
    expanded = bool(st.session_state.get("open_advanced_settings"))
    with st.expander("Advanced Settings", expanded=expanded):
        st.caption(DEDUP_STRATEGY_EXPLAINER)

        mode_opts = ["Single Model", "Compare Models", "Experimental Consensus"]
        cur_mode = _form_get("selected_mode", mode_opts[0])
        mode = st.radio(
            "Analysis mode",
            mode_opts,
            index=mode_opts.index(cur_mode) if cur_mode in mode_opts else 0,
            horizontal=True,
            key="adv_mode",
        )
        _form_set(selected_mode=mode)

        enabled = _enabled_models()
        model_names = [m.name for m in enabled]
        max_models = 1 if mode == "Single Model" else 3
        prev = [n for n in (_form_get("selected_models") or []) if n in model_names]
        if not prev and model_names:
            prev = model_names[: max(1, 2 if mode == "Experimental Consensus" else 1)]
            prev = prev[:max_models]

        if not model_names:
            st.warning("No enabled models available. Check Diagnostics or enable Demo Mode.")

        selected = st.multiselect(
            "Models",
            options=model_names,
            default=prev,
            max_selections=max_models,
            key="adv_models",
        )
        _form_set(selected_models=selected)

        if mode == "Experimental Consensus":
            agree_opts = ["At least 1 model", "At least 2 models", "All selected models"]
            cur_agree = _form_get("agreement_label", agree_opts[1])
            _form_set(
                agreement_label=st.selectbox(
                    "Minimum agreement",
                    agree_opts,
                    index=agree_opts.index(cur_agree) if cur_agree in agree_opts else 1,
                    key="adv_agree",
                )
            )

        inventory_type = _resolved_inventory() or "Fence Panel"
        if inventory_type == "Fence Panel":
            presets = FENCE_PANEL_PROMPT_PRESETS
            cur_preset = _form_get("prompt_preset", presets[0])
            preset = st.selectbox(
                "Detection prompt preset",
                presets,
                index=presets.index(cur_preset) if cur_preset in presets else 0,
                key="adv_preset",
            )
            prompt = st.text_input(
                "Detection class override / prompt",
                value=_form_get("prompt") or preset,
                key="adv_prompt",
            )
            _form_set(prompt_preset=preset, prompt=prompt, class_override=prompt)
        else:
            default_p = DEFAULT_PROMPTS.get(inventory_type, inventory_type)
            prompt = st.text_input(
                "Detection class override / prompt",
                value=_form_get("prompt") or default_p,
                key="adv_prompt_other",
            )
            _form_set(prompt=prompt, class_override=prompt)

        conf = st.slider(
            "Confidence threshold",
            0.05,
            0.95,
            float(_form_get("confidence_threshold", 0.25)),
            0.05,
            key="adv_conf",
        )
        iou = st.slider(
            "Duplicate-overlap (IoU) threshold",
            0.1,
            0.9,
            float(_form_get("iou_threshold", 0.50)),
            0.05,
            key="adv_iou",
        )
        inf_opts = ["Whole Image", "Tiled", "Thorough Multi-Scale"]
        cur_inf = _form_get("inference_mode", inf_opts[0])
        inference_mode = st.selectbox(
            "Inference method",
            inf_opts,
            index=inf_opts.index(cur_inf) if cur_inf in inf_opts else 0,
            key="adv_inf",
        )

        tile_size = int(_form_get("tile_size", 800))
        tile_overlap = float(_form_get("tile_overlap", 0.25))
        if inference_mode in {"Tiled", "Thorough Multi-Scale"}:
            t1, t2 = st.columns(2)
            with t1:
                tile_size = st.selectbox(
                    "Tile size",
                    TILE_SIZES,
                    index=TILE_SIZES.index(tile_size) if tile_size in TILE_SIZES else 2,
                    key="adv_tile",
                )
            with t2:
                tile_overlap = st.selectbox(
                    "Tile overlap",
                    TILE_OVERLAPS,
                    index=TILE_OVERLAPS.index(tile_overlap)
                    if tile_overlap in TILE_OVERLAPS
                    else 2,
                    format_func=lambda x: f"{int(x * 100)}%",
                    key="adv_overlap",
                )

        dedup_opts = ["Conservative", "NMS", "NMM", "None/debug"]
        cur_dedup = _form_get("deduplication_strategy", "Conservative")
        dedup = st.selectbox(
            "Duplicate-removal strategy",
            dedup_opts,
            index=dedup_opts.index(cur_dedup) if cur_dedup in dedup_opts else 0,
            key="adv_dedup",
        )
        _form_set(
            confidence_threshold=conf,
            iou_threshold=iou,
            inference_mode=inference_mode,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            deduplication_strategy=dedup,
        )


def _render_ai_configuration_section() -> None:
    _ensure_selected_models()

    render_configuration_summary(show_actions=True)

    from catalog_ui import render_model_catalog_section

    def _catalog_model_test(
        model: ModelConfig,
        *,
        paid_confirmed: bool = False,
        inventory_key: str = "Boxes",
    ) -> dict[str, Any]:
        """Run a single-model settings probe without touching wizard uploads."""
        from model_adapters import InferenceOptions, get_adapter
        from openrouter import is_openrouter_model
        from openrouter_runtime import (
            UserModelTestState,
            get_openrouter_inference_key,
            is_auth_rejection_error,
            is_credential_failure_message,
            openrouter_credential_label,
            openrouter_credential_ready,
            preflight_openrouter_catalog_test,
            redacted_workflow_parameters,
            set_user_model_test_state,
        )
        from poc_ux import sanitize_public_text

        data, name = _ai_config_test_image_bytes()
        model_key = model.key or model.name
        out: dict[str, Any] = {
            "model_key": model_key,
            "ok": False,
            "auth": openrouter_credential_label()
            if (is_openrouter_model(model) or getattr(model, "requires_user_api_key", False))
            else ("Configured" if api_key_configured() else "Missing"),
            "response_source": None,
            "raw_prediction_count": 0,
            "normalized_prediction_count": 0,
            "detected_classes": [],
            "processing_time": 0.0,
            "parser_status": "not_run",
            "message": "",
            "error_message": None,
            "annotated_preview": None,
            "preflight": None,
            "usage_recorded": False,
            "paid_request": False,
            "parameters_redacted": None,
            "credential_failure": False,
            "schema": None,
        }

        needs_or = is_openrouter_model(model) or bool(
            getattr(model, "requires_user_api_key", False)
        )
        if needs_or:
            user = auth_session.current_user()
            preflight = preflight_openrouter_catalog_test(
                model,
                user,
                has_test_image=bool(data),
                paid_confirmed=bool(paid_confirmed),
                inventory_key=inventory_key,
            )
            out["preflight"] = preflight.to_public_dict()
            out["schema"] = (
                preflight.schema.to_public_dict() if preflight.schema else None
            )
            out["parameters_redacted"] = redacted_workflow_parameters(
                image_name=name or "test-image",
                classes=list(preflight.classes),
            )
            if not preflight.ok:
                out["message"] = preflight.message
                out["error_message"] = preflight.message
                out["credential_failure"] = preflight.reason_code in {
                    "missing_key",
                    "not_authenticated",
                } or is_credential_failure_message(preflight.message)
                set_user_model_test_state(
                    UserModelTestState(
                        model_key=str(model_key),
                        credential_status=(
                            "verified"
                            if openrouter_credential_ready()
                            else ("rejected" if "rejected" in preflight.message.lower() else "missing")
                        ),
                        test_status=(
                            "ready_to_test"
                            if preflight.reason_code == "confirmation_required"
                            else "blocked"
                        ),
                        message=preflight.message,
                        available_for_analyze=False,
                        schema_predictions_present=(
                            preflight.schema.has_predictions_output
                            if preflight.schema
                            else None
                        ),
                    )
                )
                return out

            inference_key = get_openrouter_inference_key()
            adapter = get_adapter(model, model_api_key=inference_key)
            prompt = ", ".join(preflight.classes)
            out["paid_request"] = True
        else:
            if (model.kind or "").lower() != "local" and not api_key_configured() and not config.DEMO_MODE:
                out["message"] = "API key not configured."
                return out
            if not data:
                out["message"] = "Upload a probe image or add data/ai_config_test_image.jpg."
                return out
            adapter = get_adapter(model)
            prompt = config.inventory_detection_prompt(inventory_key or "Fence Panel")

        try:
            prepared = load_image_from_bytes(data, name or "probe.jpg")
            opts = InferenceOptions(
                prompt=prompt,
                confidence_threshold=float(model.default_confidence or 0.25),
                iou_threshold=float(model.default_iou or 0.5),
            )
            mir = adapter.predict(prepared, opts)
            zero = bool(mir.success) and len(mir.detections) == 0 and int(mir.raw_count or 0) == 0
            message = mir.error_message or (
                "Successful zero detections"
                if zero
                else ("OK" if mir.success else "Failed")
            )
            out.update(
                {
                    "ok": bool(mir.success),
                    "response_source": mir.response_source,
                    "raw_prediction_count": mir.raw_count,
                    "normalized_prediction_count": len(mir.detections),
                    "detected_classes": list(mir.classes),
                    "processing_time": mir.processing_time_seconds,
                    "parser_status": "ok" if mir.success else (mir.error_type or "failed"),
                    "message": message,
                    "error_message": None if mir.success else mir.error_message,
                    "annotated_preview": mir.annotated_image_bytes,
                }
            )
            if needs_or:
                if mir.success:
                    # Record quota once per successful paid Catalog Test execution.
                    run_token = f"catalog:{model_key}:{prepared.content_hash}:{prompt}"
                    seen = set(st.session_state.get("catalog_usage_tokens") or [])
                    if run_token not in seen:
                        _record_model_run(model)
                        seen.add(run_token)
                        st.session_state.catalog_usage_tokens = list(seen)
                        out["usage_recorded"] = True
                    set_user_model_test_state(
                        UserModelTestState(
                            model_key=str(model_key),
                            credential_status="verified",
                            test_status=(
                                "successful_zero_detections" if zero else "successful"
                            ),
                            message=message,
                            available_for_analyze=True,
                            last_tested_at=datetime.now(timezone.utc).isoformat(),
                            schema_predictions_present=True,
                        )
                    )
                else:
                    err = sanitize_public_text(mir.error_message or message, max_len=300)
                    out["message"] = err
                    out["error_message"] = err
                    out["credential_failure"] = is_auth_rejection_error(err) or is_credential_failure_message(err)
                    set_user_model_test_state(
                        UserModelTestState(
                            model_key=str(model_key),
                            credential_status=(
                                "rejected" if is_auth_rejection_error(err) else "verified"
                            ),
                            test_status="failed",
                            message=err,
                            available_for_analyze=False,
                            last_tested_at=datetime.now(timezone.utc).isoformat(),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            from poc_ux import sanitize_public_text as _sanitize

            err = _sanitize(f"{type(exc).__name__}: {exc}", max_len=300)
            out["message"] = err
            out["error_message"] = err
            out["credential_failure"] = is_auth_rejection_error(err) or is_credential_failure_message(err)
            if needs_or:
                set_user_model_test_state(
                    UserModelTestState(
                        model_key=str(model_key),
                        credential_status=(
                            "rejected" if is_auth_rejection_error(err) else "verified"
                        ),
                        test_status="failed",
                        message=err,
                        available_for_analyze=False,
                        last_tested_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
        return out

    tab_catalog, tab_probe, tab_benchmark, tab_advanced = st.tabs(
        ["Model Catalog", "Probe & Test", "Detection Benchmark", "Advanced & Samples"]
    )

    with tab_catalog:
        st.markdown(
            '<div class="aic-panel aic-panel-b"><div class="aic-panel-title">'
            "Model catalog</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">Browse workspace, foundation, and local adapters.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        render_model_catalog_section(
            run_model_test=_catalog_model_test,
            get_test_image_bytes=_ai_config_test_image_bytes,
        )
        with st.expander("Legacy registry table", expanded=False):
            models = _all_models()
            summary = summarize_models(models)
            if summary:
                st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
            st.caption(
                "Demo/local classical entries are excluded from the live Analysis selector when "
                "DEMO_MODE is false. Local Picket Counter is a NumPy/PIL heuristic in picket_counter.py, "
                "not a Roboflow model."
            )

    with tab_probe:
        st.markdown(
            '<div class="aic-panel aic-panel-r"><div class="aic-panel-title">'
            "Configuration probe</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">"
            "Optional test image for AI Configuration only — never replaces inventory uploads."
            "</p></div>",
            unsafe_allow_html=True,
        )
        probe = st.file_uploader(
            "Upload dedicated AI test image",
            type=["jpg", "jpeg", "png", "webp"],
            key="ai_config_test_uploader",
            accept_multiple_files=False,
        )
        if probe is not None:
            probe.seek(0)
            st.session_state.ai_config_test_image_bytes = probe.read()
            st.session_state.ai_config_test_image_name = probe.name
            probe.seek(0)
            st.caption(f"Probe image ready: {probe.name}")
        else:
            st.caption("If omitted, `data/ai_config_test_image.jpg` is used when present.")

        st.markdown(
            '<div class="aic-panel aic-panel-g"><div class="aic-panel-title">'
            "Test models</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("Test AI Configuration", type="primary", key="cfg_test_btn", width="stretch"):
            with st.spinner("Testing AI configuration…"):
                st.session_state.ai_config_test_result = _run_ai_configuration_test()
                test = st.session_state.ai_config_test_result or {}
                stamp = {
                    "status": "OK" if test.get("ok") else "Failed",
                    "when": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "message": (test.get("message") or "")[:200],
                }
                results_map = dict(st.session_state.get("model_test_results") or {})
                for m in _enabled_models():
                    results_map[m.name] = stamp
                st.session_state.model_test_results = results_map
            st.rerun()

        test = st.session_state.get("ai_config_test_result")
        if isinstance(test, dict):
            if test.get("ok"):
                st.success("AI configuration is working")
                st.markdown(
                    f"""
                    <div class="aic-chip-grid aic-chip-grid-4">
                      <div class="aic-chip aic-chip-g"><span class="aic-chip-label">Auth</span>
                        <span class="aic-chip-value">{test.get("auth")}</span></div>
                      <div class="aic-chip aic-chip-b"><span class="aic-chip-label">Source</span>
                        <span class="aic-chip-value">{test.get("response_source") or "—"}</span></div>
                      <div class="aic-chip aic-chip-r"><span class="aic-chip-label">Raw preds</span>
                        <span class="aic-chip-value">{test.get("raw_prediction_count")}</span></div>
                      <div class="aic-chip aic-chip-g"><span class="aic-chip-label">Normalized</span>
                        <span class="aic-chip-value">{test.get("normalized_prediction_count")}</span></div>
                    </div>
                    <div class="aic-kv-grid" style="margin-top:0.45rem;">
                      <div class="aic-kv"><b>Workspace</b><br/>{(test.get("details") or {}).get("workspace") or "—"}</div>
                      <div class="aic-kv"><b>Workflow</b><br/>{test.get("workflow") or "—"}</div>
                      <div class="aic-kv"><b>Classes</b><br/>{", ".join(test.get("detected_classes") or []) or "(none)"}</div>
                      <div class="aic-kv"><b>Time</b><br/>{float(test.get("processing_time") or 0):.2f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("AI configuration test failed")
                st.write(test.get("message") or "Unknown error.")
            with st.expander("View sanitized response details", expanded=False):
                st.json(test)

        st.markdown("##### Model test history")
        history_map = st.session_state.get("model_test_results") or {}
        if history_map:
            rows = [
                {
                    "Model": name,
                    "Status": info.get("status"),
                    "When": info.get("when"),
                    "Notes": info.get("message", ""),
                }
                for name, info in history_map.items()
                if isinstance(info, dict)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=220)
        else:
            st.caption("No model tests run in this session yet.")

    with tab_benchmark:
        from benchmark_ui import render_detection_benchmark_section

        yolo = next(
            (
                m
                for m in _enabled_models()
                if (m.name or "") == "YOLO-World"
                or (
                    (m.kind or "").lower() == "workflow"
                    and (m.supports_prompt or m.dynamic_classes)
                )
            ),
            None,
        )
        render_detection_benchmark_section(
            run_yolo_world=_run_benchmark_yolo_world,
            yolo_model_key=(yolo.key if yolo and yolo.key else "workflow:hariram-s-mzhvc/custom-workflow"),
            api_ready=api_key_configured(),
            demo_mode=bool(config.DEMO_MODE),
        )

    with tab_advanced:
        st.markdown(
            '<div class="aic-panel aic-panel-b"><div class="aic-panel-title">'
            "Advanced defaults</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">"
            "Tuning for confidence, tiling, and deduplication."
            "</p></div>",
            unsafe_allow_html=True,
        )
        _render_advanced_settings()
        _render_inventory_prompt_profiles()
        _render_sample_library_settings()


def _run_benchmark_yolo_world(
    *,
    image_bytes: bytes,
    image_name: str,
    prompts: list[str],
    prompt_set_label: str,
    confidence_threshold: float = 0.25,
) -> Any:
    """Run YOLO-World for Detection Benchmark without touching wizard state."""
    from benchmark import BenchmarkRunOutcome
    from model_adapters import InferenceOptions, get_adapter, model_key
    from inventory_profiles import prompts_to_csv as _prompts_to_csv

    # Snapshot wizard keys — must remain unchanged after this call.
    wizard_keys = (
        "analysis_results",
        "run_context",
        "uploaded_images",
        "stage",
        "inventory_choice",
        "form",
    )
    before = {k: st.session_state.get(k) for k in wizard_keys}

    yolo = next(
        (
            m
            for m in _enabled_models()
            if (m.name or "") == "YOLO-World"
            or (
                (m.kind or "").lower() == "workflow"
                and (m.supports_prompt or m.dynamic_classes)
            )
        ),
        None,
    )
    outcome = BenchmarkRunOutcome(
        prompt_set_label=prompt_set_label,
        prompt_set=list(prompts),
    )
    if yolo is None:
        outcome.execution_failed = True
        outcome.error_message = "YOLO-World workflow is not enabled."
        return outcome

    try:
        prepared = load_image_from_bytes(image_bytes, image_name or "benchmark.jpg")
        adapter = get_adapter(yolo)
        mir = adapter.predict(
            prepared,
            InferenceOptions(
                prompt=_prompts_to_csv(prompts),
                confidence_threshold=float(confidence_threshold),
                iou_threshold=float(yolo.default_iou or 0.5),
            ),
        )
        det = adapter.detector
        injection = getattr(det, "last_injection_result", None) or {}
        outcome.success = bool(mir.success)
        outcome.execution_failed = not bool(mir.success)
        outcome.raw_count = int(mir.raw_count or 0)
        outcome.normalized_count = len(mir.detections)
        outcome.final_count = int(mir.final_count or 0)
        outcome.returned_classes = list(mir.classes or [])
        outcome.processing_time = float(mir.processing_time_seconds or 0.0)
        outcome.invocation_mode = getattr(det, "last_invocation_mode", None) or (
            (mir.technical_details or {}).get("invocation_mode")
        )
        outcome.fallback_used = bool(getattr(det, "last_empty_draft_fallback", False))
        outcome.matched_step_id = (injection.get("matched_step_ids") or [None])[0]
        outcome.matched_step_type = (injection.get("matched_step_types") or [None])[0]
        outcome.field_injected = injection.get("field_used")
        outcome.warnings = list(mir.warnings or [])
        outcome.warning_count = len(outcome.warnings)
        outcome.error_message = mir.error_message
        outcome.annotated_image_bytes = mir.annotated_image_bytes
        outcome.detections = [d.to_dict() for d in mir.detections]
        outcome.technical = {
            "prompt_injection_status": (
                "injected" if injection.get("injected") else "not_injected"
            ),
            "matched_step": outcome.matched_step_id,
            "matched_step_type": outcome.matched_step_type,
            "invocation_mode": outcome.invocation_mode,
            "fallback_used": outcome.fallback_used,
            "response_source": mir.response_source,
            "raw_response_type": type(
                getattr(mir.inference_result, "source", None)
            ).__name__
            if mir.inference_result
            else None,
            "parser_status": "ok" if mir.success else (mir.error_type or "failed"),
            "normalized_count": outcome.normalized_count,
            "annotation_status": (
                "present" if outcome.annotated_image_bytes else "missing"
            ),
            "dynamic_prompt_status": getattr(det, "last_dynamic_prompt_status", None),
            "model_key": model_key(yolo),
            "injection": injection,
        }
        if outcome.fallback_used:
            outcome.execution_failed = True
            outcome.error_message = (
                (outcome.error_message or "")
                + " Unmodified workflow fallback is not allowed for benchmark runs."
            ).strip()
            outcome.success = False
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        outcome.execution_failed = True
        outcome.success = False
        outcome.error_message = f"{type(exc).__name__}: {exc}"

    after = {k: st.session_state.get(k) for k in wizard_keys}
    if before != after:
        # Restore wizard keys if somehow mutated.
        for k, v in before.items():
            if k in st.session_state or v is not None:
                st.session_state[k] = v
        outcome.warnings.append("Wizard session keys were restored after benchmark run.")
        outcome.warning_count = len(outcome.warnings)
    return outcome


def _render_inventory_prompt_profiles() -> None:
    """Compact read-only Inventory Prompt Profiles (Settings → AI Configuration)."""
    st.markdown(
        '<div class="aic-panel"><div class="aic-panel-title">'
        "Inventory prompt profiles</div>"
        "<p class=\"aic-muted\" style=\"margin:0;\">"
        "Presets from inventory_profiles.json. API keys are never shown."
        "</p></div>",
        unsafe_allow_html=True,
    )
    rows = []
    for p in load_inventory_profiles():
        if p.get("is_custom"):
            terms = "(user-entered at setup)"
        else:
            terms = prompts_to_csv(list(p.get("prompt_terms") or [])) or "—"
        rows.append(
            {
                "Inventory": p.get("display_name") or p.get("key"),
                "Prompt terms": terms,
                "Default confidence": p.get("default_confidence"),
                "Enabled": "Yes" if p.get("enabled") else "No",
                "Counting unit": p.get("counting_unit") or "—",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=260)
    else:
        st.caption("No inventory profiles loaded.")


def _render_sample_library_settings() -> None:
    """Compact read-only Built-in Sample Library status (no full gallery)."""
    st.markdown(
        '<div class="aic-panel aic-panel-g"><div class="aic-panel-title">'
        "Built-in sample library</div></div>",
        unsafe_allow_html=True,
    )
    status = load_sample_library(force_reload=True)
    manifest = (
        "OK"
        if status.manifest_valid
        else ("Invalid" if status.manifest_exists else "Missing")
    )
    st.markdown(
        f"""
        <div class="aic-chip-grid aic-chip-grid-4">
          <div class="aic-chip aic-chip-g"><span class="aic-chip-label">Directory</span>
            <span class="aic-chip-value">{"OK" if status.directory_exists else "Missing"}</span></div>
          <div class="aic-chip aic-chip-b"><span class="aic-chip-label">Manifest</span>
            <span class="aic-chip-value">{manifest}</span></div>
          <div class="aic-chip aic-chip-r"><span class="aic-chip-label">Valid</span>
            <span class="aic-chip-value">{status.valid_count}</span></div>
          <div class="aic-chip aic-chip-g"><span class="aic-chip-label">Enabled</span>
            <span class="aic-chip-value">{status.enabled_count}</span></div>
        </div>
        <div class="aic-kv-grid" style="margin-top:0.35rem;">
          <div class="aic-kv"><b>Missing files</b><br/>{len(status.missing_files)}</div>
          <div class="aic-kv"><b>Invalid files</b><br/>{len(status.invalid_files)}</div>
          <div class="aic-kv"><b>Duplicate IDs</b><br/>{len(status.duplicate_ids)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if status.manifest_error:
        st.caption(status.manifest_error)
    if status.missing_files:
        st.caption("Missing: " + ", ".join(status.missing_files[:8]))
    if status.invalid_files:
        st.caption("Invalid: " + ", ".join(status.invalid_files[:8]))
    if status.duplicate_ids:
        st.caption("Duplicate IDs: " + ", ".join(status.duplicate_ids[:8]))
    st.caption(
        "Gallery lives under Add Photos → Sample Images. "
        "Files are loaded from the project `assets/sample_images/` folder."
    )



_JSON_SAFE_TYPES = (str, bool, int, float)


def _json_safe_value(value: Any) -> Any:
    """Convert values to Streamlit/JSON-safe primitives (no objects)."""
    if value is None:
        return ""
    if isinstance(value, _JSON_SAFE_TYPES):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return str(value)


def _build_inference_sdk_probe() -> dict[str, Any]:
    """Build a plain serializable diagnostics dict for inference-sdk.

    Never returns modules, clients, exception objects, responses, or callables —
    only str/bool/int/float/list/dict values.
    """
    probe: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "sdk_version": "",
        "sdk_location": "",
        "api_key_present": bool(str(config.ROBOFLOW_API_KEY or "").strip()),
        "workspace": str(
            getattr(config, "ROBOFLOW_WORKSPACE", "")
            or getattr(config, "YOLO_WORLD_WORKSPACE", "")
            or ""
        ),
        "workflow": str(
            getattr(config, "ROBOFLOW_WORKFLOW_ID", "")
            or getattr(config, "YOLO_WORLD_WORKFLOW_ID", "")
            or ""
        ),
        "client_created": False,
        "error_type": "",
        "error_message": "",
    }
    try:
        model = _primary_workflow_model()
        if model is not None:
            if model.workspace_name:
                probe["workspace"] = str(model.workspace_name)
            if model.workflow_id:
                probe["workflow"] = str(model.workflow_id)
    except Exception:
        # Session models may be unavailable outside a Streamlit run.
        pass
    try:
        import inference_sdk

        probe["sdk_version"] = str(getattr(inference_sdk, "__version__", "unknown"))
        probe["sdk_location"] = str(getattr(inference_sdk, "__file__", "") or "")
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url=str(config.ROBOFLOW_API_URL or "https://detect.roboflow.com"),
            api_key=config.ROBOFLOW_API_KEY,
        )
        probe["client_created"] = True
        # Do not retain client / module references beyond this scope.
        del client
    except Exception as exc:
        probe["error_type"] = type(exc).__name__
        probe["error_message"] = str(exc)
        print(traceback.format_exc())

    return {str(k): _json_safe_value(v) for k, v in probe.items()}


def _render_diagnostics_section() -> None:
    snap = _config_snapshot()
    demo_label = "On" if config.DEMO_MODE else "Off"
    st.markdown(
        f"""
        <div class="aic-chip-grid">
          <div class="aic-chip aic-chip-r">
            <span class="aic-chip-label">API key</span>
            <span class="aic-chip-value">{snap["api_key"]}</span>
          </div>
          <div class="aic-chip aic-chip-b">
            <span class="aic-chip-label">Demo mode</span>
            <span class="aic-chip-value">{demo_label}</span>
          </div>
          <div class="aic-chip aic-chip-g">
            <span class="aic-chip-label">Connection</span>
            <span class="aic-chip-value">{snap["connection_label"]}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Demo Mode uses stored sample predictions from `sample_responses/mock_detection.json` "
        "instead of calling the live Roboflow workflow."
        if config.DEMO_MODE
        else "Demo Mode is off: detection uses live Roboflow or local adapters only — "
        "mock predictions are not substituted."
    )

    # Never persist SDK clients / modules / exceptions in session_state.
    st.session_state.pop("diag_sdk_probe", None)

    left, right = st.columns([1.15, 1], gap="medium")
    with left:
        st.markdown(
            f"""
            <div class="aic-panel aic-panel-g">
              <div class="aic-panel-title">Runtime snapshot</div>
              <div class="aic-kv-grid">
                <div class="aic-kv"><b>Provider</b><br/>{snap["provider"]}</div>
                <div class="aic-kv"><b>Workspace</b><br/>{snap["workspace"]}</div>
                <div class="aic-kv"><b>Workflow ID</b><br/>{snap["workflow_id"]}</div>
                <div class="aic-kv"><b>Detection mode</b><br/>{snap["detection_mode"]}</div>
                <div class="aic-kv"><b>Response source</b><br/>{snap["response_source"]}</div>
                <div class="aic-kv"><b>API status</b><br/>{masked_api_key_status()}</div>
                <div class="aic-kv"><b>Config file</b><br/>{snap["models_path"]}</div>
                <div class="aic-kv"><b>Database</b><br/>{config.DB_PATH.name}</div>
                <div class="aic-kv"><b>Python</b><br/>{sys.version.split()[0]}</div>
                <div class="aic-kv"><b>Streamlit health</b><br/>/_stcore/health</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Manage models under AI Configuration in the left panel.")

    with right:
        st.markdown(
            '<div class="aic-panel aic-panel-r"><div class="aic-panel-title">'
            "Quick checks</div>"
            "<p class=\"aic-muted\" style=\"margin:0 0 0.35rem 0;\">"
            "Real exception types/messages — never masked. Probe results are display-only."
            "</p></div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "Run inference SDK / Roboflow probe",
            key="diag_run_sdk_probe",
            width="stretch",
        ):
            st.session_state.inference_cache = {}
            probe = _build_inference_sdk_probe()
            print(type(probe))
            print(repr(probe))
            st.json(probe)
            if probe.get("error_type"):
                st.error(f"{probe.get('error_type')}: {probe.get('error_message', '')}")
                st.session_state.last_diag_error = (
                    f"{probe.get('error_type')}: {probe.get('error_message', '')}"
                )
            elif probe.get("client_created"):
                st.success(
                    f"Client created OK — inference-sdk {probe.get('sdk_version', '?')}"
                )

        if st.button("Test API connectivity", key="diag_test", width="stretch"):
            try:
                with st.spinner("Testing connectivity…"):
                    ok, msg = RoboflowDetector().test_connectivity()
                (st.success if ok else st.error)(msg)
                if not ok:
                    st.session_state.last_diag_error = msg
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                detail = f"{type(exc).__name__}: {exc}"
                st.session_state.last_diag_error = detail
                _error_box("Connectivity test failed.", detail)

        sample_warns = sample_library_diagnostics_warnings()
        if sample_warns:
            st.markdown(
                '<div class="aic-panel aic-panel-b"><div class="aic-panel-title">'
                "Sample library warnings</div></div>",
                unsafe_allow_html=True,
            )
            for w in sample_warns[:8]:
                st.warning(w)
        else:
            st.caption("Sample library: no warnings.")

    st.markdown("---")
    st.markdown("#### Model Catalog")
    st.caption(
        "Compact sync/validation status. Full model cards live under "
        "AI Configuration → Model Catalog (left panel)."
    )
    try:
        from model_catalog import catalog_diagnostics_summary

        cat_diag = catalog_diagnostics_summary()
        st.markdown(
            f"""
            <div class="aic-panel aic-panel-b">
              <div class="aic-kv-grid">
                <div class="aic-kv"><b>Last workspace sync</b><br/>{cat_diag.get("last_workspace_sync") or "—"}</div>
                <div class="aic-kv"><b>Authentication</b><br/>{cat_diag.get("authentication_status") or "—"}</div>
                <div class="aic-kv"><b>Projects discovered</b><br/>{cat_diag.get("projects_discovered") if cat_diag.get("projects_discovered") is not None else "—"}</div>
                <div class="aic-kv"><b>Versions inspected</b><br/>{cat_diag.get("versions_inspected") if cat_diag.get("versions_inspected") is not None else "—"}</div>
                <div class="aic-kv"><b>Trained OD models</b><br/>{cat_diag.get("trained_object_detection_models_found") if cat_diag.get("trained_object_detection_models_found") is not None else "—"}</div>
                <div class="aic-kv"><b>Models added / updated / stale</b><br/>
                  {cat_diag.get("models_added") or 0} / {cat_diag.get("models_updated") or 0} / {cat_diag.get("models_marked_stale") or 0}
                </div>
                <div class="aic-kv"><b>Live validated</b><br/>{cat_diag.get("live_validated_models")}</div>
                <div class="aic-kv"><b>Metadata only</b><br/>{cat_diag.get("metadata_only_models")}</div>
                <div class="aic-kv"><b>Unavailable adapters</b><br/>{cat_diag.get("unavailable_adapters")}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        errs = cat_diag.get("sanitized_errors") or []
        if errs:
            with st.expander("Catalog sync errors (sanitized)", expanded=False):
                for e in errs:
                    st.caption(str(e))
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Catalog diagnostics unavailable: {type(exc).__name__}")

    st.markdown("---")
    st.markdown("#### Dynamic Prompt Verification")
    st.caption(
        "Choose a Workflow model, then prove inventory prompts are applied for that "
        "run (injected specification / no silent unmodified fallback where applicable)."
    )
    verify_models = _dynamic_prompt_verify_models()
    verify_by_name = {m.name: m for m in verify_models}
    if not verify_models:
        st.warning("No enabled Workflow models with a workflow_id are available.")
    profiles = enabled_profiles()
    profile_labels = {
        p["key"]: p.get("display_name") or p["key"] for p in profiles
    }
    profile_keys = list(profile_labels.keys())
    if "Custom Item" not in profile_keys:
        profile_keys.append("Custom Item")
        profile_labels["Custom Item"] = "Custom Item"
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        diag_model_name = st.selectbox(
            "Model",
            options=[m.name for m in verify_models] or ["(none)"],
            key="diag_dyn_model",
            help="Enabled Workflow models from models.json / the catalog.",
        )
        chosen_verify = verify_by_name.get(diag_model_name)
        if chosen_verify is not None:
            st.caption(
                f"Workflow: `{chosen_verify.workflow_id}` · "
                f"workspace: `{chosen_verify.workspace_name or '—'}`"
            )
        diag_inv = st.selectbox(
            "Inventory profile",
            options=profile_keys,
            format_func=lambda k: profile_labels.get(k, k),
            key="diag_dyn_inventory",
        )
        diag_custom_name = ""
        diag_custom_alt = ""
        if is_custom_inventory(diag_inv):
            diag_custom_name = st.text_input(
                "Custom item name",
                value="wooden gate",
                key="diag_dyn_custom_name",
            )
            diag_custom_alt = st.text_input(
                "Alternate terms (comma-separated)",
                value="driveway gate, gate panel",
                key="diag_dyn_custom_alt",
            )
        diag_prompts, diag_prompt_errs = effective_prompts_for_inventory(
            diag_inv,
            custom_item_name=diag_custom_name or None,
            custom_alternatives=diag_custom_alt or None,
        )
        if diag_prompt_errs:
            for err in diag_prompt_errs:
                st.warning(err)
        st.caption("Effective prompts: " + (", ".join(diag_prompts) if diag_prompts else "(none)"))
        diag_conf = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            key="diag_dyn_confidence",
        )
    with dcol2:
        samples = list_enabled_samples(inventory_key=None)
        sample_ids = [s.id for s in samples]
        sample_titles = {s.id: s.title for s in samples}
        diag_sample = st.selectbox(
            "Test image (sample library)",
            options=sample_ids or ["(none)"],
            format_func=lambda i: sample_titles.get(i, i),
            key="diag_dyn_sample",
        )
        diag_upload = st.file_uploader(
            "Or upload a test image",
            type=["jpg", "jpeg", "png", "webp"],
            key="diag_dyn_upload",
        )
    if st.button(
        "Run Dynamic Prompt Verification",
        key="diag_dyn_run",
        width="stretch",
        disabled=config.DEMO_MODE or not api_key_configured() or not verify_models,
    ):
        model = verify_by_name.get(diag_model_name)
        if model is None:
            st.error("Select an available Workflow model to verify.")
        elif not diag_prompts:
            st.error("No effective prompts to verify.")
        else:
            tmp_path = None
            try:
                prepared = None
                if diag_upload is not None:
                    raw = diag_upload.getvalue()
                    name = diag_upload.name or "diag_upload.jpg"
                    prepared = load_image_from_bytes(raw, name)
                elif sample_ids and diag_sample in sample_ids:
                    sample = get_sample_by_id(diag_sample)
                    if sample is None:
                        st.error("Selected sample could not be loaded.")
                    else:
                        raw = read_sample_bytes(sample)
                        prepared = load_image_from_bytes(raw, sample.filename)
                else:
                    st.error("Provide a sample or uploaded test image.")
                if prepared is not None:
                    import tempfile
                    from pathlib import Path as _Path

                    from image_processing import save_temp_image

                    tmp_dir = _Path(tempfile.mkdtemp(prefix="aic_diag_dyn_"))
                    tmp_path = save_temp_image(prepared.inference, tmp_dir)
                    with st.spinner("Verifying dynamic prompt propagation…"):
                        report = _verify_dynamic_prompt_propagation(
                            RoboflowDetector(demo_mode=False),
                            model,
                            str(tmp_path),
                            class_names=diag_prompts,
                            confidence_threshold=float(diag_conf),
                            inventory_key=diag_inv,
                        )
                    st.session_state.diag_dyn_report = {
                        **{
                            k: v
                            for k, v in report.items()
                            if k != "annotated_image_bytes"
                        },
                        "model_name": model.name,
                        "workflow_id": model.workflow_id,
                    }
                    st.session_state.diag_dyn_annotated = report.get(
                        "annotated_image_bytes"
                    )
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                st.session_state.last_diag_error = f"{type(exc).__name__}: {exc}"
                st.error(st.session_state.last_diag_error)
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink(missing_ok=True)
                        tmp_path.parent.rmdir()
                    except Exception:  # noqa: BLE001
                        pass

    if config.DEMO_MODE:
        st.info("Dynamic Prompt Verification requires DEMO_MODE=false and a configured API key.")
    elif not api_key_configured():
        st.warning("ROBOFLOW_API_KEY is missing — live verification disabled.")

    dyn_report = st.session_state.get("diag_dyn_report")
    if dyn_report:
        status = dyn_report.get("status", "?")
        if status in {"VERIFIED_DYNAMIC", "SUCCESSFUL_ZERO_DETECTIONS"}:
            st.success(f"Status: **{status}**")
        elif status == "WORKFLOW_NOT_DYNAMIC":
            st.error(f"Status: **{status}**")
        else:
            st.warning(f"Status: **{status}**")
        display = {
            k: dyn_report.get(k)
            for k in (
                "status",
                "workflow_id",
                "invocation_mode",
                "workflow_spec_fetch_status",
                "compatible_block_found",
                "matched_step_id",
                "matched_step_type",
                "field_injected",
                "injected_class_names",
                "fallback_used",
                "raw_returned_classes",
                "raw_count",
                "normalized_count",
                "processing_time_seconds",
                "sanitized_error",
            )
        }
        st.json(display)
        ann = st.session_state.get("diag_dyn_annotated")
        if ann:
            st.image(ann, caption="Annotated preview", width="stretch")

    tab_err, tab_req, tab_raw, tab_env = st.tabs(
        ["Errors", "Last request", "Raw response", "Environment"]
    )

    with tab_err:
        if st.session_state.get("last_diag_error"):
            st.code(st.session_state.last_diag_error)
        else:
            st.caption("No API/parser errors recorded in this session.")

    with tab_req:
        results: list[InferenceResult] = st.session_state.analysis_results or []
        if not results:
            st.caption("No analysis has been run in this session.")
        else:
            rows = [
                {
                    "image": r.image_name,
                    "model": r.model_name,
                    "source": r.source,
                    "error_type": r.error_type,
                    "raw": r.raw_prediction_count,
                    "final": r.final_count,
                }
                for r in results
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=260)

    with tab_raw:
        debug_path = config.DATA_DIR / "debug" / "last_live_response.json"
        shape_path = config.DATA_DIR / "last_live_response_shape.json"
        if shape_path.exists():
            st.caption(str(shape_path))
            try:
                st.json(json.loads(shape_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                st.caption("Could not parse response shape file.")
        elif debug_path.exists():
            st.caption(
                f"Response dump present at {debug_path.name} (open on disk; not echoed here)."
            )
        else:
            st.caption("No saved live response yet.")

    with tab_env:
        env_l, env_r = st.columns(2)
        with env_l:
            st.markdown(
                f"""
                <div class="aic-panel aic-panel-b">
                  <div class="aic-panel-title">Environment</div>
                  <div class="aic-kv"><b>DEMO_MODE</b><br/>{config.DEMO_MODE}</div>
                  <div class="aic-kv"><b>ROBOFLOW_API_URL</b><br/>{config.ROBOFLOW_API_URL}</div>
                  <div class="aic-kv"><b>DATA_DIR</b><br/>{config.DATA_DIR}</div>
                  <div class="aic-kv"><b>API key configured</b><br/>{"Yes" if api_key_configured() else "No"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with env_r:
            pkgs: dict[str, str] = {}
            for name in ("streamlit", "PIL", "numpy", "pandas", "dotenv"):
                try:
                    mod = __import__(name if name != "PIL" else "PIL")
                    pkgs[name] = getattr(mod, "__version__", "unknown")
                except Exception:  # noqa: BLE001
                    pkgs[name] = "not installed"
            try:
                import inference_sdk

                pkgs["inference-sdk"] = getattr(inference_sdk, "__version__", "installed")
            except Exception as exc:  # noqa: BLE001 — show real import failure (do not mask)
                traceback.print_exc()
                pkgs["inference-sdk"] = f"import failed: {type(exc).__name__}: {exc}"
            st.markdown(
                '<div class="aic-panel aic-panel-g"><div class="aic-panel-title">'
                "Runtime packages</div></div>",
                unsafe_allow_html=True,
            )
            st.json(pkgs)


def view_panel(view: str, user) -> None:
    """Render a dedicated left-panel destination (Settings shell removed)."""
    render_page_hero(
        PANEL_TITLES.get(view, view.replace("_", " ").title()),
        PANEL_CAPTIONS.get(view, ""),
    )
    if view == "history":
        _render_history_section()
    elif view == "ai_configuration":
        _render_ai_configuration_section()
    elif view == "diagnostics":
        _render_diagnostics_section()
    elif view == "account":
        auth_ui.render_account_page(user)
    elif view == "api_keys":
        if user.is_admin:
            api_connections_ui.render_api_connections_page(user)
        else:
            st.error("Only administrators can manage API keys.")
    else:
        view_welcome(user)


# ---------------------------------------------------------------------------
# Stage 1 — Inventory Setup (dynamic inventory profiles)
# ---------------------------------------------------------------------------


def stage_setup() -> None:
    render_stepper("setup")
    render_stage_header(
        "Inventory Setup",
        "Choose the yard and inventory type you are counting.",
    )

    # Always fix photo relationship (no user selector).
    _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)

    yard_choice = st.selectbox(
        "Yard / location",
        YARD_OPTIONS,
        index=YARD_OPTIONS.index(_form_get("yard_choice", "LA Yard"))
        if _form_get("yard_choice", "LA Yard") in YARD_OPTIONS
        else 0,
        key="inv_yard",
    )
    yard_custom = _form_get("yard_custom", "")
    if yard_choice == "Other":
        yard_custom = st.text_input("Custom location name", value=yard_custom, key="inv_yard_custom")
    _form_set(yard_choice=yard_choice, yard_custom=yard_custom)

    st.markdown("#### Inventory Type")
    current = _form_get("inventory_choice", "") or ""
    profiles = enabled_profiles() or [{"key": SELECTABLE_INVENTORY_KEY, "display_name": "Fence Panels"}]
    type_keys = [p["key"] for p in profiles]
    # Prefer JSON registry; fall back to config.INVENTORY_TYPES
    if not type_keys:
        type_keys = list(INVENTORY_TYPES)

    n_cols = 4 if len(type_keys) >= 4 else 3
    cols = st.columns(n_cols)
    for i, inv in enumerate(type_keys):
        with cols[i % n_cols]:
            selectable = is_inventory_selectable(inv)
            display = inventory_display_name(inv) if selectable else inv
            if selectable:
                selected = current == inv
                label = f"✓ {display}" if selected else display
                if st.button(
                    label,
                    width="stretch",
                    type="primary" if selected else "secondary",
                    key=f"inv_tile_{inv}",
                ):
                    _form_set(inventory_choice=inv)
                    if not is_custom_inventory(inv):
                        _apply_recommended_setup(inventory_key=inv)
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div class="aic-inv-card aic-inv-card--unavailable" title="Unavailable"
                         aria-disabled="true">
                      <span class="aic-inv-unavailable" title="Unavailable"
                            aria-label="Unavailable">⊘</span>
                      <div class="aic-inv-card-title">{inv}</div>
                      <div class="aic-inv-soon">Unavailable</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    inv_choice = _form_get("inventory_choice", "") or ""
    if inv_choice and not is_inventory_selectable(inv_choice):
        _form_set(inventory_choice="")
        inv_choice = ""

    custom_ok = True
    if is_custom_inventory(inv_choice):
        from inventory_profiles import parse_custom_item_specs

        st.markdown("#### Custom Item")
        st.caption(
            f"Enter one or more items (up to {MAX_PROMPTS}). "
            "Each listed item is detected as its **own separate type** "
            "(separate boxes and separate counts). "
            "Put each item on its own line, or separate them with commas."
        )
        item_name = st.text_area(
            "Items to detect (separate types)",
            value=_custom_item_name(),
            placeholder="traffic cone\nbarrel\npallet",
            max_chars=400,
            height=110,
            key="setup_custom_item_name",
        )
        with st.expander("Optional synonyms / alternate phrases", expanded=False):
            alternatives = st.text_area(
                "Synonyms (optional)",
                value=_custom_item_alternatives(),
                placeholder="traffic cone: road cone, safety cone\nbarrel: drum",
                max_chars=240,
                height=90,
                key="setup_custom_alts",
                help=(
                    "Use 'item: alias1, alias2' so synonyms help matching but still "
                    "count under that item type. Free terms with multiple items "
                    "become additional separate types."
                ),
            )
        _form_set(custom_item_name=item_name, custom_item_alternatives=alternatives)
        specs, prompt_errs = parse_custom_item_specs(item_name, alternatives)
        prompts, _ = effective_prompts_for_inventory(
            inv_choice,
            custom_item_name=item_name,
            custom_alternatives=alternatives,
        )
        hard_errs = [
            e
            for e in prompt_errs
            if "unscoped extra terms" not in e.lower()
            and "use 'item:" not in e.lower()
        ]
        if hard_errs and not specs:
            custom_ok = False
            for err in hard_errs:
                st.caption(err)
        elif specs:
            for note in prompt_errs:
                if note and note not in hard_errs:
                    st.caption(note)
            type_labels = [s.name for s in specs]
            st.caption(
                f"**{len(type_labels)} separate item type"
                f"{'' if len(type_labels) == 1 else 's'}** "
                f"(detected independently): {prompts_to_csv(type_labels)}"
            )
            if prompts and prompts != type_labels:
                st.caption(f"Model class list (includes synonyms): {prompts_to_csv(prompts)}")
            _apply_recommended_setup(inventory_key=inv_choice, apply_selection=True)
        else:
            custom_ok = False
            for err in prompt_errs:
                st.caption(err)

    st.markdown(
        f'<p class="aic-note">{PHOTO_RELATIONSHIP_NOTE}</p>',
        unsafe_allow_html=True,
    )

    yard_ok = bool(_resolved_yard())
    inv_ok = bool(inv_choice) and is_inventory_selectable(inv_choice) and custom_ok
    if not inv_choice:
        st.caption("Select an inventory type to continue.")
    elif is_custom_inventory(inv_choice) and not custom_ok:
        st.caption("Enter at least one custom item to continue.")

    def _next() -> None:
        if not _resolved_yard():
            st.error("Location is required.")
            return
        choice = _form_get("inventory_choice", "")
        if not is_inventory_selectable(choice):
            st.error("Select an inventory type to continue.")
            return
        if is_custom_inventory(choice):
            prompts, errs = effective_prompts_for_inventory(
                choice,
                custom_item_name=_custom_item_name(),
                custom_alternatives=_custom_item_alternatives(),
            )
            if errs or not prompts:
                st.error(errs[0] if errs else "Enter at least one custom item.")
                return
        _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)
        resolved = _apply_recommended_setup(inventory_key=choice)
        if not resolved.get("ok"):
            st.error(resolved.get("error") or "Could not resolve AI setup.")
            return
        navigate_to("wizard", stage="photos")

    render_nav_buttons(
        next_label="Continue to Add Photos",
        next_disabled=not (yard_ok and inv_ok),
        on_next=_next,
        key_prefix="stage_setup",
    )


# ---------------------------------------------------------------------------
# Stage 2 — Add Photos (upload + camera + samples)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _thumb_jpeg_bytes(raw: bytes, max_edge: int = 160, quality: int = 72) -> bytes:
    """Downscale for gallery display (keeps large libraries snappy)."""
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


@st.cache_data(show_spinner=False)
def _sample_thumb_from_path(path_str: str, mtime: float, max_edge: int = 160) -> bytes:
    raw = Path(path_str).read_bytes()
    return _thumb_jpeg_bytes(raw, max_edge=max_edge)


def _add_sample_by_id(sample_id: str) -> str | None:
    sample = get_sample_by_id(sample_id)
    if sample is None:
        return "Sample not found."
    try:
        data = read_sample_bytes(sample)
    except OSError as exc:
        return f"{sample.filename}: {exc}"
    return _add_image_bytes(
        data,
        sample.filename,
        source="sample",
        mime_type=sample.mime_type,
        sample_id=sample.id,
    )


def _clear_sample_selection_widget_keys() -> None:
    """Drop sample checkbox widget keys before widgets are created this run."""
    for key in list(st.session_state.keys()):
        if str(key).startswith("sample_sel_"):
            del st.session_state[key]


def _render_sample_images_tab() -> None:
    """Paginated sample gallery — click Add (no separate preview panel)."""
    st.caption(
        "Optional sample gallery. Use **Add** on a card, or select several then **Add selected**. "
        "Administrators can add samples in the Admin Console."
    )
    # Clear checkbox widget state before instantiation (never mutate after widgets exist).
    if st.session_state.pop("sample_clear_pending", False):
        st.session_state.sample_selected_ids = []
        _clear_sample_selection_widget_keys()

    inv_key = _resolved_inventory() or SELECTABLE_INVENTORY_KEY
    samples = list_enabled_samples(inventory_key=inv_key)
    lib = load_sample_library()
    if lib.warnings:
        st.caption(
            f"Built-in sample library notes: {len(lib.warnings)} warning(s). See Diagnostics."
        )
    st.caption(f"Showing samples for **{inv_key}**.")

    if not samples:
        render_empty_state(
            f"No sample images for {inv_key}",
            "Upload samples for this inventory type in Admin Console → Sample Images, "
            "or add files under assets/sample_images/ and register them in manifest.json. "
            "Samples are filtered by the inventory type chosen in Inventory Setup.",
        )
        return

    # Drop legacy preview state so it cannot expand the page.
    st.session_state.pop("sample_preview_id", None)

    page_size = 6
    total = len(samples)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = int(st.session_state.get("sample_gallery_page") or 0)
    page = max(0, min(page, total_pages - 1))
    st.session_state.sample_gallery_page = page

    nav_l, nav_m, nav_r = st.columns([1, 2.2, 1])
    with nav_l:
        if st.button("← Prev", width="stretch", disabled=page <= 0, key="sample_page_prev"):
            st.session_state.sample_gallery_page = page - 1
            st.rerun()
    with nav_m:
        st.caption(f"Samples {page * page_size + 1}–{min(total, (page + 1) * page_size)} of {total}")
    with nav_r:
        if st.button(
            "Next →",
            width="stretch",
            disabled=page >= total_pages - 1,
            key="sample_page_next",
        ):
            st.session_state.sample_gallery_page = page + 1
            st.rerun()

    page_samples = samples[page * page_size : (page + 1) * page_size]
    selected_ids: set[str] = set(st.session_state.get("sample_selected_ids") or [])

    cols = st.columns(3)
    for i, sample in enumerate(page_samples):
        with cols[i % 3]:
            st.markdown('<div class="aic-sample-card">', unsafe_allow_html=True)
            try:
                if sample.path is not None:
                    thumb = _sample_thumb_from_path(
                        str(sample.path), sample.path.stat().st_mtime, max_edge=160
                    )
                else:
                    thumb = _thumb_jpeg_bytes(read_sample_bytes(sample), max_edge=160)
                st.image(thumb, width="stretch")
            except OSError:
                st.warning(sample.title)
                st.markdown("</div>", unsafe_allow_html=True)
                continue
            st.markdown(f"**{sample.title}**")
            st.caption(f"{sample.width}×{sample.height}")
            checked = st.checkbox(
                "Select",
                value=sample.id in selected_ids,
                key=f"sample_sel_{sample.id}",
            )
            if checked:
                selected_ids.add(sample.id)
            else:
                selected_ids.discard(sample.id)
            if st.button("Add", key=f"sample_add_{sample.id}", width="stretch"):
                err = _add_sample_by_id(sample.id)
                if err:
                    st.warning(err)
                else:
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.sample_selected_ids = list(selected_ids)

    a1, a2 = st.columns(2)
    with a1:
        if st.button(
            f"Add selected ({len(selected_ids)})",
            type="primary",
            width="stretch",
            key="sample_add_selected",
            disabled=not selected_ids,
        ):
            added = 0
            for sid in list(selected_ids):
                err = _add_sample_by_id(sid)
                if err:
                    st.warning(err)
                else:
                    added += 1
            if added:
                st.session_state.sample_selected_ids = []
                st.session_state.sample_clear_pending = True
                st.rerun()
    with a2:
        if st.button("Clear selection", width="stretch", key="sample_clear_sel"):
            st.session_state.sample_selected_ids = []
            st.session_state.sample_clear_pending = True
            st.rerun()


def _add_image_bytes(
    data: bytes,
    name: str,
    *,
    source: str = "upload",
    mime_type: str = "image/jpeg",
    sample_id: str | None = None,
) -> str | None:
    """Validate and append an image; return error text, or None on success."""
    try:
        validate_upload(data, name, config.MAX_UPLOAD_BYTES)
        meta = _image_meta(
            name, data, source=source, mime_type=mime_type, sample_id=sample_id
        )
    except ImageProcessingError as exc:
        return f"{name}: {exc}"
    except Exception:  # noqa: BLE001
        return f"{name}: could not read image."
    existing_ids = {img["id"] for img in st.session_state.uploaded_images}
    if meta["id"] in existing_ids:
        return "This image is already included."
    st.session_state.uploaded_images.append(meta)
    return None


def _render_detection_prompt_picker(key_prefix: str = "photos") -> str:
    """Ask what to detect; stores the value on the shared form for Analyze."""
    inventory_type = _resolved_inventory() or "Fence Panel"
    if inventory_type == "Fence Panel":
        presets = list(FENCE_PANEL_PROMPT_PRESETS)
    else:
        default_p = DEFAULT_PROMPTS.get(inventory_type, inventory_type)
        presets = [default_p] + [
            h for h in getattr(config, "DETECTION_PROMPT_HINTS", []) if h != default_p
        ]

    st.markdown("#### What do you want to detect?")
    st.caption(
        "Type what the AI should look for (for example: wood fence, fence panel, fence post). "
        "Separate multiple classes with commas."
    )

    cur_preset = _form_get("prompt_preset", presets[0])
    if cur_preset not in presets:
        presets = [cur_preset] + presets

    preset_key = f"{key_prefix}_detect_preset"
    prompt_key = f"{key_prefix}_detect_prompt"
    last_preset_key = f"_{key_prefix}_last_detect_preset"

    preset = st.selectbox(
        "Quick suggestions",
        presets,
        index=presets.index(cur_preset) if cur_preset in presets else 0,
        key=preset_key,
    )

    if prompt_key not in st.session_state:
        st.session_state[prompt_key] = (_form_get("prompt") or preset).strip() or preset
        st.session_state[last_preset_key] = preset
    elif st.session_state.get(last_preset_key) != preset:
        # Changing a suggestion refreshes the text box; custom typing is kept until then.
        st.session_state[prompt_key] = preset
        st.session_state[last_preset_key] = preset

    prompt = st.text_input(
        "Detect these objects",
        placeholder="e.g. wood fence, fence panel",
        key=prompt_key,
    )
    prompt = (prompt or "").strip()
    _form_set(prompt_preset=preset, prompt=prompt, class_override=prompt)
    if prompt:
        st.caption(f"Will detect: **{prompt}**")
    else:
        st.warning("Enter what you want to detect before continuing.")
    return prompt


@st.dialog("Photo preview")
def _enlarge_photo_dialog(img: dict[str, Any]) -> None:
    st.image(img["data"], width="stretch")
    st.caption(
        f"{img.get('name', 'photo')} · {img.get('width')}×{img.get('height')} px · "
        f"{_format_bytes(int(img.get('size_bytes') or 0))} · {img.get('source') or 'upload'}"
    )


def _render_selected_photos_strip(*, nonce: int) -> None:
    """Compact paginated thumbnail strip (enlarge on demand)."""
    images = list(st.session_state.uploaded_images or [])
    st.markdown('<div class="aic-photo-strip">', unsafe_allow_html=True)
    if not images:
        st.markdown(
            '<p class="aic-photo-strip-title">No photos added yet</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Upload an image, use the camera, or choose a sample image."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    page_size = 4
    total = len(images)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = int(st.session_state.get("selected_photos_page") or 0)
    page = max(0, min(page, total_pages - 1))
    st.session_state.selected_photos_page = page
    slice_imgs = images[page * page_size : (page + 1) * page_size]

    top_l, top_m, top_r = st.columns([2.2, 1.4, 1])
    with top_l:
        st.markdown(
            f'<p class="aic-photo-strip-title">{total} photo(s) ready</p>',
            unsafe_allow_html=True,
        )
    with top_m:
        if total_pages > 1:
            st.caption(f"Page {page + 1} / {total_pages}")
    with top_r:
        if st.button("Clear all", width="stretch", key="clear_photos"):
            st.session_state.uploaded_images = []
            st.session_state.pending_camera = None
            st.session_state.selected_photos_page = 0
            st.session_state.uploader_nonce = int(nonce) + 1
            st.rerun()

    if total_pages > 1:
        p1, p2 = st.columns(2)
        with p1:
            if st.button("←", width="stretch", disabled=page <= 0, key="sel_photos_prev"):
                st.session_state.selected_photos_page = page - 1
                st.rerun()
        with p2:
            if st.button(
                "→",
                width="stretch",
                disabled=page >= total_pages - 1,
                key="sel_photos_next",
            ):
                st.session_state.selected_photos_page = page + 1
                st.rerun()

    cols = st.columns(len(slice_imgs) or 1)
    for i, img in enumerate(slice_imgs):
        with cols[i]:
            try:
                thumb = _thumb_jpeg_bytes(img["data"], max_edge=140)
                st.image(thumb, width="stretch")
            except Exception:  # noqa: BLE001
                st.image(img["data"], width="stretch")
            st.caption(img.get("name", "photo")[:28])
            b1, b2 = st.columns(2)
            with b1:
                if st.button("View", key=f"view_img_{img['id']}", width="stretch"):
                    _enlarge_photo_dialog(img)
            with b2:
                if st.button("Remove", key=f"rm_img_{img['id']}", width="stretch"):
                    st.session_state.uploaded_images = [
                        x for x in st.session_state.uploaded_images if x["id"] != img["id"]
                    ]
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def stage_photos() -> None:
    render_stepper("photos")
    _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)

    inv = _resolved_inventory()
    inv_label = (
        inventory_display_name(inv, custom_item_name=_custom_item_name())
        if inv
        else "(not selected)"
    )
    n_photos = len(st.session_state.uploaded_images)
    status_txt = "Ready" if n_photos >= 1 else "Selected"
    prompts, _ = effective_prompts_for_inventory(
        inv,
        custom_item_name=_custom_item_name() if is_custom_inventory(inv) else None,
        custom_alternatives=_custom_item_alternatives()
        if is_custom_inventory(inv)
        else None,
    )

    render_stage_header(
        "Add Photos",
        "Upload, capture, or pick a sample — keep this step compact.",
    )
    st.markdown(
        f"""
        <div class="aic-chip-grid">
          <div class="aic-chip">
            <span class="aic-chip-label">Inventory</span>
            <span class="aic-chip-value">{inv_label}</span>
          </div>
          <div class="aic-chip">
            <span class="aic-chip-label">Status</span>
            <span class="aic-chip-value">{status_txt}</span>
          </div>
          <div class="aic-chip">
            <span class="aic-chip-label">Prompts</span>
            <span class="aic-chip-value">{len(prompts) if prompts else "—"}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if prompts:
        with st.expander("View Detection Terms", expanded=False):
            st.caption(prompts_to_csv(prompts))

    nonce = st.session_state.get("uploader_nonce", 0)
    # Single active source (tabs render every panel and get laggy with samples).
    source = st.segmented_control(
        "Photo source",
        options=["Upload Images", "Use Camera", "Sample Images"],
        default=st.session_state.get("photo_source_mode") or "Upload Images",
        key="photo_source_seg",
        label_visibility="collapsed",
    )
    if source:
        st.session_state.photo_source_mode = source
    source = st.session_state.get("photo_source_mode") or "Upload Images"

    max_mb = max(1, int(config.MAX_UPLOAD_BYTES / (1024 * 1024)))
    if source == "Upload Images":
        st.caption(
            f"JPG / PNG · multiple files · max {max_mb} MB each. "
            "Thumbnails stay small — use **View** to enlarge."
        )
        new_files = st.file_uploader(
            "Drag and drop inventory photos here",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"photo_uploader_{nonce}",
        )
        if new_files:
            for up in new_files:
                up.seek(0)
                err = _add_image_bytes(up.getvalue(), up.name)
                if err:
                    st.error(err)

        if config.DEMO_MODE and not st.session_state.uploaded_images:
            if st.button("Add built-in demo image", key="add_demo_img"):
                data = _make_demo_image()
                err = _add_image_bytes(data, "demo_yard.jpg")
                if err:
                    st.error(err)
                else:
                    st.rerun()

    elif source == "Use Camera":
        st.caption("Compact camera capture. Add the still, or retake.")
        cam_l, cam_r = st.columns([1.35, 1], vertical_alignment="top")
        with cam_l:
            shot = st.camera_input("Capture inventory photo", key=f"photo_camera_{nonce}")
            if shot is not None:
                shot.seek(0)
                data = bytes(shot.getvalue())
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
                name = f"camera_{stamp}.jpg"
                try:
                    validate_upload(data, name, config.MAX_UPLOAD_BYTES)
                    pending = _image_meta(
                        name, data, source="camera", mime_type="image/jpeg"
                    )
                    st.session_state.pending_camera = pending
                except ImageProcessingError as exc:
                    st.error(f"Camera image invalid: {exc}")
                    st.session_state.pending_camera = None
        with cam_r:
            pending = st.session_state.get("pending_camera")
            if isinstance(pending, dict) and pending.get("data"):
                st.markdown("**Ready to add**")
                try:
                    st.image(
                        _thumb_jpeg_bytes(pending["data"], max_edge=220),
                        width="stretch",
                    )
                except Exception:  # noqa: BLE001
                    st.image(pending["data"], width="stretch")
                st.caption(
                    f"{pending['width']}×{pending['height']} · "
                    f"{_format_bytes(pending['size_bytes'])}"
                )
                if st.button(
                    "Add This Photo",
                    type="primary",
                    width="stretch",
                    key="cam_add",
                ):
                    err = _add_image_bytes(
                        pending["data"],
                        pending["name"],
                        source="camera",
                        mime_type="image/jpeg",
                    )
                    st.session_state.pending_camera = None
                    st.session_state.uploader_nonce = int(nonce) + 1
                    if err:
                        st.error(err)
                    st.rerun()
                if st.button("Retake / Discard", width="stretch", key="cam_retake"):
                    st.session_state.pending_camera = None
                    st.session_state.uploader_nonce = int(nonce) + 1
                    st.rerun()
            else:
                st.caption("Capture appears here as a small preview.")

    else:
        _render_sample_images_tab()

    _render_selected_photos_strip(nonce=int(nonce))

    can_next = len(st.session_state.uploaded_images) >= 1 and is_inventory_selectable(
        _form_get("inventory_choice", "")
    )

    def _next() -> None:
        if not st.session_state.uploaded_images:
            st.error("Add at least one valid image.")
            return
        choice = _form_get("inventory_choice", "")
        if not is_inventory_selectable(choice):
            st.error("Select an inventory type on Inventory Setup before analyzing.")
            return
        if is_custom_inventory(choice):
            prompts, errs = effective_prompts_for_inventory(
                choice,
                custom_item_name=_custom_item_name(),
                custom_alternatives=_custom_item_alternatives(),
            )
            if errs or not prompts:
                st.error(errs[0] if errs else "Custom item name is required.")
                return
        resolved = _apply_recommended_setup(inventory_key=choice)
        if not resolved.get("ok"):
            st.error(
                resolved.get("error")
                or "No valid AI model is configured for this inventory."
            )
            return
        if st.session_state.analysis_status in {"complete", "partial"}:
            st.session_state.analysis_status = "idle"
            st.session_state.analysis_results = []
            st.session_state.analysis_failures = []
            st.session_state.save_status = "idle"
            st.session_state.saved_record = None
        _ensure_selected_models()
        navigate_to("wizard", stage="analyze")

    render_nav_buttons(
        back_stage="setup",
        next_label="Continue to Analyze",
        next_disabled=not can_next,
        on_next=_next,
        key_prefix="stage_photos",
    )


# ---------------------------------------------------------------------------
# Stage 4 — Run Analysis
# ---------------------------------------------------------------------------


def _ai_config_is_valid() -> bool:
    """True when analysis can run with current local configuration."""
    if config.DEMO_MODE:
        return True
    selected = _ensure_selected_models()
    if not selected:
        return False
    enabled = {m.name: m for m in _enabled_models()}
    selected_models = [enabled[n] for n in selected if n in enabled]
    needs_api = any((m.kind or "").lower() != "local" for m in selected_models)
    if needs_api and not api_key_configured():
        return False
    if not needs_api and selected_models:
        return True
    model = _primary_workflow_model()
    return model is not None


def _render_analysis_failure_state(results: list[InferenceResult], failures: list[str]) -> None:
    from poc_ux import classify_user_error, sanitize_public_text

    first_msg = ""
    first_type = None
    for fail in failures:
        first_msg = str(fail)
        break
    for r in results:
        if r.error_message or r.errors:
            first_msg = r.error_message or "; ".join(r.errors)
            first_type = r.error_type
            break
    err = classify_user_error(
        error_type=first_type,
        message=first_msg,
        api_configured=bool(api_key_configured() or config.DEMO_MODE),
        dynamic_prompt_failed="dynamic" in (first_msg or "").lower()
        or "class_names" in (first_msg or "").lower(),
    )
    # Prefer OpenRouter-specific guidance over Roboflow Workflow wording.
    display_message = err.message
    if "openrouter" in (first_msg or "").lower() or "could not be parsed into a valid inventory count" in (
        first_msg or ""
    ).lower():
        display_message = (
            "OpenRouter returned a response, but it could not be parsed into a "
            "valid inventory count."
        )
    render_empty_state(err.title, display_message)
    with st.expander("Technical Details", expanded=False):
        for fail in failures[:8]:
            st.caption(sanitize_public_text(fail))
        tech_rows = list(st.session_state.get("analysis_technical_details") or [])
        for row in tech_rows[-5:]:
            st.json(
                {
                    k: v
                    for k, v in row.items()
                    if "api_key" not in str(k).lower() and "secret" not in str(k).lower()
                }
            )
        for r in results:
            st.json(
                {
                    k: v
                    for k, v in r.summary_dict().items()
                    if "api_key" not in str(k).lower()
                }
            )
    if st.button("Open AI Configuration", key="af_settings"):
        open_settings(section="ai_configuration")


def _render_zero_detection_empty(results: list[InferenceResult]) -> None:
    render_empty_state(
        "Analysis completed successfully, but no matching objects were found.",
        "Try adjusting the detection terms or confidence threshold. "
        "This is not the same as an inference failure.",
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Try Another Photo", width="stretch", key="zd_photo"):
            navigate_to("wizard", stage="photos")
    with a2:
        if st.button("Adjust Detection Sensitivity", width="stretch", key="zd_sens"):
            st.session_state.open_advanced_settings = True
            open_settings(section="ai_configuration")
    with a3:
        if st.button("Continue to Review", type="primary", width="stretch", key="zd_rev"):
            navigate_to("wizard", stage="review")

    with st.expander("View Technical Details", expanded=False):
        for r in results:
            st.markdown(f"**{r.image_name}** · {r.model_name}")
            st.write(f"Raw API prediction count: {r.raw_prediction_count}")
            st.write(f"Normalized prediction count: {r.normalized_prediction_count}")
            st.write(f"Final count: {r.final_count}")
            classes = sorted({d.class_name for d in r.detections}) or ["(none)"]
            st.write(f"Detected classes: {', '.join(classes)}")
            st.write(f"Response source: {r.source or ('demo' if config.DEMO_MODE else 'live Roboflow')}")
            st.write(f"Invocation mode: {r.invocation_mode or '(n/a)'}")
            st.write(f"Error type: {r.error_type or '(none)'}")
            st.write(f"Processing time: {r.processing_time_seconds:.2f}s")
            if r.warnings:
                st.write("Warnings:")
                for w in r.warnings:
                    st.caption(f"• {w}")
            st.json(r.summary_dict())


def stage_analyze() -> None:
    render_stepper("analyze")
    render_stage_header(
        "Analyze",
        "Choose a model (or compare), then run detection on your photos.",
    )
    st.caption(
        "For complete individual counts (stacked, scattered, overlapping, or odd angles), "
        "prefer **OpenRouter VLM Detector**. YOLO-World is faster but often treats a "
        "stack or group as one object. Always review before saving."
    )

    images = st.session_state.uploaded_images
    inference_ui = _form_get("inference_mode", "Whole Image")
    inference_mode = _inference_api_name(inference_ui)
    config_ok = _ai_config_is_valid()
    inv_key = _resolved_inventory()
    inv_label = (
        inventory_display_name(inv_key, custom_item_name=_custom_item_name()) or "—"
    )
    ai_label = "Connected" if config_ok else "Needs attention"

    # Defaults / prompts only — do NOT overwrite the user's model selection each rerun.
    if inv_key:
        _apply_recommended_setup(
            inventory_key=inv_key,
            apply_selection=not bool(_form_get("selected_models") or []),
        )

    from catalog_ui import format_model_info_markdown, format_model_option
    from model_catalog import get_all_catalog_models, remove_stale_model_selection

    selectable, blocked = _analysis_models_with_blocked()
    model_names = [m.name for m in selectable]
    _render_blocked_models_notice(blocked)
    # Compare peers: enabled/valid Roboflow + confirmed local inference (not demo fixtures).
    compare_models = compare_peer_models(selectable)
    compare_names = [m.name for m in compare_models]
    compare_available = len(compare_names) >= COMPARE_MIN_MODELS

    cleaned, stale_note = remove_stale_model_selection(
        _form_get("selected_models") or [],
        inventory_key=inv_key or SELECTABLE_INVENTORY_KEY,
    )
    if stale_note:
        st.info(stale_note)
    if cleaned != (_form_get("selected_models") or []):
        _form_set(selected_models=cleaned)

    if not model_names:
        st.error("No compatible validated model is available.")
        if st.button("Open AI Configuration", key="analyze_missing_model_settings"):
            open_settings(section="ai_configuration")
        render_nav_buttons(back_stage="photos", key_prefix="an_nomodel")
        return

    mode_opts = ["Single Model", "Compare Models"]
    cur_mode = _form_get("selected_mode", "Single Model")
    if cur_mode not in mode_opts:
        cur_mode = "Single Model"
    if not compare_available and cur_mode == "Compare Models":
        cur_mode = "Single Model"
        _form_set(selected_mode=cur_mode)

    if not compare_available:
        if len(compare_names) == 1:
            st.info(
                "Only one compatible validated model is currently available. "
                "Add and validate another object-detection model to use comparison."
            )
        else:
            st.caption(
                "At least two configured and validated models are required for comparison."
            )
        mode_ui = "Single Model"
        _form_set(selected_mode=mode_ui)
        st.radio(
            "Analysis Mode",
            mode_opts,
            index=0,
            horizontal=True,
            key="analyze_mode_radio_locked",
            disabled=True,
        )
    else:
        mode_ui = st.radio(
            "Analysis Mode",
            mode_opts,
            index=mode_opts.index(cur_mode),
            horizontal=True,
            key="analyze_mode_radio",
        )
        _form_set(selected_mode=mode_ui)

    prev = sanitize_selected_model_names(
        cleaned or _form_get("selected_models") or [], model_names
    )
    if mode_ui == "Single Model" and not prev:
        prev = _ensure_selected_models() or model_names[:1]

    entries_by_name = {e.display_name: e for e in get_all_catalog_models()}
    selectable_by_name = {m.name: m for m in selectable}

    st.markdown("#### Choose model")
    st.caption("Select a model for this run. Use **Info** for source, purpose, and import path.")

    if mode_ui == "Single Model":
        single_prev = prev[0] if prev and prev[0] in model_names else model_names[0]
        # Radio keeps a stable widget selection (buttons were reset by recommended setup).
        choice = st.radio(
            "Model",
            options=model_names,
            index=model_names.index(single_prev),
            format_func=lambda n: format_model_option(
                selectable_by_name[n], entries_by_name
            ),
            key="analyze_single_model_radio",
        )
        selected_names = [choice] if choice in model_names else [model_names[0]]
        _form_set(selected_models=selected_names)
        info_model = selectable_by_name.get(selected_names[0])
        info_entry = entries_by_name.get(selected_names[0])
        with st.popover("Model info"):
            if info_model is not None:
                st.markdown(format_model_info_markdown(info_model, info_entry))
        st.caption(f"Selected: **{selected_names[0]}** · Detecting: **{inv_label}**")
    else:
        compare_prev = sanitize_compare_selection(prev, compare_names)
        selected_set = set(compare_prev)
        for name in compare_names:
            m = next(x for x in compare_models if x.name == name)
            entry = entries_by_name.get(name)
            checked = name in selected_set
            card_cls = "aic-model-pick aic-model-pick-selected" if checked else "aic-model-pick"
            st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
            row_l, row_m, row_r = st.columns([0.7, 3.5, 1], vertical_alignment="center")
            with row_l:
                on = st.checkbox(
                    "Use",
                    value=checked,
                    key=f"analyze_cmp_{name}",
                    label_visibility="collapsed",
                )
                if on:
                    selected_set.add(name)
                else:
                    selected_set.discard(name)
            with row_m:
                st.markdown(f"**{name}**")
                st.caption(format_model_option(m, entries_by_name))
            with row_r:
                with st.popover("Info"):
                    st.markdown(format_model_info_markdown(m, entry))
            st.markdown("</div>", unsafe_allow_html=True)

        # Preserve prior order, then append newly checked names.
        ordered = [n for n in compare_prev if n in selected_set]
        ordered.extend(n for n in compare_names if n in selected_set and n not in ordered)
        selected_names = sanitize_compare_selection(ordered, compare_names)
        if len(selected_set) > COMPARE_MAX_MODELS:
            st.warning(f"Select at most {COMPARE_MAX_MODELS} models for comparison.")
        cmp_errs = validate_compare_selection(selected_names, compare_names)
        if cmp_errs:
            st.caption(cmp_errs[0])

    _form_set(selected_models=selected_names)
    selected_models = [m for m in selectable if m.name in selected_names]
    # Preserve selection order from the multiselect / selectbox.
    if mode_ui == "Compare Models":
        order = {n: i for i, n in enumerate(selected_names)}
        selected_models.sort(key=lambda m: order.get(m.name, 999))

    run_ctx, prompt_errs = _build_current_run_context(
        selected_models=selected_models, images=list(images)
    )
    detect_prompts = list(run_ctx.effective_prompts) if run_ctx else []
    detect_prompt = prompts_to_csv(detect_prompts) if detect_prompts else ""
    if not detect_prompt:
        detect_prompt = config.inventory_detection_prompt(inv_key)
        detect_prompts, _ = effective_prompts_for_inventory(inv_key)
    _form_set(
        prompt=detect_prompt,
        class_override=detect_prompt,
        effective_prompts=detect_prompts,
        counting_unit=(run_ctx.counting_unit if run_ctx else ""),
    )

    model_label = (
        selected_models[0].name
        if len(selected_models) == 1
        else f"{len(selected_models)} models"
    )
    st.markdown(
        f"""
        <div class="aic-chip-grid">
          <div class="aic-chip">
            <span class="aic-chip-label">Inventory</span>
            <span class="aic-chip-value">{inv_label}</span>
          </div>
          <div class="aic-chip">
            <span class="aic-chip-label">Photos</span>
            <span class="aic-chip-value">{len(images)}</span>
          </div>
          <div class="aic-chip">
            <span class="aic-chip-label">Model</span>
            <span class="aic-chip-value">{model_label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Detecting: **{inv_label}**")
    if detect_prompts:
        st.caption("Detection terms: " + prompts_to_csv(detect_prompts))
    if prompt_errs:
        for err in prompt_errs:
            st.warning(err)
    if mode_ui == "Compare Models":
        st.caption(comparison_run_caption(len(images), len(selected_models)))
    st.caption(f"AI: {ai_label}")

    status = st.session_state.analysis_status
    if status in {"complete", "partial", "running", "error"}:
        st.info("Open the Running step for progress and results.")
        if st.button(
            "Go to Running / Results",
            type="primary",
            width="stretch",
            key="an_goto_running",
        ):
            navigate_to("wizard", stage="running")
        render_nav_buttons(back_stage="photos", key_prefix="an_done")
        return

    running = bool(st.session_state.analyze_running)
    needs_api = any((m.kind or "").lower() != "local" for m in selected_models)
    compare_ready = mode_ui != "Compare Models" or (
        compare_available
        and COMPARE_MIN_MODELS <= len(selected_models) <= COMPARE_MAX_MODELS
        and not validate_compare_selection(selected_names, compare_names)
    )
    run_label = "Run Comparison" if mode_ui == "Compare Models" else "Run Analysis"
    run = st.button(
        run_label,
        type="primary",
        width="stretch",
        disabled=running
        or not images
        or not selected_models
        or not config_ok
        or not compare_ready
        or (needs_api and not config.DEMO_MODE and not api_key_configured()),
        key="run_analysis_btn",
    )
    render_nav_buttons(back_stage="photos", key_prefix="an_idle")

    if not run:
        return

    if needs_api and not config.DEMO_MODE and not api_key_configured():
        _error_box(
            "Live analysis disabled: ROBOFLOW_API_KEY is missing.",
            "Set ROBOFLOW_API_KEY or enable DEMO_MODE=true, or select only local models.",
        )
        return

    # Persist canonical run context before navigating to Running.
    run_ctx, ctx_errs = _build_current_run_context(
        selected_models=selected_models, images=list(images)
    )
    if run_ctx is None:
        _error_box(
            "Cannot start analysis.",
            (ctx_errs[0] if ctx_errs else "Invalid detection prompts."),
        )
        return
    st.session_state.run_context = run_ctx.to_dict()
    _form_set(
        prompt=run_ctx.prompt_csv(),
        class_override=run_ctx.prompt_csv(),
        effective_prompts=list(run_ctx.effective_prompts),
        counting_unit=run_ctx.counting_unit,
        confidence_threshold=run_ctx.confidence_threshold,
    )

    # Dedicated Running page — do not expand Analyze with live progress.
    import uuid

    st.session_state.analyze_running = True
    st.session_state.analysis_status = "running"
    st.session_state._analysis_executing = False
    st.session_state.analysis_run_id = str(uuid.uuid4())
    navigate_to("wizard", stage="running")


def _execute_analysis_run(
    *,
    images: list[dict[str, Any]],
    selected_models: list[ModelConfig],
    mode_ui: str,
    inference_mode: str,
) -> None:
    """Run inference pipeline on the Running page (progress UI lives here)."""
    from poc_ux import (
        compare_progress_caption,
        progress_phase_label,
        sanitize_public_text,
    )

    detector = RoboflowDetector()
    preview_slot = st.empty()
    progress = st.progress(0.0, text=progress_phase_label(0))
    status_box = st.empty()
    phase_box = st.empty()
    results: list[InferenceResult] = []
    failures: list[str] = []
    comparison_summaries: list[dict[str, Any]] = []
    compare_successes = 0
    st.session_state.analysis_technical_details = []
    run_id = st.session_state.get("analysis_run_id")

    run_ctx = AnalysisRunContext.from_dict(st.session_state.get("run_context"))
    if run_ctx is None:
        run_ctx, _ = _build_current_run_context(
            selected_models=selected_models, images=list(images)
        )
    if run_ctx is not None:
        st.session_state.run_context = run_ctx.to_dict()

    prompt = (
        run_ctx.prompt_csv()
        if run_ctx and run_ctx.effective_prompts
        else _form_get("prompt", "")
    )
    conf = float(
        (run_ctx.confidence_threshold if run_ctx else None)
        or _form_get("confidence_threshold", 0.25)
    )
    iou = float(_form_get("iou_threshold", 0.5))
    tile_size = int(_form_get("tile_size", 800))
    tile_overlap = float(_form_get("tile_overlap", 0.25))
    dedup = _form_get("deduplication_strategy", "Conservative")
    total = max(1, len(images) * len(selected_models))
    step_i = 0

    def _show_analysis_preview(item: dict[str, Any], img_i: int, model_i: int, model_name: str) -> None:
        with preview_slot.container():
            st.markdown('<div class="aic-img-card">', unsafe_allow_html=True)
            st.image(item["data"], width="stretch", output_format="JPEG")
            st.markdown("</div>", unsafe_allow_html=True)
            if len(selected_models) > 1:
                st.caption(progress_label(model_i, len(selected_models), img_i, len(images)))
                st.caption(f"Running: {model_name}")
            else:
                st.caption(f"Analyzing image {img_i} of {len(images)} with {model_name}…")

    try:
        for img_i, item in enumerate(images, start=1):
            # Same confirmed source bytes for every selected model on this image.
            source_bytes = item["data"]
            try:
                prepared = load_image_from_bytes(source_bytes, item["name"])
            except ImageProcessingError as exc:
                failures.append(f"{item['name']}: {exc}")
                for model in selected_models:
                    comparison_summaries.append(
                        {
                            "model": model.name,
                            "model_key": model_key(model),
                            "status": "Configuration failure",
                            "raw_count": None,
                            "final_count": None,
                            "avg_confidence": None,
                            "max_confidence": None,
                            "classes": "—",
                            "processing_time": 0.0,
                            "warning_count": 0,
                            "warnings": "",
                            "source": "error",
                            "error": str(exc),
                            "error_type": "configuration",
                            "image": item["name"],
                            "cached": False,
                            "success": False,
                        }
                    )
                continue

            for model_i, model in enumerate(selected_models, start=1):
                step_i += 1
                _show_analysis_preview(item, img_i, model_i, model.name)
                phase_box.caption(
                    f"{progress_phase_label(0)} → {progress_phase_label(1)} → "
                    f"{progress_phase_label(2)}"
                )
                if len(selected_models) > 1:
                    prog_txt = compare_progress_caption(
                        current_model=model.name,
                        model_index=model_i,
                        total_models=len(selected_models),
                        completed=max(0, step_i - 1),
                        failures=len(failures),
                        successes=compare_successes,
                    )
                else:
                    prog_txt = (
                        f"{progress_phase_label(2)}: {model.name} · "
                        f"image {img_i} of {len(images)}"
                    )
                # Indeterminate-style progress: step fraction only (no fake precision).
                progress.progress(min(0.95, step_i / total), text=prog_txt)
                status_box.caption(prog_txt)

                # Each model runs independently with its own thresholds / prompt rules.
                model_conf = conf
                model_iou = iou
                if len(selected_models) > 1:
                    if model.default_confidence is not None:
                        model_conf = float(model.default_confidence)
                    if model.default_iou is not None:
                        model_iou = float(model.default_iou)
                model_prompt = prompt or ""
                if not (model.supports_prompt or model.dynamic_classes):
                    model_prompt = ""
                options = InferenceOptions(
                    prompt=model_prompt,
                    confidence_threshold=model_conf,
                    iou_threshold=model_iou,
                    inference_mode=inference_mode,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    deduplication_strategy=dedup,
                )
                key = _cache_key(
                    prepared.content_hash,
                    model.name,
                    model_prompt,
                    model_conf,
                    inference_mode,
                    tile_size,
                    tile_overlap,
                    dedup,
                    model_iou,
                )
                cached = st.session_state.inference_cache.get(key)
                if cached is not None:
                    cached = _canonicalize_result_classes(cached, run_ctx)
                    results.append(cached)
                    comparison_summaries.append(
                        summary_row_from_cached(cached, model_key=model_key(model))
                    )
                    continue

                phase_box.caption(progress_phase_label(2))
                # OpenRouter models use the shared secure inference-key accessor.
                from openrouter import is_openrouter_model
                from openrouter_runtime import get_openrouter_inference_key

                needs_or_key = is_openrouter_model(model) or getattr(
                    model, "requires_user_api_key", False
                )
                deployment_key = get_openrouter_inference_key() if needs_or_key else ""
                adapter = get_adapter(
                    model,
                    detector=None if deployment_key else detector,
                    model_api_key=deployment_key,
                )
                mir = adapter.predict(prepared, options)
                _record_model_run(model)
                phase_box.caption(progress_phase_label(3))
                comparison_summaries.append(
                    summary_row_from_mir(mir, image_name=prepared.image_name)
                )
                if mir.success and mir.inference_result is not None:
                    ir = _canonicalize_result_classes(mir.inference_result, run_ctx)
                    st.session_state.inference_cache[key] = ir
                    results.append(ir)
                    compare_successes += 1
                else:
                    # Do not convert failures into zero-detection InferenceResults
                    fail_msg = sanitize_public_text(
                        f"{prepared.image_name} / {model.name}: "
                        f"{mir.error_message or mir.error_type or 'failed'}"
                    )
                    failures.append(fail_msg)
                    tech = dict(mir.technical_details or {})
                    if tech:
                        st.session_state.setdefault("analysis_technical_details", [])
                        st.session_state.analysis_technical_details.append(
                            {
                                "model": model.name,
                                "image": prepared.image_name,
                                "error_type": mir.error_type,
                                "message": fail_msg,
                                "selected_model": tech.get("selected_model"),
                                "http_status": tech.get("http_status"),
                                "response_type": tech.get("response_type"),
                                "parser_stage": tech.get("parser_stage"),
                                "retryable": tech.get("retryable"),
                                "response_preview": sanitize_public_text(
                                    str(tech.get("response_preview") or ""),
                                    max_len=400,
                                ),
                            }
                        )

        phase_box.caption(progress_phase_label(4))
        st.session_state.analysis_results = results
        st.session_state.analysis_failures = failures
        st.session_state.comparison_summaries = comparison_summaries
        if not failures:
            st.session_state.analysis_technical_details = []
        st.session_state.analysis_run_id = run_id
        st.session_state.review_edits = {
            "excluded_ids": [],
            "manual_detections": [],
            "class_overrides": {},
        }
        selected_keys = [model_key(m) for m in selected_models]
        selected_display = [m.name for m in selected_models]
        returned_classes: list[str] = sorted(
            {
                d.class_name
                for r in results
                for d in (r.detections or [])
                if d.class_name
            }
        )
        st.session_state.analysis_meta = {
            "yard": _resolved_yard(),
            "inventory_type": (
                run_ctx.inventory_key if run_ctx else _resolved_inventory()
            ),
            "inventory_display_name": (
                run_ctx.inventory_display_name
                if run_ctx
                else inventory_display_name(_resolved_inventory())
            ),
            "custom_item_name": run_ctx.custom_item_name if run_ctx else None,
            "primary_item_types": list(run_ctx.primary_item_types or [])
            if run_ctx
            else [],
            "class_alias_map": dict(run_ctx.class_alias_map or {}) if run_ctx else {},
            "counting_unit": (
                run_ctx.counting_unit
                if run_ctx
                else counting_unit_for(_resolved_inventory())
            ),
            "effective_prompts": (
                list(run_ctx.effective_prompts) if run_ctx else list(
                    _form_get("effective_prompts") or []
                )
            ),
            "photo_relationship": _form_get("photo_relationship"),
            "number_of_photos": len(images),
            "selected_mode": mode_ui,
            "selected_prompt": prompt,
            "inference_mode": inference_mode,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "deduplication_strategy": dedup,
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "comparison_mode": mode_ui == "Compare Models",
            "selected_model_keys": selected_keys,
            "selected_model_names": selected_display,
            "image_sources": [img.get("source") for img in images],
            "image_hashes": [img.get("content_hash") or img.get("id") for img in images],
            "sample_ids": [img.get("sample_id") for img in images if img.get("sample_id")],
            "returned_classes": returned_classes,
            "run_context": run_ctx.to_dict() if run_ctx else None,
            "yolo_world_request_path": (
                "Roboflow workflow specification class_names injection "
                "(published custom-workflow; parameters=None)"
            ),
        }

        if mode_ui == "Experimental Consensus" and results:
            agreement_label = _form_get("agreement_label", "At least 2 models")
            min_agree = 1
            if str(agreement_label).startswith("At least 2"):
                min_agree = 2
            elif str(agreement_label).startswith("All"):
                min_agree = len(selected_models)
            by_image: dict[str, dict[str, list]] = {}
            for r in results:
                by_image.setdefault(r.image_name, {})[r.model_name] = r.detections
            cons_list = []
            for img_name, model_map in by_image.items():
                cons_dets, multi, single = build_consensus_detections(
                    model_map,
                    min_agreement=min_agree,
                    iou_threshold=iou,
                )
                cons_list.append(
                    ConsensusResult(
                        consensus_detections=cons_dets,
                        consensus_count=len(cons_dets),
                        multi_model_supported=multi,
                        single_model_only=single,
                        min_agreement=min_agree,
                        model_results=[r for r in results if r.image_name == img_name],
                        warnings=[
                            "Experimental consensus — not a simple sum",
                            "Not guaranteed truth; review individual model results.",
                        ],
                    )
                )
            st.session_state.consensus_result = cons_list
        else:
            st.session_state.consensus_result = None

        if results and not failures:
            st.session_state.analysis_status = "complete"
            st.session_state.accepted_result_key = _result_key(results[0])
        elif results and failures:
            st.session_state.analysis_status = "partial"
            st.session_state.accepted_result_key = _result_key(results[0])
        else:
            st.session_state.analysis_status = "error"
            st.session_state.accepted_result_key = None

        progress.progress(1.0, text="Done")
    finally:
        pass


def stage_running() -> None:
    """Dedicated analysis progress / interim results page."""
    render_stepper("running")
    render_stage_header(
        "Running analysis",
        "Live progress and interim results — then continue to Review & Save.",
    )

    images = st.session_state.uploaded_images or []
    status = st.session_state.analysis_status
    mode_ui = _form_get("selected_mode", "Single Model")
    inference_ui = _form_get("inference_mode", "Whole Image")
    inference_mode = _inference_api_name(inference_ui)
    selected_names = list(_form_get("selected_models") or [])

    st.markdown(
        f"""
        <div class="aic-chip-grid">
          <div class="aic-chip aic-chip-r">
            <span class="aic-chip-label">Status</span>
            <span class="aic-chip-value">{status or "idle"}</span>
          </div>
          <div class="aic-chip aic-chip-b">
            <span class="aic-chip-label">Photos</span>
            <span class="aic-chip-value">{len(images)}</span>
          </div>
          <div class="aic-chip aic-chip-g">
            <span class="aic-chip-label">Mode</span>
            <span class="aic-chip-value">{mode_ui}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if selected_names:
        st.caption("Models: " + ", ".join(selected_names))

    if status in {"idle", None, ""} or not images:
        st.markdown(
            '<div class="aic-panel aic-panel-b"><div class="aic-panel-title">'
            "Not started</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">"
            "Start analysis from the Analyze step to see progress here."
            "</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Back to Analyze", key="run_back_idle", width="stretch"):
            navigate_to("wizard", stage="analyze")
        return

    if status == "running":
        st.markdown(
            '<div class="aic-panel aic-panel-r"><div class="aic-panel-title">'
            "In progress</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">"
            "Running inference on your photos. Keep this tab open until complete."
            "</p></div>",
            unsafe_allow_html=True,
        )
        if st.session_state.get("_analysis_executing"):
            st.info("Analysis is already in progress…")
            return
        selectable = {m.name: m for m in _analysis_models()}
        selected_models = [selectable[n] for n in selected_names if n in selectable]
        if not selected_models:
            st.error("No models selected.")
            if st.button("Back to Analyze", key="run_back_nomodel"):
                navigate_to("wizard", stage="analyze")
            return
        st.session_state._analysis_executing = True
        try:
            _execute_analysis_run(
                images=list(images),
                selected_models=selected_models,
                mode_ui=mode_ui,
                inference_mode=inference_mode,
            )
        finally:
            st.session_state._analysis_executing = False
            st.session_state.analyze_running = False
        st.rerun()
        return

    results: list[InferenceResult] = st.session_state.analysis_results or []
    failures: list[str] = st.session_state.analysis_failures or []
    meta = st.session_state.analysis_meta or {}
    total = sum(r.final_count for r in results) if results else 0
    models = meta.get("selected_model_names") or selected_names

    st.markdown(
        f"""
        <div class="aic-panel aic-panel-g">
          <div class="aic-panel-title">Interim overview</div>
          <div class="aic-chip-grid aic-chip-grid-4">
            <div class="aic-chip aic-chip-g">
              <span class="aic-chip-label">Result</span>
              <span class="aic-chip-value">{status}</span>
            </div>
            <div class="aic-chip aic-chip-b">
              <span class="aic-chip-label">Photos</span>
              <span class="aic-chip-value">{int(meta.get("number_of_photos") or len(images))}</span>
            </div>
            <div class="aic-chip aic-chip-r">
              <span class="aic-chip-label">Detections</span>
              <span class="aic-chip-value">{total}</span>
            </div>
            <div class="aic-chip aic-chip-g">
              <span class="aic-chip-label">Failures</span>
              <span class="aic-chip-value">{len(failures)}</span>
            </div>
          </div>
          <div class="aic-kv" style="margin-top:0.4rem;">
            <b>Models</b><br/>{", ".join(str(m) for m in models) or "—"}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if status == "error" or (not results and failures):
        _show_user_facing_error(
            message=failures[0] if failures else "No successful results.",
            error_type="api_error",
        )
        if st.button("Back to Analyze", key="run_back_err", width="stretch"):
            st.session_state.analysis_status = "idle"
            st.session_state.analyze_running = False
            navigate_to("wizard", stage="analyze")
        return

    pipeline_fault = any(
        (r.error_type in {"empty_workflow_output", "api_error"} or r.errors)
        for r in results
    )
    if pipeline_fault or failures:
        _render_analysis_failure_state(results, failures)
    elif total == 0 and results:
        _render_zero_detection_empty(results)
    else:
        st.markdown(
            '<div class="aic-panel aic-panel-g"><div class="aic-panel-title">'
            "Ready for review</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">"
            "Analysis finished. Continue to Review &amp; Save when ready."
            "</p></div>",
            unsafe_allow_html=True,
        )

    tab_sum, tab_fail = st.tabs(["Run summary", f"Failures ({len(failures)})"])
    with tab_sum:
        summaries = st.session_state.get("comparison_summaries") or []
        if summaries:
            st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch", height=240)
        else:
            st.caption("No run summary rows for this session.")
    with tab_fail:
        if failures:
            for fail in failures:
                st.caption(fail)
        else:
            st.caption("No failures.")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Back to Analyze", width="stretch", key="run_back_done"):
            navigate_to("wizard", stage="analyze")
    with b2:
        if st.button(
            "Continue to Review & Save",
            type="primary",
            width="stretch",
            key="run_to_review",
            disabled=not results,
        ):
            navigate_to("wizard", stage="review")




def _cache_key(
    image_hash: str,
    model_name: str,
    prompt: str,
    confidence: float,
    inference_mode: str,
    tile_size: int,
    tile_overlap: float,
    dedup: str,
    iou: float,
) -> str:
    raw = "|".join(
        [
            image_hash,
            model_name,
            prompt,
            f"{confidence:.4f}",
            inference_mode,
            str(tile_size),
            f"{tile_overlap:.3f}",
            dedup,
            f"{iou:.4f}",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage 6 — Review and Save
# ---------------------------------------------------------------------------


def _set_review_selection(detection_id: str | None) -> None:
    """Update the active detection and keep navigator widgets in sync."""
    st.session_state.selected_detection_id = detection_id
    if detection_id:
        st.session_state.rev_det_jump = detection_id
    else:
        st.session_state.pop("rev_det_jump", None)
    st.session_state.pop("rev_det_list", None)


def _canonicalize_result_classes(
    result: InferenceResult,
    run_ctx: AnalysisRunContext | None,
) -> InferenceResult:
    """Map synonym labels onto primary custom item types (separate per type)."""
    if result is None or run_ctx is None:
        return result
    alias_map = dict(run_ctx.class_alias_map or {})
    if not alias_map and not run_ctx.primary_item_types:
        return result
    changed = False
    for det in result.detections or []:
        new_name = canonicalize_detection_class(det.class_name, alias_map)
        if new_name != det.class_name:
            det.class_name = new_name
            changed = True
    if changed or run_ctx.primary_item_types:
        # Keep totals as sum of individual detections (already separate objects).
        included = [
            d
            for d in (result.detections or [])
            if getattr(d, "included_in_count", True)
            and not getattr(d, "excluded_by_region", False)
        ]
        if included:
            if any(getattr(d, "count_only", False) for d in included):
                total = sum(
                    int(getattr(d, "item_count", 1) or 1)
                    if getattr(d, "count_only", False)
                    else 1
                    for d in included
                )
            else:
                total = len(included)
            result.final_count = total
            result.raw_count = total
    return result


def _compute_reviewed() -> tuple[int, dict[str, Any]]:
    results: list[InferenceResult] = st.session_state.analysis_results or []
    key = st.session_state.accepted_result_key
    result = next((r for r in results if _result_key(r) == key), results[0] if results else None)
    rs = st.session_state.review_state
    if result is None:
        return 0, {"result": None}

    edits = st.session_state.get("review_edits") or {}
    excluded = set(edits.get("excluded_ids") or [])
    manuals = list(edits.get("manual_detections") or [])
    base_count = sum(
        (
            int(getattr(d, "item_count", 1) or 1)
            if bool(getattr(d, "count_only", False))
            else 1
        )
        for d in result.detections
        if d.detection_id not in excluded and getattr(d, "included_in_count", True)
    ) + len(manuals)

    direct = None
    if rs.get("use_direct") and rs.get("direct_count") is not None:
        direct = int(rs["direct_count"])
    reviewed = compute_reviewed_count(
        base_count,
        false_positive_adjustment=int(rs.get("false_positives") or 0),
        missed_item_adjustment=int(rs.get("missed_items") or 0),
        direct_reviewed_count=direct,
    )
    pct = compute_percentage_error(result.final_count, reviewed)
    return reviewed, {
        "result": result,
        "reviewed_count": reviewed,
        "base_visible_count": base_count,
        "false_positive_adjustment": int(rs.get("false_positives") or 0),
        "missed_item_adjustment": int(rs.get("missed_items") or 0),
        "notes": rs.get("notes") or "",
        "percentage_error": pct,
        "excluded_ids": list(excluded),
        "manual_count": len(manuals),
    }


def _save_inventory() -> None:
    if st.session_state.save_status in {"saved", "saving"}:
        return
    reviewed, payload = _compute_reviewed()
    result: InferenceResult | None = payload.get("result")
    if result is None:
        st.error("No accepted result available to save.")
        return
    meta = st.session_state.analysis_meta or {}
    notes = st.session_state.review_state.get("notes") or ""
    edits = st.session_state.get("review_edits") or {}
    # Append structured comparison metadata for newer records (old history still loads)
    returned_classes = sorted(
        {
            d.class_name
            for d in result.detections
            if d.class_name
        }
        | set(meta.get("returned_classes") or [])
    )
    extra = {
        "comparison_mode": bool(meta.get("comparison_mode")),
        "selected_model_keys": meta.get("selected_model_keys") or [],
        "selected_model_names": meta.get("selected_model_names") or [],
        "comparison_summaries": st.session_state.get("comparison_summaries") or [],
        "image_sources": meta.get("image_sources"),
        "image_hashes": meta.get("image_hashes"),
        "review_edits": edits,
        "accepted_model": result.model_name,
        "model_chosen_for_review": result.model_name,
        "final_reviewed_detections": [
            d.to_dict()
            for d in result.detections
            if d.detection_id not in set(edits.get("excluded_ids") or [])
        ]
        + list(edits.get("manual_detections") or []),
        "final_saved_count": reviewed,
        "selected_prompt": meta.get("selected_prompt"),
        "confidence_threshold": meta.get("confidence_threshold"),
        "inventory_type": meta.get("inventory_type"),
        "inventory_display_name": meta.get("inventory_display_name"),
        "custom_item_name": meta.get("custom_item_name"),
        "counting_unit": meta.get("counting_unit"),
        "effective_prompts": meta.get("effective_prompts") or [],
        "returned_classes": returned_classes,
        "run_context": meta.get("run_context"),
        "model": result.model_name,
    }
    try:
        notes_combined = (notes + "\n" if notes else "") + "AIC_META=" + json.dumps(extra)
    except Exception:  # noqa: BLE001
        notes_combined = notes
    st.session_state.save_status = "saving"
    record = {
        "yard": meta.get("yard"),
        "inventory_type": meta.get("inventory_type"),
        "photo_relationship": meta.get("photo_relationship"),
        "number_of_photos": meta.get("number_of_photos"),
        "selected_mode": meta.get("selected_mode"),
        "accepted_model": f"{result.model_name} · {result.image_name}",
        "selected_prompt": meta.get("selected_prompt"),
        "inference_mode": meta.get("inference_mode"),
        "tile_size": meta.get("tile_size"),
        "tile_overlap": meta.get("tile_overlap"),
        "deduplication_strategy": meta.get("deduplication_strategy"),
        "confidence_threshold": meta.get("confidence_threshold"),
        "iou_threshold": meta.get("iou_threshold"),
        "raw_ai_count": result.raw_count,
        "ai_count": result.final_count,
        "reviewed_count": reviewed,
        "false_positive_adjustment": payload["false_positive_adjustment"],
        "missed_item_adjustment": payload["missed_item_adjustment"],
        "average_confidence": result.avg_confidence,
        "suspected_overlap_count": result.suspected_overlap_count,
        "suspected_occlusion_count": result.suspected_occlusion_count,
        "processing_time_seconds": result.processing_time_seconds,
        "percentage_error": payload["percentage_error"],
        "notes": notes_combined,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    owner = auth_session.current_user()
    if owner is None:
        st.session_state.save_status = "idle"
        st.error("You must be signed in to save inventory history.")
        return
    # Ownership is mandatory so history can never land in a shared/unowned pool.
    record["user_id"] = owner.user_id
    record["username"] = owner.username
    try:
        row_id = insert_inventory_count(record)
        st.session_state.saved_record = {
            "id": row_id,
            "created_at": record["created_at"],
            "yard": record["yard"],
            "inventory_type": record["inventory_type"],
            "reviewed_count": reviewed,
        }
        st.session_state.save_status = "saved"
        st.rerun()
    except DatabaseError as exc:
        st.session_state.save_status = "idle"
        _error_box("Failed to save record.", str(exc))


def _build_review_detections(accepted: InferenceResult) -> tuple[list[Detection], set[str], dict[str, Any]]:
    edits = st.session_state.setdefault(
        "review_edits",
        {"excluded_ids": [], "manual_detections": [], "class_overrides": {}},
    )
    excluded = set(edits.get("excluded_ids") or [])
    overrides = edits.get("class_overrides") or {}
    review_dets: list[Detection] = []
    for d in accepted.detections:
        if d.detection_id in excluded:
            continue
        clone = Detection(**{**d.to_dict()})
        if d.detection_id in overrides:
            clone.class_name = str(overrides[d.detection_id])
        review_dets.append(clone)
    for md in edits.get("manual_detections") or []:
        review_dets.append(
            Detection(
                detection_id=md["detection_id"],
                class_name=md.get("class_name") or "manual",
                confidence=float(md.get("confidence") or 1.0),
                x1=float(md["x1"]),
                y1=float(md["y1"]),
                x2=float(md["x2"]),
                y2=float(md["y2"]),
                center_x=float(md["center_x"]),
                center_y=float(md["center_y"]),
                width=float(md["width"]),
                height=float(md["height"]),
                source_model=accepted.model_name,
                source_image=accepted.image_name,
                is_manual=True,
                included_in_count=True,
            )
        )
    review_dets = assign_marker_numbers(review_dets)
    return review_dets, excluded, edits


def stage_review() -> None:
    render_stepper("review")
    st.markdown('<div class="aic-review-compact">', unsafe_allow_html=True)
    render_stage_header(
        "Review & Save",
        "Confirm detections, adjust counts, then save the inventory record.",
    )

    if st.session_state.save_status == "saved" and st.session_state.saved_record:
        rec = st.session_state.saved_record
        st.markdown(
            f"""
            <div class="aic-panel aic-panel-g">
              <div class="aic-panel-title">Saved successfully</div>
              <div class="aic-chip-grid aic-chip-grid-4">
                <div class="aic-chip aic-chip-g">
                  <span class="aic-chip-label">Record ID</span>
                  <span class="aic-chip-value">{rec.get("id")}</span>
                </div>
                <div class="aic-chip aic-chip-b">
                  <span class="aic-chip-label">Inventory</span>
                  <span class="aic-chip-value">{rec.get("inventory_type") or "—"}</span>
                </div>
                <div class="aic-chip aic-chip-r">
                  <span class="aic-chip-label">Reviewed</span>
                  <span class="aic-chip-value">{rec.get("reviewed_count")}</span>
                </div>
                <div class="aic-chip aic-chip-g">
                  <span class="aic-chip-label">Saved</span>
                  <span class="aic-chip-value">{rec.get("created_at") or "—"}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("View History", width="stretch", key="post_hist"):
                open_settings(section="history")
        with c2:
            if st.button(
                "Start New Analysis",
                type="primary",
                width="stretch",
                key="post_new",
            ):
                reset_active_analysis(go_home=False, start_wizard=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    results: list[InferenceResult] = st.session_state.analysis_results or []
    if not results or st.session_state.analysis_status not in {"complete", "partial"}:
        st.markdown(
            '<div class="aic-panel aic-panel-b"><div class="aic-panel-title">'
            "Nothing to review yet</div>"
            "<p class=\"aic-muted\" style=\"margin:0;\">"
            "Finish a run on the Running step, then return here to review and save."
            "</p></div>",
            unsafe_allow_html=True,
        )
        render_nav_buttons(back_stage="running", key_prefix="rev_gate")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.session_state.analysis_status == "partial":
        st.warning("Showing successful results. Some runs failed.")

    photo_names: list[str] = []
    for r in results:
        if r.image_name not in photo_names:
            photo_names.append(r.image_name)

    meta = st.session_state.analysis_meta or {}
    photo_n = int(meta.get("number_of_photos") or len(photo_names))
    inv_disp = (
        meta.get("inventory_display_name")
        or inventory_display_name(
            meta.get("inventory_type"),
            custom_item_name=meta.get("custom_item_name"),
        )
        or meta.get("inventory_type")
        or "—"
    )
    unit = meta.get("counting_unit") or counting_unit_for(meta.get("inventory_type"))
    st.markdown(
        f'<p class="aic-review-meta">'
        f"{inv_disp} · {photo_n} photo"
        f'{"s" if photo_n != 1 else ""} · {meta.get("selected_mode") or "Single Model"}'
        f"<br/>Counting unit: {unit}"
        f"</p>",
        unsafe_allow_html=True,
    )

    active_image = st.session_state.get("review_active_image") or photo_names[0]
    if active_image not in photo_names:
        active_image = photo_names[0]
    st.session_state.review_active_image = active_image

    models_for_image = []
    for r in results:
        if r.image_name == active_image and r.model_name not in models_for_image:
            models_for_image.append(r.model_name)
    if not models_for_image:
        models_for_image = [results[0].model_name]
        active_image = results[0].image_name
        st.session_state.review_active_image = active_image

    view_model = st.session_state.get("review_active_model") or models_for_image[0]
    if view_model not in models_for_image:
        view_model = models_for_image[0]
    st.session_state.review_active_model = view_model

    # Viewing vs accepted-for-review are independent; tabs never rerun inference.
    viewed = next(
        (r for r in results if r.image_name == active_image and r.model_name == view_model),
        next(r for r in results if r.image_name == active_image),
    )
    accepted_key = st.session_state.get("accepted_result_key")
    accepted = next((r for r in results if _result_key(r) == accepted_key), None)
    if accepted is None or accepted.image_name != active_image:
        # Default accepted result: currently viewed successful result for this photo
        accepted = viewed
        st.session_state.accepted_result_key = _result_key(accepted)
    # Display canvas follows the active tab/view model
    display_result = viewed

    summaries = st.session_state.get("comparison_summaries") or []
    if len(models_for_image) > 1 or (
        st.session_state.analysis_meta or {}
    ).get("comparison_mode"):
        st.caption(f"Selected for Review: **{accepted.model_name}**")
        view_mode = "Tabs"
        if len(models_for_image) == 2:
            view_mode = st.radio(
                "View",
                ["Tabs", "Side by Side"],
                horizontal=True,
                key="compare_view_mode",
            )
        img_results = [r for r in results if r.image_name == active_image]
        with st.expander("Model comparison details", expanded=False):
            # Factual labels only (not accuracy)
            st.caption(
                "Tabs switch the view only — they do not rerun inference. "
                "Factual labels do not prove accuracy."
            )
            if img_results:
                fastest = min(img_results, key=lambda r: r.processing_time_seconds)
                most_det = max(img_results, key=lambda r: r.final_count)
                highest_conf = max(img_results, key=lambda r: r.avg_confidence)
                fewest_warn = min(
                    img_results,
                    key=lambda r: len(r.warnings or []) + r.suspected_overlap_count,
                )
                st.caption(
                    f"Fastest: **{fastest.model_name}** · "
                    f"Most detections: **{most_det.model_name}** · "
                    f"Highest avg confidence: **{highest_conf.model_name}** · "
                    f"Fewest warnings: **{fewest_warn.model_name}**"
                )
            rows = [
                {
                    "Model": s.get("model"),
                    "Status": s.get("status"),
                    "Raw count": format_count_display(s.get("raw_count")),
                    "Final count": format_count_display(s.get("final_count")),
                    "Avg conf": format_count_display(s.get("avg_confidence")),
                    "Max conf": format_count_display(s.get("max_confidence")),
                    "Classes": s.get("classes"),
                    "Time (s)": s.get("processing_time"),
                    "Warnings": s.get("warning_count")
                    if s.get("warning_count") is not None
                    else (
                        len(str(s.get("warnings") or "").split(";"))
                        if s.get("warnings")
                        else 0
                    ),
                }
                for s in (summaries or [])
                if s.get("image") == active_image
            ] or [
                {
                    "Model": r.model_name,
                    "Status": "Success with detections"
                    if r.final_count
                    else "Success with zero detections",
                    "Raw count": r.raw_count,
                    "Final count": r.final_count,
                    "Avg conf": round(r.avg_confidence, 4),
                    "Max conf": round(r.max_confidence, 4),
                    "Classes": ", ".join(sorted({d.class_name for d in r.detections}))
                    or "(none)",
                    "Time (s)": round(r.processing_time_seconds, 3),
                    "Warnings": len(r.warnings or []),
                }
                for r in img_results
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.session_state.compare_side_by_side = view_mode == "Side by Side"
    elif summaries:
        with st.expander("Run summary", expanded=False):
            st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")

    viz_options = [
        "Roboflow Labels",
        "Numbered Markers",
        "Bounding Boxes",
        "Both",
    ]
    default_viz = st.session_state.get("annotation_style_label", "Roboflow Labels")
    if default_viz not in viz_options:
        default_viz = "Roboflow Labels"
    style = st.radio(
        "Visualization",
        viz_options,
        index=viz_options.index(default_viz),
        horizontal=True,
        key="rev_viz_style",
        help=(
            "Roboflow Labels draws class-colored boxes with class/confidence chips "
            "(like Roboflow Annotate). The other options keep the original marker styles."
        ),
    )
    st.session_state.annotation_style_label = style
    style_key = {
        "Roboflow Labels": "roboflow",
        "Bounding Boxes": "boxes",
        "Numbered Markers": "markers",
        "Both": "both",
    }[style]

    # Canvas / detection list follow the active tab (view); save uses accepted_result_key.
    review_dets, excluded, edits = _build_review_detections(display_result)
    det_ids = {d.detection_id for d in review_dets}
    # Keep list widget and marker selection synchronized before rendering the image.
    if st.session_state.get("rev_det_list") in det_ids:
        st.session_state.selected_detection_id = st.session_state.rev_det_list
    selected_id = st.session_state.get("selected_detection_id")
    if selected_id and selected_id not in det_ids:
        selected_id = review_dets[0].detection_id if review_dets else None
        st.session_state.selected_detection_id = selected_id
    elif not selected_id and review_dets:
        selected_id = review_dets[0].detection_id
        st.session_state.selected_detection_id = selected_id

    match = next(
        (
            img
            for img in st.session_state.uploaded_images
            if img["name"] == display_result.image_name
        ),
        None,
    )

    excluded_dets = [d for d in display_result.detections if d.detection_id in excluded]
    for i, d in enumerate(sorted(excluded_dets, key=lambda x: (x.center_y, x.center_x)), start=1):
        if d.marker_number is None:
            d.marker_number = i

    meta_early = st.session_state.get("analysis_meta") or {}
    alias_map_early = {
        str(k).casefold(): str(v)
        for k, v in dict(meta_early.get("class_alias_map") or {}).items()
    }
    # Only the item types the user asked to find (not every model class).
    requested_types = list(meta_early.get("primary_item_types") or [])
    if not requested_types:
        requested_types = list(meta_early.get("effective_prompts") or [])
    type_options = available_item_types(
        list(review_dets) + list(excluded_dets),
        primary_types=requested_types,
        alias_map=alias_map_early,
        requested_only=True,
    )
    type_choices = [ITEM_TYPE_ALL] + type_options
    current_type = st.session_state.get("rev_item_type_filter", ITEM_TYPE_ALL)
    if current_type not in type_choices:
        current_type = ITEM_TYPE_ALL
        st.session_state.rev_item_type_filter = ITEM_TYPE_ALL

    if len(type_options) > 1:
        prev_type = st.session_state.get("_rev_item_type_prev", ITEM_TYPE_ALL)
        picked = st.selectbox(
            "Item type",
            type_choices,
            index=type_choices.index(current_type),
            key="rev_item_type_filter",
            help=(
                "Show only one of the item types you chose to detect. "
                "Marker numbers stay shared across all types."
            ),
        )
        st.caption("Numbering is shared across all item types.")
        current_type = picked
        if picked != prev_type:
            st.session_state._rev_item_type_prev = picked
            _set_review_selection(None)
            selected_id = None

    filt_label = st.session_state.get("rev_det_filter", "All")
    filt_key = {
        "All": "all",
        "Included": "included",
        "Excluded": "excluded",
        "Warnings": "warnings",
        "Manual": "manual",
    }.get(filt_label, "all")
    nav_pool = filter_detections(
        review_dets,
        filt_key,
        excluded_detections=excluded_dets,
        item_type=current_type,
        alias_map=alias_map_early,
    )
    # Canvas follows the selected item type; marker numbers remain global.
    canvas_dets = filter_detections(
        review_dets,
        "all",
        item_type=current_type,
        alias_map=alias_map_early,
    )
    # Prefer an explicit jump only when it is still in the current pool.
    jump_id = st.session_state.get("rev_det_jump")
    if jump_id in {d.detection_id for d in nav_pool}:
        selected_id = jump_id
        st.session_state.selected_detection_id = jump_id
    elif nav_pool and selected_id not in {d.detection_id for d in nav_pool}:
        selected_id = nav_pool[0].detection_id
        _set_review_selection(selected_id)
    elif not nav_pool:
        selected_id = None
        st.session_state.selected_detection_id = None

    # Wide canvas + compact inspector; top-align so the tall right panel
    # does not leave a large empty gap under the image.
    st.markdown('<div class="aic-review-layout">', unsafe_allow_html=True)
    left, right = st.columns([2.55, 1.0], gap="large")

    with left:
        # Model result selector (comparison) — tabs or side-by-side; no re-inference
        if len(models_for_image) > 1 and not st.session_state.get("compare_side_by_side"):
            mcols = st.columns(len(models_for_image))
            for i, name in enumerate(models_for_image):
                with mcols[i]:
                    label = name
                    if name == accepted.model_name:
                        label = f"{name} ✓"
                    if st.button(
                        label,
                        key=f"rev_model_tab_{name}",
                        width="stretch",
                        type="primary" if name == view_model else "secondary",
                    ):
                        st.session_state.review_active_model = name
                        st.session_state.selected_detection_id = None
                        st.rerun()
            if st.button("Use This Result", type="primary", key="rev_use_model"):
                st.session_state.accepted_result_key = _result_key(display_result)
                st.session_state.review_active_model = display_result.model_name
                st.success(
                    f"Selected for Review: {display_result.model_name}"
                )
                st.rerun()

        if (
            len(models_for_image) == 2
            and st.session_state.get("compare_side_by_side")
            and match
        ):
            st.caption("Side by side — same image & overlays; independent predictions.")
            side_cols = st.columns(2)
            for col, name in zip(side_cols, models_for_image):
                with col:
                    st.markdown(f"**{name}**")
                    r_side = next(
                        (
                            r
                            for r in results
                            if r.image_name == active_image and r.model_name == name
                        ),
                        None,
                    )
                    if r_side is None:
                        st.warning("No result")
                        continue
                    dets_side, _, _ = _build_review_detections(r_side)
                    try:
                        base_s = Image.open(io.BytesIO(match["data"])).convert("RGB")
                        ann = annotate_image(
                            base_s,
                            dets_side,
                            model_name=name,
                            style=style_key,
                        )
                        st.image(image_to_png_bytes(ann), width="stretch")
                    except Exception:  # noqa: BLE001
                        if r_side.annotated_image_bytes:
                            st.image(r_side.annotated_image_bytes, width="stretch")
                    st.caption(
                        f"Count {r_side.final_count} · "
                        f"conf {r_side.avg_confidence:.2f} · "
                        f"{r_side.processing_time_seconds:.2f}s"
                    )
                    if st.button(f"Use {name}", key=f"rev_use_side_{name}"):
                        st.session_state.review_active_model = name
                        st.session_state.accepted_result_key = _result_key(r_side)
                        st.session_state.compare_side_by_side = False
                        st.rerun()
            # Skip single canvas below when side-by-side is active
            match = None

        st.markdown('<div class="aic-review-canvas"></div>', unsafe_allow_html=True)
        if match:
            base_img = Image.open(io.BytesIO(match["data"])).convert("RGB")
            # Keep a large on-screen preview without crushing dense scenes.
            preview = preview_resize(base_img, max_side=1600)
            # Scale detections into preview space when the source was downscaled.
            scale_x = preview.size[0] / float(base_img.size[0] or 1)
            scale_y = preview.size[1] / float(base_img.size[1] or 1)
            draw_dets = canvas_dets
            if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
                from copy import deepcopy

                draw_dets = []
                for d in canvas_dets:
                    c = deepcopy(d)
                    c.x1 *= scale_x
                    c.x2 *= scale_x
                    c.y1 *= scale_y
                    c.y2 *= scale_y
                    c.center_x *= scale_x
                    c.center_y *= scale_y
                    c.width *= scale_x
                    c.height *= scale_y
                    draw_dets.append(c)
            annotated = annotate_image(
                preview,
                draw_dets,
                model_name=display_result.model_name,
                style=style_key,
                selected_detection_id=selected_id,
                show_legend=False,
            )
            st.image(image_to_png_bytes(annotated), width="stretch")
        elif display_result.annotated_image_bytes:
            st.image(display_result.annotated_image_bytes, width="stretch")
        else:
            st.info("No annotated image available.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Compact image navigator only — detection tools live in the right workspace
        if len(photo_names) > 1:
            idx = photo_names.index(active_image)
            nav_l, nav_m, nav_r = st.columns([1, 2, 1])
            with nav_l:
                if st.button("← Previous", disabled=idx <= 0, key="rev_prev", width="stretch"):
                    st.session_state.review_active_image = photo_names[idx - 1]
                    st.session_state.selected_detection_id = None
                    st.rerun()
            with nav_m:
                st.markdown(
                    f"<div style='text-align:center;padding-top:0.4rem;'>Image {idx + 1} of {len(photo_names)}</div>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                if st.button("Next →", disabled=idx >= len(photo_names) - 1, key="rev_next", width="stretch"):
                    st.session_state.review_active_image = photo_names[idx + 1]
                    st.session_state.selected_detection_id = None
                    st.rerun()
            tcols = st.columns(min(4, len(photo_names)))
            for i, name in enumerate(photo_names):
                with tcols[i % len(tcols)]:
                    count = next(
                        (
                            r.final_count
                            for r in results
                            if r.image_name == name and r.model_name == view_model
                        ),
                        next((r.final_count for r in results if r.image_name == name), 0),
                    )
                    warn_n = next(
                        (
                            len(r.warnings or [])
                            + r.suspected_overlap_count
                            + r.suspected_occlusion_count
                            for r in results
                            if r.image_name == name and r.model_name == view_model
                        ),
                        0,
                    )
                    label = f"Img {i+1} · {count}"
                    if warn_n:
                        label += f" ⚠{warn_n}"
                    if st.button(
                        label,
                        key=f"rev_thumb_{i}",
                        width="stretch",
                        type="primary" if name == active_image else "secondary",
                    ):
                        st.session_state.review_active_image = name
                        st.session_state.selected_detection_id = None
                        st.rerun()

    with right:
        included = len(review_dets)
        excluded_n = len(excluded)
        warn_count = (
            display_result.suspected_overlap_count
            + display_result.suspected_occlusion_count
            + len(display_result.warnings or [])
        )
        reviewed, payload = _compute_reviewed()

        # Recompute pool from filter (widget lives in Detection tab)
        st.markdown('<div class="aic-review-workspace">', unsafe_allow_html=True)
        if display_result.model_name != accepted.model_name:
            st.caption(
                f"Viewing **{display_result.model_name}** · "
                f"Selected for Review: **{accepted.model_name}**"
            )
        st.markdown(
            f"""
            <div class="aic-metric-grid">
              <div class="aic-metric-tile">AI<b>{display_result.final_count}</b></div>
              <div class="aic-metric-tile">Final<b>{reviewed}</b></div>
              <div class="aic-metric-tile">Incl.<b>{included}</b></div>
              <div class="aic-metric-tile">Excl.<b>{excluded_n}</b></div>
              <div class="aic-metric-tile">Dupes<b>{display_result.duplicates_removed}</b></div>
              <div class="aic-metric-tile">Warn<b>{warn_count}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        meta = st.session_state.get("analysis_meta") or {}
        primary_types = list(meta.get("primary_item_types") or [])
        alias_map = {
            str(k).casefold(): str(v)
            for k, v in dict(meta.get("class_alias_map") or {}).items()
        }
        if len(primary_types) > 1 or (
            is_custom_inventory(meta.get("inventory_type")) and primary_types
        ):
            by_type = counts_by_item_type(
                review_dets,
                primary_types=primary_types,
                alias_map=alias_map,
            )
            if by_type:
                chips = " · ".join(
                    f"**{name}**: {count}" for name, count in by_type.items()
                )
                st.caption(f"Counts by item type (separate): {chips}")

        tab_det, tab_adj, tab_issues = st.tabs(["Detection", "Adjust", "Issues"])

        with tab_det:
            prev_filt = st.session_state.get("_rev_det_filter_prev", "All")
            st.radio(
                "Filter",
                ["All", "Included", "Excluded", "Warnings", "Manual"],
                horizontal=True,
                key="rev_det_filter",
                label_visibility="collapsed",
            )
            filt_label = st.session_state.get("rev_det_filter", "All")
            if filt_label != prev_filt:
                st.session_state._rev_det_filter_prev = filt_label
                _set_review_selection(None)
                selected_id = None
            if len(type_options) > 1:
                st.caption(
                    f"Item type view: **{current_type}** "
                    "(only types you chose to detect; numbers stay shared)"
                )
            filt_key = {
                "All": "all",
                "Included": "included",
                "Excluded": "excluded",
                "Warnings": "warnings",
                "Manual": "manual",
            }.get(filt_label, "all")
            nav_pool = filter_detections(
                review_dets,
                filt_key,
                excluded_detections=excluded_dets,
                item_type=current_type,
                alias_map=alias_map_early,
            )
            if nav_pool and selected_id not in {d.detection_id for d in nav_pool}:
                selected_id = nav_pool[0].detection_id
                _set_review_selection(selected_id)

            if nav_pool:
                cur_idx = index_of_detection(nav_pool, selected_id)
                n1, n2, n3 = st.columns([1, 1.2, 1])
                with n1:
                    if st.button(
                        "Prev",
                        key="rev_det_prev",
                        width="stretch",
                        disabled=cur_idx <= 0,
                    ):
                        nid = step_detection_id(nav_pool, selected_id, delta=-1)
                        _set_review_selection(nid)
                        st.rerun()
                with n2:
                    st.caption(f"{cur_idx + 1} / {len(nav_pool)}")
                with n3:
                    if st.button(
                        "Next",
                        key="rev_det_next",
                        width="stretch",
                        disabled=cur_idx >= len(nav_pool) - 1,
                    ):
                        nid = step_detection_id(nav_pool, selected_id, delta=1)
                        _set_review_selection(nid)
                        st.rerun()
                options = [d.detection_id for d in nav_pool]
                # Drive the selectbox from the active selection (Prev/Next/Exclude).
                if selected_id in options:
                    st.session_state.rev_det_jump = selected_id
                elif st.session_state.get("rev_det_jump") not in options:
                    st.session_state.rev_det_jump = options[0]
                    selected_id = options[0]
                    st.session_state.selected_detection_id = selected_id
                jump = st.selectbox(
                    "Detection",
                    options=options,
                    format_func=lambda did: format_detection_option(
                        next(d for d in nav_pool if d.detection_id == did),
                        excluded=did in excluded,
                    ),
                    key="rev_det_jump",
                    label_visibility="collapsed",
                )
                if jump != selected_id:
                    st.session_state.selected_detection_id = jump
                    selected_id = jump
            else:
                st.caption("No detections match this filter.")

            selected = next(
                (
                    d
                    for d in list(review_dets) + list(excluded_dets)
                    if d.detection_id == selected_id
                ),
                None,
            )
            if selected:
                color = color_for_detection(selected, selected.marker_number or 1)
                is_excl = selected.detection_id in excluded
                status = (
                    "Excluded"
                    if is_excl
                    else ("Manual" if selected.is_manual else "Included")
                )
                conf_txt = format_confidence_percent(selected.confidence)
                band = confidence_band(selected.confidence)
                warn_bits = []
                if is_low_confidence_warning(selected.confidence):
                    warn_bits.append("Low confidence")
                if selected.suspected_overlap:
                    warn_bits.append("Possible duplicate")
                if selected.suspected_occlusion:
                    warn_bits.append("Partial / occluded")
                warn_html = (
                    f"<br/><span style='opacity:0.85;'>⚠ {' · '.join(warn_bits)}</span>"
                    if warn_bits
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="aic-det-row-selected">
                      <span class="aic-det-chip" style="background:{css_rgb(color)};">{selected.marker_number}</span>
                      <b>#{selected.marker_number}</b> {selected.class_name}<br/>
                      {CONFIDENCE_LABEL}: {conf_txt} ({band}) · {status}
                      {warn_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                a1, a2 = st.columns(2)
                with a1:
                    if is_excl:
                        if st.button("Include", key="rev_incl_sel", width="stretch"):
                            nxt = next_detection_id_after_toggle(
                                nav_pool, selected.detection_id
                            )
                            excluded.discard(selected.detection_id)
                            edits["excluded_ids"] = list(excluded)
                            st.session_state.review_edits = edits
                            # Stay on this detection after it moves back to Included,
                            # unless the Excluded filter would hide it.
                            if filt_key == "excluded":
                                _set_review_selection(nxt)
                            else:
                                _set_review_selection(selected.detection_id)
                            st.rerun()
                    elif st.button("Exclude", key="rev_excl_sel", width="stretch"):
                        nxt = next_detection_id_after_toggle(
                            nav_pool, selected.detection_id
                        )
                        if not selected.is_manual:
                            excluded.add(selected.detection_id)
                            edits["excluded_ids"] = list(excluded)
                        else:
                            edits["manual_detections"] = [
                                m
                                for m in (edits.get("manual_detections") or [])
                                if m.get("detection_id") != selected.detection_id
                            ]
                            nxt = next_detection_id_after_toggle(
                                nav_pool, selected.detection_id
                            )
                        st.session_state.review_edits = edits
                        _set_review_selection(nxt)
                        st.rerun()
                with a2:
                    new_label = st.text_input(
                        "Label",
                        value=selected.class_name,
                        key=f"rev_edit_label_{selected.detection_id}",
                        label_visibility="collapsed",
                    )
                    if new_label != selected.class_name and st.button(
                        "Save label", key="rev_save_label", width="stretch"
                    ):
                        if selected.is_manual:
                            manuals = list(edits.get("manual_detections") or [])
                            for m in manuals:
                                if m.get("detection_id") == selected.detection_id:
                                    m["class_name"] = new_label
                            edits["manual_detections"] = manuals
                        else:
                            overrides = dict(edits.get("class_overrides") or {})
                            overrides[selected.detection_id] = new_label
                            edits["class_overrides"] = overrides
                        st.session_state.review_edits = edits
                        st.rerun()
                if excluded and st.button("Restore excluded", key="rev_restore"):
                    edits["excluded_ids"] = []
                    st.session_state.review_edits = edits
                    st.rerun()
                with st.expander("Details", expanded=False):
                    st.caption(CONFIDENCE_HELP)
                    st.write(f"Stable detection ID: `{selected.detection_id}`")
                    st.write(f"Model: {display_result.model_name}")
                    st.write(
                        f"Box: ({selected.x1:.0f},{selected.y1:.0f})-"
                        f"({selected.x2:.0f},{selected.y2:.0f})"
                    )
                    st.write(f"Source: {display_result.source or '(n/a)'}")
            else:
                st.caption("Select a detection to edit.")

            page = int(st.session_state.get("rev_det_page", 0) or 0)
            page_items, page, total_pages = paginate(nav_pool, page, min(8, PAGE_SIZE))
            st.session_state.rev_det_page = page
            if page_items:
                for d in page_items:
                    color = color_for_detection(d, d.marker_number or 1)
                    warn = "⚠" if (
                        d.suspected_overlap
                        or d.suspected_occlusion
                        or is_low_confidence_warning(d.confidence)
                    ) else ""
                    state = (
                        "Excl"
                        if d.detection_id in excluded
                        else ("Man" if d.is_manual else "Incl")
                    )
                    conf_txt = format_confidence_percent(d.confidence)
                    st.markdown(
                        f"""
                        <div class="{'aic-det-row-selected' if d.detection_id == selected_id else ''}"
                             style="padding:0.15rem 0.3rem;font-size:0.84rem;">
                          <span class="aic-det-chip" style="background:{css_rgb(color)};">{d.marker_number}</span>
                          {d.class_name} · {conf_txt} · {state} {warn}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                if total_pages > 1:
                    p1, p2, p3 = st.columns([1, 2, 1])
                    with p1:
                        if st.button("◀", key="rev_page_prev", disabled=page <= 0):
                            st.session_state.rev_det_page = page - 1
                            st.rerun()
                    with p2:
                        st.caption(f"{page + 1}/{total_pages}")
                    with p3:
                        if st.button(
                            "▶",
                            key="rev_page_next",
                            disabled=page >= total_pages - 1,
                        ):
                            st.session_state.rev_det_page = page + 1
                            st.rerun()

        with tab_adj:
            st.caption(f"Final count: **{reviewed}**")
            rs = st.session_state.review_state
            use_direct = st.checkbox(
                "Enter reviewed count directly",
                value=bool(rs.get("use_direct")),
                key="rev_direct_cb",
            )
            fp = st.number_input(
                "False positives (−)",
                min_value=0,
                value=int(rs.get("false_positives") or 0),
                step=1,
                key="rev_fp",
            )
            missed = st.number_input(
                "Missed items (+)",
                min_value=0,
                value=int(rs.get("missed_items") or 0),
                step=1,
                key="rev_missed",
            )
            direct_val = int(
                rs.get("direct_count") if rs.get("direct_count") is not None else included
            )
            if use_direct:
                direct_val = st.number_input(
                    "Direct reviewed count",
                    min_value=0,
                    value=direct_val,
                    step=1,
                    key="rev_direct_val",
                )
            notes = st.text_area(
                "Notes", value=rs.get("notes") or "", height=70, key="rev_notes"
            )
            st.session_state.review_state = {
                "use_direct": use_direct,
                "direct_count": direct_val if use_direct else None,
                "false_positives": int(fp),
                "missed_items": int(missed),
                "notes": notes,
            }
            with st.expander("Add manual detection", expanded=False):
                st.caption("Set coordinates, then confirm.")
                mx = st.number_input(
                    "X",
                    min_value=0,
                    value=int((match["width"] if match else 100) // 2),
                    key="man_x",
                )
                my = st.number_input(
                    "Y",
                    min_value=0,
                    value=int((match["height"] if match else 100) // 2),
                    key="man_y",
                )
                mcls = st.text_input("Class label", value="manual", key="man_cls")
                if st.button("Confirm Marker", key="man_add"):
                    half = 12.0
                    mid = {
                        "detection_id": f"manual_{hashlib.sha256(f'{mx}_{my}_{mcls}'.encode()).hexdigest()[:12]}",
                        "class_name": mcls or "manual",
                        "confidence": 1.0,
                        "x1": float(mx) - half,
                        "y1": float(my) - half,
                        "x2": float(mx) + half,
                        "y2": float(my) + half,
                        "center_x": float(mx),
                        "center_y": float(my),
                        "width": half * 2,
                        "height": half * 2,
                        "added_at": datetime.now(timezone.utc).isoformat(),
                    }
                    manuals = list(edits.get("manual_detections") or [])
                    manuals.append(mid)
                    edits["manual_detections"] = manuals
                    st.session_state.review_edits = edits
                    st.session_state.selected_detection_id = mid["detection_id"]
                    st.rerun()
            if st.button("Reset adjustments", key="rev_reset_edits"):
                st.session_state.review_edits = {
                    "excluded_ids": [],
                    "manual_detections": [],
                    "class_overrides": {},
                }
                st.session_state.selected_detection_id = None
                st.rerun()

        with tab_issues:
            st.caption(f"AI removed {display_result.duplicates_removed} duplicate(s).")
            overlap_pairs = [d for d in review_dets if d.suspected_overlap]
            if overlap_pairs:
                for d in overlap_pairs[:6]:
                    c1, c2 = st.columns([2.2, 1])
                    with c1:
                        st.caption(
                            f"#{d.marker_number} possible duplicate "
                            f"({d.class_name}, {d.confidence:.0%})"
                        )
                    with c2:
                        if st.button("Exclude", key=f"dup_excl_{d.detection_id}"):
                            excluded.add(d.detection_id)
                            edits["excluded_ids"] = list(excluded)
                            st.session_state.review_edits = edits
                            st.rerun()
            else:
                st.caption("No open duplicate candidates.")

            shown = False
            for d in review_dets:
                if d.suspected_occlusion:
                    st.caption(f"#{d.marker_number} — partially outside / occluded")
                    shown = True
                if d.confidence < 0.35:
                    st.caption(f"#{d.marker_number} — low confidence ({d.confidence:.0%})")
                    shown = True
            for w in (display_result.warnings or [])[:6]:
                st.caption(str(w))
                shown = True
            if not shown and not overlap_pairs:
                st.caption("No warnings.")

            with st.expander("Result summary", expanded=False):
                st.write(f"Avg confidence: {display_result.avg_confidence:.1%}")
                st.write(f"Processing time: {display_result.processing_time_seconds:.2f}s")
                st.write(f"API calls: {display_result.api_calls_used}")
                st.write(f"Inference mode: {display_result.inference_mode}")
                if display_result.strategy_counts:
                    st.json(display_result.strategy_counts)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # .aic-review-layout

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("← Back", width="stretch", key="rev_back"):
            navigate_to("wizard", stage="running")
    with b2:
        if st.button("Re-run Analysis", width="stretch", key="rev_rerun"):
            st.session_state.analysis_status = "idle"
            st.session_state.analysis_results = []
            st.session_state.analysis_failures = []
            st.session_state.save_status = "idle"
            st.session_state.saved_record = None
            st.session_state.inference_cache = {}
            st.session_state.selected_detection_id = None
            navigate_to("wizard", stage="analyze")
    with b3:
        save_disabled = st.session_state.save_status in {"saved", "saving"}
        if st.button(
            "Save Inventory",
            type="primary",
            width="stretch",
            disabled=save_disabled,
            key="rev_save",
        ):
            _save_inventory()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _render_wizard() -> None:
    stage = normalize_stage(st.session_state.get("wizard_stage") or "setup")
    st.session_state.wizard_stage = stage

    if stage == "review" and st.session_state.analysis_status not in {"complete", "partial"}:
        if st.session_state.save_status != "saved":
            st.warning("Complete analysis before reviewing.")
            stage = (
                "running"
                if st.session_state.analysis_status in {"running", "error"}
                else "analyze"
            )
            st.session_state.wizard_stage = stage

    dispatch = {
        "setup": stage_setup,
        "photos": stage_photos,
        "analyze": stage_analyze,
        "running": stage_running,
        "review": stage_review,
    }
    if stage not in dispatch:
        stage = "setup"
        st.session_state.wizard_stage = stage
    dispatch[stage]()


def main() -> None:
    st.set_page_config(
        page_title="AI Inventory Counter",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    config.reload_settings()
    _init_session()
    inject_css()
    ensure_data_dir()
    try:
        initialize_database()
        model_access.ensure_default_policies()
        from shape_detection_storage import ensure_default_feature_policy

        ensure_default_feature_policy()
    except DatabaseError:
        pass

    # Login gate — nothing below this point renders for an anonymous visitor.
    user = auth_session.enforce_session()
    if user is None:
        auth_ui.render_login_page()
        return

    if user.force_password_change:
        auth_ui.render_force_password_change(user)
        return

    # Left panel owns Home / Administration / History / AI Config / … / Profile.
    auth_ui.render_app_sidebar(user)

    raw_view = st.session_state.get("app_view") or (
        "admin" if user.is_admin else "welcome"
    )
    if raw_view == "settings":
        raw_view = st.session_state.get("settings_section") or "ai_configuration"

    view = normalize_view(raw_view)
    if view in ADMIN_ONLY_VIEWS and not user.is_admin:
        auth_ui.deny_access(view, user=user)
        view = "welcome"
    st.session_state.app_view = view

    if view == "wizard":
        if st.button("Start Fresh", key="toolbar_start_fresh"):
            reset_active_analysis(go_home=True)
        _render_wizard()
        return

    if view == "admin":
        admin_console.render_admin_console(user)
        return

    if view == "shape_detection":
        from shape_detection_storage import (
            ensure_default_feature_policy,
            shape_detection_allowed,
        )
        from shape_detection_ui import render_shape_detection_page

        try:
            ensure_default_feature_policy()
        except Exception:  # noqa: BLE001
            pass
        allowed, deny_msg = shape_detection_allowed(user)
        if not allowed:
            from ui_helpers import render_page_hero

            render_page_hero("Shape Detection", "Testing Phase")
            st.error(deny_msg or "Shape Detection is unavailable.")
            if st.button("Back to Dashboard", key="shape_denied_back"):
                navigate_to("welcome")
                st.rerun()
            return
        render_shape_detection_page(user)
        return

    if view in {
        "history",
        "ai_configuration",
        "diagnostics",
        "account",
        "api_keys",
    }:
        view_panel(view, user)
        return

    view_welcome(user)


if __name__ == "__main__":
    main()
