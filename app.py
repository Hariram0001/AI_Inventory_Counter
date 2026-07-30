"""AI Inventory Counter — redesigned Streamlit wizard UI."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

# Constants first — zero Streamlit / UI dependencies (safe under Streamlit re-entry)
from app_constants import (
    PHOTO_REL_INTERNAL_TO_DISPLAY,
    SETTINGS_SECTION_LABELS,
    SETTINGS_SECTIONS,
    STAGES,
    get_settings_section_from_label,
)

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import config
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
    validate_upload,
)
from inventory_config import (
    FIXED_PHOTO_RELATIONSHIP,
    PHOTO_RELATIONSHIP_NOTE,
    SELECTABLE_INVENTORY_KEY,
    form_updates_from_recommendation,
    inventory_display_name,
    is_inventory_selectable,
    resolve_recommended_model,
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
    PAGE_SIZE,
    filter_detections,
    format_detection_option,
    index_of_detection,
    paginate,
    step_detection_id,
)
from overlap import build_consensus_detections
from schemas import ConsensusResult, Detection, InferenceResult, ModelConfig
from ui_helpers import (
    default_form,
    inject_css,
    leave_settings,
    navigate_to,
    normalize_stage,
    normalize_view,
    open_settings,
    render_empty_state,
    render_nav_buttons,
    render_page_toolbar,
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
        "compare_side_by_side": False,
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
    # Old top-level history/diagnostics → settings sections
    raw_view = st.session_state.get("app_view")
    if raw_view == "history":
        view = "settings"
        st.session_state.settings_section = "history"
    elif raw_view in {"diagnostics", "setup"}:
        view = "settings"
        st.session_state.settings_section = "diagnostics"
    st.session_state.app_view = view
    if st.session_state.get("settings_section") not in SETTINGS_SECTIONS:
        st.session_state.settings_section = "ai_configuration"


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


def _apply_recommended_setup(*, inventory_key: str | None = None) -> dict[str, Any]:
    """Resolve and persist Recommended AI Setup for the selected inventory."""
    key = inventory_key if inventory_key is not None else _resolved_inventory()
    _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)
    if not key or not is_inventory_selectable(key):
        _form_set(
            recommended_setup_resolved=False,
            recommended_model_name="",
            recommended_setup_error="",
            selected_models=[],
        )
        return {"ok": False, "error": "Select Fence Panels to continue."}
    resolved = resolve_recommended_model(
        key,
        _all_models(),
        getattr(config, "INVENTORY_MODEL_RECOMMENDATIONS", {}),
        allow_demo=bool(config.DEMO_MODE),
    )
    _form_set(**form_updates_from_recommendation(resolved))
    return resolved


def _all_models() -> list[ModelConfig]:
    return merge_session_models(load_models_from_file(), st.session_state.session_models)


def _enabled_models() -> list[ModelConfig]:
    """Enabled/valid models for Settings and internal use (may include demo when DEMO_MODE)."""
    return get_enabled_valid_models(_all_models())


def _analysis_models() -> list[ModelConfig]:
    """Models shown in the Analysis selector — never silent demo substitutes when live."""
    return get_selectable_analysis_models(
        _all_models(),
        _resolved_inventory() or SELECTABLE_INVENTORY_KEY,
        allow_demo=bool(config.DEMO_MODE),
    )


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
    st.error(message)
    if detail:
        with st.expander("Technical details"):
            st.code(detail)


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
    config.reload_settings()
    model = _primary_workflow_model()
    connected = bool(config.DEMO_MODE or api_key_configured())
    detection_mode = "Demo" if config.DEMO_MODE else "Live Workflow"
    response_source = "demo source" if config.DEMO_MODE else "live Roboflow"
    return {
        "connected": connected,
        "connection_label": "Connected" if connected else "Not Connected",
        "provider": "Roboflow",
        "workspace": (model.workspace_name if model and model.workspace_name else "—"),
        "workflow_name": (model.name if model else "—"),
        "workflow_id": (model.workflow_id if model and model.workflow_id else "—"),
        "model_id": (model.model_id if model and model.model_id else "—"),
        "kind": (model.kind if model else "—"),
        "detection_mode": detection_mode,
        "response_source": response_source,
        "api_key": "Configured" if api_key_configured() else "Missing",
        "source_label": "Local project settings (.env + models.json)",
        "models_path": str(config.MODELS_JSON_PATH.name),
    }


def render_configuration_summary(*, show_actions: bool = True) -> dict[str, Any]:
    snap = _config_snapshot()
    status_html = render_status_badge(snap["connected"], "Connected", "Not Connected")
    st.markdown(
        f"""
        <div class="aic-card">
          <div style="margin-bottom:0.55rem;"><b>Connection Status:</b> {status_html}</div>
          <div><b>Provider:</b> {snap["provider"]}</div>
          <div><b>Workspace:</b> {snap["workspace"]}</div>
          <div><b>Workflow:</b> {snap["workflow_name"]}</div>
          <div><b>Workflow ID:</b> {snap["workflow_id"]}</div>
          <div><b>Detection Mode:</b> {snap["detection_mode"]}</div>
          <div><b>Response Source:</b> {snap["response_source"]}</div>
          <div><b>API Key:</b> {snap["api_key"]}</div>
          <p class="aic-muted" style="margin-top:0.75rem;margin-bottom:0;">
            Configuration is automatically loaded from the active workflow and local project settings.
            <br/>Source: {snap["source_label"]}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_actions:
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Refresh Configuration", use_container_width=True, key="cfg_refresh"):
                config.reload_settings()
                st.session_state.config_refresh_nonce = (
                    int(st.session_state.get("config_refresh_nonce", 0)) + 1
                )
                st.rerun()
        with a2:
            if st.button("Advanced Settings", use_container_width=True, key="cfg_adv_toggle"):
                st.session_state.open_advanced_settings = not bool(
                    st.session_state.get("open_advanced_settings")
                )
                st.rerun()
    return snap


# ---------------------------------------------------------------------------
# Welcome / Settings sections
# ---------------------------------------------------------------------------


def view_welcome() -> None:
    render_page_toolbar(
        mode="home",
        on_settings=lambda: open_settings(section="ai_configuration"),
    )
    st.markdown('<div class="aic-hero-title">AI Inventory Counter</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="aic-hero-sub">Upload inventory photos and use AI to identify, '
        "count, review, and save detected items.</div>",
        unsafe_allow_html=True,
    )
    if st.button("Get Started", type="primary", use_container_width=False, key="get_started"):
        reset_active_analysis(go_home=False, start_wizard=True)


def _render_history_section() -> None:
    st.markdown("#### Inventory History")
    st.caption("Previously saved inventory analyses.")

    try:
        initialize_database()
        rows = get_inventory_history()
    except DatabaseError as exc:
        _error_box("Could not load history.", str(exc))
        return

    if not rows:
        render_empty_state(
            "No inventory history yet",
            "Saved analyses will appear here after you complete a review and save.",
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

    f1, f2 = st.columns(2)
    yards = ["All"] + sorted(
        {str(v) for v in df.get("yard", pd.Series(dtype=str)).dropna().unique()}
    )
    types = ["All"] + sorted(
        {str(v) for v in df.get("inventory_type", pd.Series(dtype=str)).dropna().unique()}
    )
    with f1:
        yard_f = st.selectbox("Filter by location", yards, key="hist_yard")
    with f2:
        type_f = st.selectbox("Filter by inventory type", types, key="hist_type")

    filtered = df
    if yard_f != "All":
        filtered = filtered[filtered["yard"] == yard_f]
    if type_f != "All":
        filtered = filtered[filtered["inventory_type"] == type_f]

    if filtered.empty:
        st.info("No records match the selected filters.")
    else:
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
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="inventory_count_history.csv",
            mime="text/csv",
            key="hist_csv",
        )
        with st.expander("View full record details"):
            st.dataframe(filtered, hide_index=True, use_container_width=True)


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
        result["message"] = msg
        result["details"]["connectivity"] = msg

        model = _primary_workflow_model()
        if model is None:
            result["workflow"] = "Missing"
            result["ok"] = False
            result["message"] = "No enabled workflow/model is configured in models.json."
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
            result["response_source"] = "demo_mock"
            result["message"] = "Demo Mode active — live API not required."
            result["parser_status"] = "skipped_demo"
            result["processing_time"] = time.perf_counter() - started
            return result

        if not ok:
            result["workflow"] = "Unavailable"
            result["ok"] = False
            result["processing_time"] = time.perf_counter() - started
            return result

        image_bytes, image_name = _ai_config_test_image_bytes()
        if not image_bytes:
            result["ok"] = True
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
        result["message"] = (
            f"Live probe complete: raw={inference.raw_prediction_count}, "
            f"normalized={inference.normalized_prediction_count}, "
            f"final={inference.final_count}."
        )
        result["details"]["annotated"] = bool(inference.annotated_image_bytes)
    except DetectorError as exc:
        result["ok"] = False
        result["auth"] = result.get("auth") or "Failed"
        result["message"] = str(exc)[:400]
        result["parser_status"] = "api_error"
        st.session_state.last_diag_error = result["message"]
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["auth"] = "Failed"
        result["message"] = str(exc)[:400]
        result["parser_status"] = "unexpected_error"
        st.session_state.last_diag_error = result["message"]
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
    st.markdown("#### AI Configuration")
    st.caption(
        "Configuration is automatically loaded from the active workflow and local project settings."
    )
    _ensure_selected_models()

    st.markdown("##### Active Configuration")
    render_configuration_summary(show_actions=True)

    from catalog_ui import render_model_catalog_section

    def _catalog_model_test(model: ModelConfig) -> dict[str, Any]:
        """Run a single-model settings probe without touching wizard uploads."""
        from model_adapters import InferenceOptions, get_adapter

        data, name = _ai_config_test_image_bytes()
        out: dict[str, Any] = {
            "model_key": model.key or model.name,
            "ok": False,
            "auth": "Configured" if api_key_configured() else "Missing",
            "response_source": None,
            "raw_prediction_count": 0,
            "normalized_prediction_count": 0,
            "detected_classes": [],
            "processing_time": 0.0,
            "parser_status": "not_run",
            "message": "",
            "error_message": None,
            "annotated_preview": None,
        }
        if (model.kind or "").lower() != "local" and not api_key_configured() and not config.DEMO_MODE:
            out["message"] = "API key not configured."
            return out
        if not data:
            out["message"] = "Upload a probe image or add data/ai_config_test_image.jpg."
            return out
        try:
            prepared = load_image_from_bytes(data, name or "probe.jpg")
            adapter = get_adapter(model)
            opts = InferenceOptions(
                prompt=config.inventory_detection_prompt("Fence Panel"),
                confidence_threshold=float(model.default_confidence or 0.25),
                iou_threshold=float(model.default_iou or 0.5),
            )
            mir = adapter.predict(prepared, opts)
            out.update(
                {
                    "ok": bool(mir.success),
                    "response_source": mir.response_source,
                    "raw_prediction_count": mir.raw_count,
                    "normalized_prediction_count": len(mir.detections),
                    "detected_classes": list(mir.classes),
                    "processing_time": mir.processing_time_seconds,
                    "parser_status": "ok" if mir.success else (mir.error_type or "failed"),
                    "message": mir.error_message or ("OK" if mir.success else "Failed"),
                    "error_message": mir.error_message,
                    "annotated_preview": mir.annotated_image_bytes,
                }
            )
        except Exception as exc:  # noqa: BLE001
            out["message"] = str(exc)[:300]
            out["error_message"] = out["message"]
        return out

    render_model_catalog_section(
        run_model_test=_catalog_model_test,
        get_test_image_bytes=_ai_config_test_image_bytes,
    )

    with st.expander("Legacy registry table", expanded=False):
        models = _all_models()
        summary = summarize_models(models)
        if summary:
            st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
        st.caption(
            "Demo/local classical entries are excluded from the live Analysis selector when "
            "DEMO_MODE is false. Local Picket Counter is a NumPy/PIL heuristic in picket_counter.py, "
            "not a Roboflow model."
        )

    st.markdown("##### Configuration probe image")
    st.caption(
        "Optional. Used only by Test AI Configuration — never replaces inventory uploads. "
        "If omitted, `data/ai_config_test_image.jpg` is used when present."
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

    st.markdown("##### Test Models")
    if st.button("Test AI Configuration", type="primary", key="cfg_test_btn"):
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
                - API key configured: {"Yes" if test.get("api_key_configured") else "No"}
                - Authentication: {test.get("auth")}
                - Workspace: {(test.get("details") or {}).get("workspace")}
                - Workflow: {test.get("workflow")}
                - Demo mode: {"On" if test.get("demo_mode") else "Off"}
                - Response source: {test.get("response_source")}
                - Test image: {(test.get("test_image") or {}).get("name") or "(connectivity only)"}
                - Raw prediction count: {test.get("raw_prediction_count")}
                - Normalized prediction count: {test.get("normalized_prediction_count")}
                - Detected classes: {", ".join(test.get("detected_classes") or []) or "(none)"}
                - Parser status: {test.get("parser_status")}
                - Processing time: {float(test.get("processing_time") or 0):.2f} seconds
                """
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
            {"Model": name, "Status": info.get("status"), "When": info.get("when"), "Notes": info.get("message", "")}
            for name, info in history_map.items()
            if isinstance(info, dict)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No model tests run in this session yet.")

    st.markdown("##### Advanced Defaults")
    _render_advanced_settings()

    _render_sample_library_settings()


def _render_sample_library_settings() -> None:
    """Compact read-only Built-in Sample Library status (no full gallery)."""
    st.markdown("##### Built-in Sample Library")
    status = load_sample_library(force_reload=True)
    st.markdown(
        f"""
        - **Sample directory:** {"OK" if status.directory_exists else "Missing"}
        - **Manifest:** {"OK" if status.manifest_valid else ("Invalid" if status.manifest_exists else "Missing")}
        - **Valid samples:** {status.valid_count}
        - **Enabled samples:** {status.enabled_count}
        - **Missing files:** {len(status.missing_files)}
        - **Invalid files:** {len(status.invalid_files)}
        - **Duplicate IDs:** {len(status.duplicate_ids)}
        """
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


def _render_diagnostics_section() -> None:
    st.markdown("#### Diagnostics")
    st.caption(
        "Technical runtime and troubleshooting only. "
        "Model registry and model tests live under AI Configuration; "
        "saved counts live under Inventory History."
    )

    snap = _config_snapshot()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("API Key", snap["api_key"])
    with c2:
        st.metric("Demo Mode", "On" if config.DEMO_MODE else "Off")
    with c3:
        st.metric("Connection", snap["connection_label"])

    st.caption(
        "Demo Mode uses stored sample predictions from `sample_responses/mock_detection.json` "
        "instead of calling the live Roboflow workflow."
        if config.DEMO_MODE
        else "Demo Mode is off: detection uses live Roboflow or local adapters only — "
        "mock predictions are not substituted."
    )

    st.markdown("##### Inference SDK probe")
    st.caption(
        "Shows the real import/client status. Exceptions are not masked as "
        "'inference-sdk is not installed'."
    )
    if st.button("Run inference SDK / Roboflow probe", key="diag_sdk_probe"):
        # Invalidate any session-cached inference results so a fresh client is used.
        st.session_state.inference_cache = {}
        probe: dict[str, Any] = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "demo_mode": bool(config.DEMO_MODE),
            "api_key_configured": bool(config.ROBOFLOW_API_KEY),
            "api_url": config.ROBOFLOW_API_URL,
            "workspace_env": getattr(config, "ROBOFLOW_WORKSPACE", ""),
            "workflow_id_env": getattr(config, "ROBOFLOW_WORKFLOW_ID", ""),
            "inference_sdk_import": None,
            "inference_sdk_version": None,
            "inference_sdk_file": None,
            "client_created": False,
            "client_error": None,
            "connectivity_ok": None,
            "connectivity_message": None,
            "roboflow_probe_response": None,
            "active_model_workspace": None,
            "active_model_workflow_id": None,
            "traceback": None,
        }
        model = _primary_workflow_model()
        if model is not None:
            probe["active_model_workspace"] = model.workspace_name
            probe["active_model_workflow_id"] = model.workflow_id
            probe["active_model_name"] = model.name
            probe["active_model_kind"] = model.kind
        try:
            import inference_sdk

            probe["inference_sdk_import"] = "ok"
            probe["inference_sdk_version"] = getattr(
                inference_sdk, "__version__", "unknown"
            )
            probe["inference_sdk_file"] = getattr(inference_sdk, "__file__", None)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            probe["inference_sdk_import"] = f"{type(exc).__name__}: {exc}"
            probe["traceback"] = traceback.format_exc()
            st.session_state.last_diag_error = probe["inference_sdk_import"]
            st.session_state.diag_sdk_probe = probe
            st.rerun()

        try:
            det = RoboflowDetector()
            client = det._get_client()
            probe["client_created"] = client is not None
            if client is not None and hasattr(client, "get_server_info"):
                try:
                    info = client.get_server_info()
                    if isinstance(info, dict):
                        probe["roboflow_probe_response"] = {
                            k: info[k] for k in list(info)[:20]
                        }
                    else:
                        probe["roboflow_probe_response"] = {
                            "type": type(info).__name__,
                            "repr": repr(info)[:500],
                        }
                except Exception as probe_exc:  # noqa: BLE001
                    traceback.print_exc()
                    probe["roboflow_probe_response"] = {
                        "error": f"{type(probe_exc).__name__}: {probe_exc}"
                    }
            ok, msg = det.test_connectivity()
            probe["connectivity_ok"] = ok
            probe["connectivity_message"] = msg
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            probe["client_error"] = f"{type(exc).__name__}: {exc}"
            probe["traceback"] = traceback.format_exc()
            st.session_state.last_diag_error = probe["client_error"]
        st.session_state.diag_sdk_probe = probe
        st.rerun()

    probe_state = st.session_state.get("diag_sdk_probe")
    if isinstance(probe_state, dict):
        st.json(probe_state)
        if probe_state.get("traceback"):
            with st.expander("Probe traceback", expanded=True):
                st.code(probe_state["traceback"])

    st.markdown(
        f"""
        <div class="aic-card">
          <div><b>Provider:</b> {snap["provider"]}</div>
          <div><b>Workspace:</b> {snap["workspace"]}</div>
          <div><b>Workflow ID:</b> {snap["workflow_id"]}</div>
          <div><b>Detection mode:</b> {snap["detection_mode"]}</div>
          <div><b>Response source:</b> {snap["response_source"]}</div>
          <div><b>API status:</b> {masked_api_key_status()}</div>
          <div><b>Config file:</b> {snap["models_path"]}</div>
          <div><b>Database:</b> {config.DB_PATH.name}</div>
          <div><b>Python:</b> {sys.version.split()[0]}</div>
          <div><b>Streamlit health:</b> /_stcore/health</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Manage models in Settings → AI Configuration.")

    sample_warns = sample_library_diagnostics_warnings()
    if sample_warns:
        st.markdown("##### Sample library warnings")
        for w in sample_warns[:12]:
            st.warning(w)
    else:
        st.caption("Sample library: no warnings.")

    if st.button("Test API connectivity", key="diag_test"):
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

    if st.session_state.get("last_diag_error"):
        with st.expander("Last API / parser error", expanded=True):
            st.code(st.session_state.last_diag_error)

    results: list[InferenceResult] = st.session_state.analysis_results or []
    with st.expander("Last request summary (this session)", expanded=False):
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
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    debug_path = config.DATA_DIR / "debug" / "last_live_response.json"
    shape_path = config.DATA_DIR / "last_live_response_shape.json"
    with st.expander("Sanitized raw-response details", expanded=False):
        if shape_path.exists():
            st.caption(str(shape_path))
            try:
                st.json(json.loads(shape_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                st.caption("Could not parse response shape file.")
        elif debug_path.exists():
            st.caption(f"Response dump present at {debug_path.name} (open on disk; not echoed here).")
        else:
            st.caption("No saved live response yet.")

    with st.expander("Environment", expanded=False):
        st.write(f"**DEMO_MODE:** {config.DEMO_MODE}")
        st.write(f"**ROBOFLOW_API_URL:** {config.ROBOFLOW_API_URL}")
        st.write(f"**DATA_DIR:** `{config.DATA_DIR}`")
        st.write(f"**API key configured:** {'Yes' if api_key_configured() else 'No'}")

    with st.expander("Runtime packages", expanded=False):
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
        st.json(pkgs)


def view_settings() -> None:
    render_page_toolbar(mode="settings", on_back=leave_settings)
    st.markdown("## Settings")
    st.caption("Manage AI configuration, saved inventory records, and application diagnostics.")

    labels = [SETTINGS_SECTION_LABELS[s] for s in SETTINGS_SECTIONS]
    current = st.session_state.get("settings_section", "ai_configuration")
    if current not in SETTINGS_SECTIONS:
        current = "ai_configuration"
    try:
        index = SETTINGS_SECTIONS.index(current)
    except ValueError:
        index = 0

    choice = st.radio(
        "Settings section",
        labels,
        index=index,
        horizontal=True,
        label_visibility="collapsed",
        key="settings_section_radio",
    )
    st.session_state.settings_section = get_settings_section_from_label(choice)

    section = st.session_state.settings_section
    st.write("")
    if section == "ai_configuration":
        _render_ai_configuration_section()
    elif section == "history":
        _render_history_section()
    else:
        _render_diagnostics_section()


# ---------------------------------------------------------------------------
# Stage 1 — Inventory Setup (selectable Fence Panels only)
# ---------------------------------------------------------------------------


def stage_setup() -> None:
    render_stepper("setup")
    st.subheader("Inventory Setup")
    st.caption("Choose the inventory type you are counting.")

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

    # Responsive grid — shared card structure; no horizontal/nested scrolling.
    n_cols = 4 if len(INVENTORY_TYPES) >= 4 else 3
    cols = st.columns(n_cols)
    for i, inv in enumerate(INVENTORY_TYPES):
        with cols[i % n_cols]:
            selectable = is_inventory_selectable(inv)
            display = inventory_display_name(inv) if selectable else inv
            if selectable:
                selected = current == inv
                # Primary (red) selected styling via Streamlit; same min-height as unavailable cards
                label = f"✓ {display}" if selected else display
                if st.button(
                    label,
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                    key=f"inv_tile_{inv}",
                ):
                    _form_set(inventory_choice=inv)
                    _apply_recommended_setup(inventory_key=inv)
                    st.rerun()
            else:
                # Same card shell as selectable tiles; red unavailable indicator + Coming Soon
                st.markdown(
                    f"""
                    <div class="aic-inv-card aic-inv-card--unavailable" title="Coming Soon"
                         aria-disabled="true">
                      <span class="aic-inv-unavailable" title="Coming Soon"
                            aria-label="Coming Soon">⊘</span>
                      <div class="aic-inv-card-title">{inv}</div>
                      <div class="aic-inv-soon">Coming Soon</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    inv_choice = _form_get("inventory_choice", "") or ""
    if inv_choice and not is_inventory_selectable(inv_choice):
        _form_set(inventory_choice="")
        inv_choice = ""

    st.markdown(
        f'<p class="aic-note">{PHOTO_RELATIONSHIP_NOTE}</p>',
        unsafe_allow_html=True,
    )

    yard_ok = bool(_resolved_yard())
    inv_ok = bool(inv_choice) and is_inventory_selectable(inv_choice)
    if not inv_ok:
        st.caption("Select Fence Panels to continue.")

    def _next() -> None:
        if not _resolved_yard():
            st.error("Location is required.")
            return
        if not is_inventory_selectable(_form_get("inventory_choice", "")):
            st.error("Select Fence Panels to continue.")
            return
        _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)
        _apply_recommended_setup(inventory_key=SELECTABLE_INVENTORY_KEY)
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


def _render_sample_images_tab() -> None:
    """Compact gallery of project-bundled sample images (Fence Panels)."""
    st.caption(
        "Built-in samples ship with the app. Selecting a card does not add it — "
        "use **Add Selected Photos** or preview then **Add This Photo**."
    )
    samples = list_enabled_samples(inventory_key=SELECTABLE_INVENTORY_KEY)
    lib = load_sample_library()
    if lib.warnings:
        # Compact notice only; full detail lives in Settings / Diagnostics
        st.caption(f"Sample library notes: {len(lib.warnings)} warning(s). See Settings.")

    if not samples:
        render_empty_state(
            "No sample images available yet",
            "Add JPEG/PNG files under assets/sample_images/ and register them in manifest.json.",
        )
        return

    selected_ids: set[str] = set(st.session_state.get("sample_selected_ids") or [])
    preview_id = st.session_state.get("sample_preview_id")

    # Thumbnail grid (3 columns)
    cols = st.columns(3)
    for i, sample in enumerate(samples):
        with cols[i % 3]:
            try:
                data = read_sample_bytes(sample)
            except OSError:
                st.warning(sample.title)
                continue
            st.image(data, use_container_width=True)
            st.markdown(f"**{sample.title}**")
            st.caption(
                f"{sample.description[:80]}{'…' if len(sample.description) > 80 else ''}\n\n"
                f"{sample.width}×{sample.height} · Fence Panels"
            )
            checked = st.checkbox(
                "Select",
                value=sample.id in selected_ids,
                key=f"sample_sel_{sample.id}",
            )
            if checked:
                selected_ids.add(sample.id)
            else:
                selected_ids.discard(sample.id)
            if st.button("Preview", key=f"sample_prev_{sample.id}", use_container_width=True):
                st.session_state.sample_preview_id = sample.id
                st.rerun()

    st.session_state.sample_selected_ids = list(selected_ids)

    a1, a2 = st.columns(2)
    with a1:
        if st.button(
            "Add Selected Photos",
            type="primary",
            use_container_width=True,
            key="sample_add_selected",
            disabled=not selected_ids,
        ):
            added = 0
            for sid in list(selected_ids):
                sample = get_sample_by_id(sid)
                if sample is None:
                    continue
                try:
                    data = read_sample_bytes(sample)
                except OSError as exc:
                    st.error(f"{sample.filename}: {exc}")
                    continue
                err = _add_image_bytes(
                    data,
                    sample.filename,
                    source="sample",
                    mime_type=sample.mime_type,
                    sample_id=sample.id,
                )
                if err:
                    st.warning(err)
                else:
                    added += 1
            if added:
                st.success(f"Added {added} sample photo(s).")
                st.session_state.sample_selected_ids = []
                st.rerun()
    with a2:
        if st.button("Clear selection", use_container_width=True, key="sample_clear_sel"):
            st.session_state.sample_selected_ids = []
            for sample in samples:
                st.session_state[f"sample_sel_{sample.id}"] = False
            st.rerun()

    if preview_id:
        sample = get_sample_by_id(str(preview_id))
        if sample is None:
            st.session_state.sample_preview_id = None
        else:
            st.divider()
            st.markdown(f"### Preview · {sample.title}")
            try:
                data = read_sample_bytes(sample)
                st.image(data, use_container_width=True)
            except OSError as exc:
                st.error(str(exc))
                return
            inv_label = inventory_display_name(sample.app_inventory_key) or sample.inventory_type
            st.markdown(
                f"""
                - **Description:** {sample.description or '—'}
                - **Dimensions:** {sample.width}×{sample.height}
                - **Source:** Built-in Sample
                - **Inventory compatibility:** {inv_label}
                """
            )
            p1, p2 = st.columns(2)
            with p1:
                if st.button(
                    "Add This Photo",
                    type="primary",
                    use_container_width=True,
                    key="sample_add_preview",
                ):
                    err = _add_image_bytes(
                        data,
                        sample.filename,
                        source="sample",
                        mime_type=sample.mime_type,
                        sample_id=sample.id,
                    )
                    if err:
                        st.warning(err)
                    else:
                        st.success("Sample photo added.")
                        st.rerun()
            with p2:
                if st.button("Close preview", use_container_width=True, key="sample_close_prev"):
                    st.session_state.sample_preview_id = None
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


def stage_photos() -> None:
    render_stepper("photos")
    _form_set(photo_relationship=FIXED_PHOTO_RELATIONSHIP)

    inv = _resolved_inventory()
    inv_label = inventory_display_name(inv) if inv else "(not selected)"
    n_photos = len(st.session_state.uploaded_images)
    status_txt = "Ready" if n_photos >= 1 else "Selected"
    photos_line = f"<div>Photos: {n_photos}</div>" if n_photos else ""

    head_l, head_r = st.columns([2.4, 1.1], vertical_alignment="top")
    with head_l:
        st.subheader("Add Photos")
        st.caption("Upload or capture inventory photos.")
    with head_r:
        st.markdown(
            f"""
            <div class="aic-photos-status" title="Selected inventory status">
              <span class="dot"></span><b>{inv_label}</b>
              {photos_line}
              <div>Status: {status_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nonce = st.session_state.get("uploader_nonce", 0)
    tab_upload, tab_camera, tab_samples = st.tabs(
        ["Upload Images", "Use Camera", "Sample Images"]
    )

    max_mb = max(1, int(config.MAX_UPLOAD_BYTES / (1024 * 1024)))
    with tab_upload:
        st.caption(
            f"JPG, JPEG, or PNG · multiple files supported · max {max_mb} MB per file."
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

    with tab_camera:
        st.caption(
            "Capture a still photo. Review the preview, then press **Add This Photo**. "
            "If the camera does not open, check browser permissions for this site."
        )
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

        pending = st.session_state.get("pending_camera")
        if isinstance(pending, dict) and pending.get("data"):
            st.markdown("**Preview (not yet added)**")
            st.image(pending["data"], use_container_width=True)
            st.caption(
                f"{pending['name']} · {pending['width']}×{pending['height']} · "
                f"{_format_bytes(pending['size_bytes'])}"
            )
            c_add, c_retake = st.columns(2)
            with c_add:
                if st.button("Add This Photo", type="primary", use_container_width=True, key="cam_add"):
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
                    else:
                        st.success("Camera photo added.")
                    st.rerun()
            with c_retake:
                if st.button("Retake / Discard", use_container_width=True, key="cam_retake"):
                    st.session_state.pending_camera = None
                    st.session_state.uploader_nonce = int(nonce) + 1
                    st.rerun()

    with tab_samples:
        _render_sample_images_tab()

    images = st.session_state.uploaded_images
    if not images:
        render_empty_state(
            "No photos added yet",
            "Upload images or use the camera to continue.",
        )
    else:
        top_l, top_r = st.columns([2, 1])
        with top_l:
            st.markdown(f"**{len(images)} photo(s) ready**")
        with top_r:
            if st.button("Clear all", use_container_width=True, key="clear_photos"):
                st.session_state.uploaded_images = []
                st.session_state.pending_camera = None
                st.session_state.uploader_nonce = int(nonce) + 1
                st.rerun()

        for img in list(images):
            c1, c2, c3 = st.columns([1.2, 2.5, 0.8])
            with c1:
                st.image(img["data"], use_container_width=True)
            with c2:
                src = img.get("source") or "upload"
                st.markdown(f"**{img['name']}**")
                st.caption(
                    f"{img['width']} × {img['height']} px · {_format_bytes(img['size_bytes'])} · {src}"
                )
            with c3:
                if st.button("Remove", key=f"rm_img_{img['id']}"):
                    st.session_state.uploaded_images = [
                        x for x in st.session_state.uploaded_images if x["id"] != img["id"]
                    ]
                    st.rerun()

    st.divider()

    can_next = len(st.session_state.uploaded_images) >= 1 and is_inventory_selectable(
        _form_get("inventory_choice", "")
    )

    def _next() -> None:
        if not st.session_state.uploaded_images:
            st.error("Add at least one valid image.")
            return
        if not is_inventory_selectable(_form_get("inventory_choice", "")):
            st.error("Select Fence Panels on Inventory Setup before analyzing.")
            return
        resolved = _apply_recommended_setup(inventory_key=SELECTABLE_INVENTORY_KEY)
        if not resolved.get("ok"):
            st.error(
                resolved.get("error")
                or "No valid AI model is configured for Fence Panels."
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
    render_empty_state(
        "Analysis did not complete successfully",
        "The live request failed or returned an invalid workflow response. "
        "This is not the same as finding zero inventory items.",
    )
    for fail in failures:
        st.error(fail)
    for r in results:
        if r.error_message or r.errors:
            st.error(r.error_message or "; ".join(r.errors))
    with st.expander("View Technical Details", expanded=True):
        for r in results:
            st.json(r.summary_dict())
    if st.button("Open AI Settings", key="af_settings"):
        open_settings(section="ai_configuration")


def _render_zero_detection_empty(results: list[InferenceResult]) -> None:
    render_empty_state(
        "No inventory items were detected",
        "The live request succeeded, but the model found no matching objects "
        "after filtering. This is a genuine zero-detection result.",
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Try Another Photo", use_container_width=True, key="zd_photo"):
            navigate_to("wizard", stage="photos")
    with a2:
        if st.button("Adjust Detection Sensitivity", use_container_width=True, key="zd_sens"):
            st.session_state.open_advanced_settings = True
            open_settings(section="ai_configuration")
    with a3:
        if st.button("Continue to Review", type="primary", use_container_width=True, key="zd_rev"):
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
    st.subheader("Analyze")

    images = st.session_state.uploaded_images
    inference_ui = _form_get("inference_mode", "Whole Image")
    inference_mode = _inference_api_name(inference_ui)
    config_ok = _ai_config_is_valid()
    inv_label = inventory_display_name(_resolved_inventory()) or "—"
    ai_label = "Connected" if config_ok else "Needs attention"

    # Ensure backend defaults exist without rendering recommendation UI.
    if _resolved_inventory():
        _apply_recommended_setup(inventory_key=_resolved_inventory())

    from catalog_ui import format_model_option
    from model_catalog import get_all_catalog_models, remove_stale_model_selection

    selectable = _analysis_models()
    model_names = [m.name for m in selectable]
    # Compare peers: enabled/valid Roboflow + confirmed local inference (not demo fixtures).
    compare_models = compare_peer_models(selectable)
    compare_names = [m.name for m in compare_models]
    compare_available = len(compare_names) >= COMPARE_MIN_MODELS

    cleaned, stale_note = remove_stale_model_selection(
        _form_get("selected_models") or [],
        inventory_key=_resolved_inventory() or "Fence Panel",
    )
    if stale_note:
        st.info(stale_note)
    if cleaned != (_form_get("selected_models") or []):
        _form_set(selected_models=cleaned)

    if not model_names:
        st.error("No compatible live model is configured.")
        if st.button("Open AI Settings", key="analyze_missing_model_settings"):
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

    st.markdown(f"**Detecting:** {inv_label}")
    st.caption(f"Photos: {len(images)}")

    if not compare_available:
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
    if mode_ui == "Single Model":
        single_prev = prev[0] if prev and prev[0] in model_names else model_names[0]
        choice = st.selectbox(
            "Model",
            options=model_names,
            index=model_names.index(single_prev),
            format_func=lambda n: format_model_option(
                next(m for m in selectable if m.name == n),
                entries_by_name,
            ),
            key="analyze_single_model",
            help="Compatible models (name · source · task · status).",
        )
        selected_names = [choice] if choice in model_names else []
        if selected_names:
            st.caption(f"Model: **{selected_names[0]}** · Detecting: **{inv_label}**")
    else:
        compare_prev = sanitize_compare_selection(prev, compare_names)
        # Do not auto-select every / any models — preserve prior valid compare picks only.
        selected_names = st.multiselect(
            "Models (2–3)",
            options=compare_names,
            default=compare_prev,
            max_selections=COMPARE_MAX_MODELS,
            format_func=lambda n: format_model_option(
                next(m for m in compare_models if m.name == n),
                entries_by_name,
            ),
            key="analyze_compare_models",
        )
        selected_names = sanitize_compare_selection(selected_names, compare_names)
        cmp_errs = validate_compare_selection(selected_names, compare_names)
        if cmp_errs:
            st.caption(cmp_errs[0])

    _form_set(selected_models=selected_names)
    selected_models = [m for m in selectable if m.name in selected_names]
    # Preserve selection order from the multiselect / selectbox.
    if mode_ui == "Compare Models":
        order = {n: i for i, n in enumerate(selected_names)}
        selected_models.sort(key=lambda m: order.get(m.name, 999))

    detect_prompt = (_form_get("prompt") or "").strip() or config.inventory_detection_prompt(
        _resolved_inventory()
    )
    _form_set(prompt=detect_prompt, class_override=detect_prompt)

    st.caption(f"Inventory prompts → model: {detect_prompt}")
    if mode_ui == "Compare Models":
        st.caption(comparison_run_caption(len(images), len(selected_models)))
    else:
        st.caption(
            f"{len(images)} photos × {len(selected_models)} model = "
            f"{max(0, len(images) * len(selected_models))} analysis runs"
        )
    st.markdown(
        f'<p class="aic-analyze-status">AI: {ai_label}</p>',
        unsafe_allow_html=True,
    )

    status = st.session_state.analysis_status
    if status == "complete":
        results = st.session_state.analysis_results or []
        failures = st.session_state.analysis_failures or []
        total = sum(r.final_count for r in results)
        pipeline_fault = any(
            (r.error_type in {"empty_workflow_output", "api_error"} or r.errors)
            for r in results
        )
        if pipeline_fault or failures:
            _render_analysis_failure_state(results, failures)
        elif total == 0 and results:
            _render_zero_detection_empty(results)
        else:
            sources = sorted({(r.source or "") for r in results if r.source})
            if config.DEMO_MODE:
                source_note = " (demo)"
            elif sources == {"local_classical"}:
                source_note = " (local picket counter)"
            elif "live_roboflow" in sources and "local_classical" in sources:
                source_note = " (live Roboflow + local)"
            elif "live_roboflow" in sources:
                source_note = " (live Roboflow)"
            else:
                source_note = ""
            st.success(f"Analysis completed successfully{source_note}")
            if st.button("Continue to Review", type="primary", use_container_width=True):
                navigate_to("wizard", stage="review")
        render_nav_buttons(back_stage="photos", key_prefix="an_done")
        return

    if status == "partial":
        st.warning("Analysis finished with partial results. Some images or models failed.")
        for fail in st.session_state.analysis_failures:
            st.error(fail)
        results = st.session_state.analysis_results or []
        total = sum(r.final_count for r in results) if results else -1
        pipeline_fault = any(
            (r.error_type in {"empty_workflow_output", "api_error"} or r.errors)
            for r in results
        )
        if results and pipeline_fault and total == 0:
            _render_analysis_failure_state(results, st.session_state.analysis_failures or [])
        elif results and total == 0:
            _render_zero_detection_empty(results)
        elif results:
            if st.button("Continue to Review", type="primary", use_container_width=True):
                navigate_to("wizard", stage="review")
        render_nav_buttons(back_stage="photos", key_prefix="an_partial")
        return

    if status == "error":
        st.error("Analysis did not produce any successful results.")
        for fail in st.session_state.analysis_failures:
            st.error(fail)

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
        use_container_width=True,
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

    st.session_state.analyze_running = True
    st.session_state.analysis_status = "running"
    detector = RoboflowDetector()
    preview_slot = st.empty()
    progress = st.progress(0.0, text="Starting analysis…")
    status_box = st.empty()
    results: list[InferenceResult] = []
    failures: list[str] = []
    comparison_summaries: list[dict[str, Any]] = []

    prompt = _form_get("prompt", "")
    conf = float(_form_get("confidence_threshold", 0.25))
    iou = float(_form_get("iou_threshold", 0.5))
    tile_size = int(_form_get("tile_size", 800))
    tile_overlap = float(_form_get("tile_overlap", 0.25))
    dedup = _form_get("deduplication_strategy", "Conservative")
    options = InferenceOptions(
        prompt=prompt or "",
        confidence_threshold=conf,
        iou_threshold=iou,
        inference_mode=inference_mode,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        deduplication_strategy=dedup,
    )
    total = max(1, len(images) * len(selected_models))
    step_i = 0

    def _show_analysis_preview(item: dict[str, Any], img_i: int, model_i: int, model_name: str) -> None:
        with preview_slot.container():
            st.markdown('<div class="aic-img-card">', unsafe_allow_html=True)
            st.image(item["data"], use_container_width=True, output_format="JPEG")
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
                prog_txt = (
                    progress_label(model_i, len(selected_models), img_i, len(images))
                    if len(selected_models) > 1
                    else f"Analyzing image {img_i} of {len(images)} with {model.name}…"
                )
                progress.progress(step_i / total, text=prog_txt)
                status_box.caption(f"{prog_txt}\n\nRunning: {model.name}")

                key = _cache_key(
                    prepared.content_hash,
                    model.name,
                    prompt,
                    conf,
                    inference_mode,
                    tile_size,
                    tile_overlap,
                    dedup,
                    iou,
                )
                cached = st.session_state.inference_cache.get(key)
                if cached is not None:
                    results.append(cached)
                    comparison_summaries.append(
                        summary_row_from_cached(cached, model_key=model_key(model))
                    )
                    continue

                adapter = get_adapter(model, detector=detector)
                mir = adapter.predict(prepared, options)
                comparison_summaries.append(
                    summary_row_from_mir(mir, image_name=prepared.image_name)
                )
                if mir.success and mir.inference_result is not None:
                    st.session_state.inference_cache[key] = mir.inference_result
                    results.append(mir.inference_result)
                else:
                    # Do not convert failures into zero-detection InferenceResults
                    failures.append(
                        f"{prepared.image_name} / {model.name}: "
                        f"{mir.error_message or mir.error_type or 'failed'}"
                    )

        st.session_state.analysis_results = results
        st.session_state.analysis_failures = failures
        st.session_state.comparison_summaries = comparison_summaries
        st.session_state.review_edits = {
            "excluded_ids": [],
            "manual_detections": [],
            "class_overrides": {},
        }
        selected_keys = [model_key(m) for m in selected_models]
        selected_display = [m.name for m in selected_models]
        st.session_state.analysis_meta = {
            "yard": _resolved_yard(),
            "inventory_type": _resolved_inventory(),
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
        st.session_state.analyze_running = False

    st.rerun()


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
    base_count = sum(1 for d in result.detections if d.detection_id not in excluded) + len(manuals)

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
    st.subheader("Review & Save")

    if st.session_state.save_status == "saved" and st.session_state.saved_record:
        st.success("Inventory analysis saved successfully")
        rec = st.session_state.saved_record
        st.markdown(
            f"""
            <div class="aic-card">
              <div><b>Record ID:</b> {rec.get("id")}</div>
              <div><b>Saved:</b> {rec.get("created_at")}</div>
              <div><b>Inventory type:</b> {rec.get("inventory_type")}</div>
              <div><b>Reviewed count:</b> {rec.get("reviewed_count")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("View History", use_container_width=True, key="post_hist"):
                open_settings(section="history")
        with c2:
            if st.button(
                "Start New Analysis",
                type="primary",
                use_container_width=True,
                key="post_new",
            ):
                reset_active_analysis(go_home=False, start_wizard=True)
        return

    results: list[InferenceResult] = st.session_state.analysis_results or []
    if not results or st.session_state.analysis_status not in {"complete", "partial"}:
        st.warning("Run analysis before reviewing results.")
        render_nav_buttons(back_stage="analyze", key_prefix="rev_gate")
        return

    if st.session_state.analysis_status == "partial":
        st.warning("Showing successful results. Some runs failed.")

    photo_names: list[str] = []
    for r in results:
        if r.image_name not in photo_names:
            photo_names.append(r.image_name)

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
        st.markdown("### Model Comparison")
        st.caption(
            "Switching model tabs does not rerun inference. "
            "Factual labels below do not prove accuracy."
        )
        st.info(f"**Selected for Review:** {accepted.model_name}")
        view_mode = "Tabs"
        if len(models_for_image) == 2:
            view_mode = st.radio(
                "View",
                ["Tabs", "Side by Side"],
                horizontal=True,
                key="compare_view_mode",
            )
        # Factual labels only (not accuracy)
        img_results = [r for r in results if r.image_name == active_image]
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
                f"Highest average confidence: **{highest_conf.model_name}** · "
                f"Fewest warnings: **{fewest_warn.model_name}** "
                "(these labels do not prove accuracy)"
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
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.session_state.compare_side_by_side = view_mode == "Side by Side"
    elif summaries:
        with st.expander("Comparison summary", expanded=False):
            st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)

    style = st.radio(
        "Visualization",
        ["Numbered Markers", "Bounding Boxes", "Both"],
        index=["Numbered Markers", "Bounding Boxes", "Both"].index(
            st.session_state.get("annotation_style_label", "Both")
        )
        if st.session_state.get("annotation_style_label", "Both")
        in {"Numbered Markers", "Bounding Boxes", "Both"}
        else 2,
        horizontal=True,
        key="rev_viz_style",
    )
    st.session_state.annotation_style_label = style
    style_key = {
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
    filt_label = st.session_state.get("rev_det_filter", "All")
    filt_key = {
        "All": "all",
        "Included": "included",
        "Excluded": "excluded",
        "Warnings": "warnings",
        "Manual": "manual",
    }.get(filt_label, "all")
    nav_pool = filter_detections(
        review_dets, filt_key, excluded_detections=excluded_dets
    )
    if nav_pool and selected_id not in {d.detection_id for d in nav_pool}:
        selected_id = nav_pool[0].detection_id
        st.session_state.selected_detection_id = selected_id
    if st.session_state.get("rev_det_jump") in {d.detection_id for d in nav_pool}:
        selected_id = st.session_state.rev_det_jump
        st.session_state.selected_detection_id = selected_id

    left, right = st.columns([2.1, 1.0], gap="medium")

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
                        use_container_width=True,
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
                        st.image(image_to_png_bytes(ann), use_container_width=True)
                    except Exception:  # noqa: BLE001
                        if r_side.annotated_image_bytes:
                            st.image(r_side.annotated_image_bytes, use_container_width=True)
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

        st.markdown('<div class="aic-img-card">', unsafe_allow_html=True)
        if match:
            base_img = Image.open(io.BytesIO(match["data"])).convert("RGB")
            annotated = annotate_image(
                base_img,
                review_dets,
                model_name=display_result.model_name,
                style=style_key,
                selected_detection_id=selected_id,
                show_legend=False,
            )
            st.image(image_to_png_bytes(annotated), use_container_width=True)
        elif display_result.annotated_image_bytes:
            st.image(display_result.annotated_image_bytes, use_container_width=True)
        else:
            st.info("No annotated image available.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Scalable detection navigator (no per-detection button strip)
        st.radio(
            "Filter",
            ["All", "Included", "Excluded", "Warnings", "Manual"],
            horizontal=True,
            key="rev_det_filter",
        )
        # Recompute pool after filter widget (session key updated for next run / same run)
        filt_label = st.session_state.get("rev_det_filter", "All")
        filt_key = {
            "All": "all",
            "Included": "included",
            "Excluded": "excluded",
            "Warnings": "warnings",
            "Manual": "manual",
        }.get(filt_label, "all")
        nav_pool = filter_detections(
            review_dets, filt_key, excluded_detections=excluded_dets
        )
        if nav_pool:
            if selected_id not in {d.detection_id for d in nav_pool}:
                selected_id = nav_pool[0].detection_id
                st.session_state.selected_detection_id = selected_id
            cur_idx = index_of_detection(nav_pool, selected_id)
            n1, n2, n3 = st.columns([1, 1.4, 1])
            with n1:
                if st.button(
                    "Previous",
                    key="rev_det_prev",
                    use_container_width=True,
                    disabled=cur_idx <= 0,
                ):
                    st.session_state.selected_detection_id = step_detection_id(
                        nav_pool, selected_id, delta=-1
                    )
                    st.rerun()
            with n2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:0.45rem;'>"
                    f"<b>Detection</b> {cur_idx + 1} of {len(nav_pool)}</div>",
                    unsafe_allow_html=True,
                )
            with n3:
                if st.button(
                    "Next",
                    key="rev_det_next",
                    use_container_width=True,
                    disabled=cur_idx >= len(nav_pool) - 1,
                ):
                    st.session_state.selected_detection_id = step_detection_id(
                        nav_pool, selected_id, delta=1
                    )
                    st.rerun()
            jump_ids = [d.detection_id for d in nav_pool]
            jump = st.selectbox(
                "Jump to",
                options=jump_ids,
                index=cur_idx,
                format_func=lambda did: format_detection_option(
                    next(d for d in nav_pool if d.detection_id == did),
                    excluded=did in excluded,
                ),
                key="rev_det_jump",
            )
            st.session_state.selected_detection_id = jump
            selected_id = jump
            j1, j2 = st.columns([2, 1])
            with j1:
                jump_num = st.number_input(
                    "Enter detection number",
                    min_value=1,
                    max_value=max(1, max((d.marker_number or 1) for d in nav_pool)),
                    value=int(nav_pool[cur_idx].marker_number or cur_idx + 1),
                    step=1,
                    key="rev_det_num_jump",
                )
            with j2:
                st.write("")
                if st.button("Go", key="rev_det_num_go", use_container_width=True):
                    match_d = next(
                        (d for d in nav_pool if int(d.marker_number or 0) == int(jump_num)),
                        None,
                    )
                    if match_d:
                        st.session_state.selected_detection_id = match_d.detection_id
                        st.rerun()
        else:
            st.caption("No detections match this filter.")

        # Compact image navigator
        if len(photo_names) > 1:
            idx = photo_names.index(active_image)
            nav_l, nav_m, nav_r = st.columns([1, 2, 1])
            with nav_l:
                if st.button("← Previous", disabled=idx <= 0, key="rev_prev", use_container_width=True):
                    st.session_state.review_active_image = photo_names[idx - 1]
                    st.session_state.selected_detection_id = None
                    st.rerun()
            with nav_m:
                st.markdown(
                    f"<div style='text-align:center;padding-top:0.4rem;'>Image {idx + 1} of {len(photo_names)}</div>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                if st.button("Next →", disabled=idx >= len(photo_names) - 1, key="rev_next", use_container_width=True):
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
                        use_container_width=True,
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
        st.markdown("#### Analysis Summary")
        if display_result.model_name != accepted.model_name:
            st.caption(
                f"Viewing **{display_result.model_name}** · "
                f"Selected for Review: **{accepted.model_name}**"
            )
        st.markdown(
            f"""
            <div class="aic-metric-grid">
              <div class="aic-metric-tile">AI Count<b>{display_result.final_count}</b></div>
              <div class="aic-metric-tile">Final Count<b>{reviewed}</b></div>
              <div class="aic-metric-tile">Duplicates<b>{display_result.duplicates_removed}</b></div>
              <div class="aic-metric-tile">Warnings<b>{warn_count}</b></div>
              <div class="aic-metric-tile">Included<b>{included}</b></div>
              <div class="aic-metric-tile">Excluded<b>{excluded_n}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Selected detection panel
        selected = next(
            (d for d in list(review_dets) + list(excluded_dets) if d.detection_id == selected_id),
            None,
        )
        st.markdown("#### Selected Detection")
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
                f"<br/><span style='color:#b54708;'>⚠ {' · '.join(warn_bits)}</span>"
                if warn_bits
                else ""
            )
            st.markdown(
                f"""
                <div class="aic-det-row-selected">
                  <span class="aic-det-chip" style="background:{css_rgb(color)};">{selected.marker_number}</span>
                  <b>Detection {selected.marker_number}</b><br/>
                  {selected.class_name}<br/>
                  {CONFIDENCE_LABEL}: {conf_txt} ({band}) · Status: {status}
                  {warn_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(CONFIDENCE_HELP)
            a1, a2 = st.columns(2)
            with a1:
                if is_excl:
                    if st.button("Include", key="rev_incl_sel", use_container_width=True):
                        excluded.discard(selected.detection_id)
                        edits["excluded_ids"] = list(excluded)
                        st.session_state.review_edits = edits
                        st.rerun()
                elif st.button("Exclude", key="rev_excl_sel", use_container_width=True):
                    if not selected.is_manual:
                        excluded.add(selected.detection_id)
                        edits["excluded_ids"] = list(excluded)
                    else:
                        edits["manual_detections"] = [
                            m
                            for m in (edits.get("manual_detections") or [])
                            if m.get("detection_id") != selected.detection_id
                        ]
                    st.session_state.review_edits = edits
                    st.session_state.selected_detection_id = None
                    st.rerun()
            with a2:
                new_label = st.text_input(
                    "Edit label",
                    value=selected.class_name,
                    key="rev_edit_label",
                    label_visibility="collapsed",
                )
                if new_label != selected.class_name and st.button(
                    "Save label", key="rev_save_label", use_container_width=True
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
            if excluded and st.button("Restore all excluded", key="rev_restore"):
                edits["excluded_ids"] = []
                st.session_state.review_edits = edits
                st.rerun()
        else:
            st.caption("Use Previous / Next or Jump to select a detection.")

        # Compact paginated detection list (max ~15 rows)
        st.markdown("##### Detections")
        page = int(st.session_state.get("rev_det_page", 0) or 0)
        page_items, page, total_pages = paginate(nav_pool, page, PAGE_SIZE)
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
                    "Excluded"
                    if d.detection_id in excluded
                    else ("Manual" if d.is_manual else "Included")
                )
                conf_txt = format_confidence_percent(d.confidence)
                st.markdown(
                    f"""
                    <div class="{'aic-det-row-selected' if d.detection_id == selected_id else ''}"
                         style="padding:0.2rem 0.35rem;">
                      <span class="aic-det-chip" style="background:{css_rgb(color)};">{d.marker_number}</span>
                      {d.class_name} · {CONFIDENCE_LABEL_SHORT} {conf_txt} · {state} {warn}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            p1, p2, p3 = st.columns([1, 2, 1])
            with p1:
                if st.button("◀", key="rev_page_prev", disabled=page <= 0):
                    st.session_state.rev_det_page = page - 1
                    st.rerun()
            with p2:
                st.caption(f"Page {page + 1} of {total_pages} · {len(nav_pool)} detections")
            with p3:
                if st.button("▶", key="rev_page_next", disabled=page >= total_pages - 1):
                    st.session_state.rev_det_page = page + 1
                    st.rerun()
        else:
            st.caption("No detections for this filter.")

        tab_adj, tab_dup, tab_warn, tab_det = st.tabs(
            ["Adjustments", "Duplicates", "Warnings", "Details"]
        )

        with tab_adj:
            st.write(f"Current final count: **{reviewed}**")
            rs = st.session_state.review_state
            use_direct = st.checkbox(
                "Enter reviewed count directly",
                value=bool(rs.get("use_direct")),
                key="rev_direct_cb",
            )
            fp = st.number_input(
                "False positives to subtract",
                min_value=0,
                value=int(rs.get("false_positives") or 0),
                step=1,
                key="rev_fp",
            )
            missed = st.number_input(
                "Missed items to add",
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
                "Review notes", value=rs.get("notes") or "", height=60, key="rev_notes"
            )
            st.session_state.review_state = {
                "use_direct": use_direct,
                "direct_count": direct_val if use_direct else None,
                "false_positives": int(fp),
                "missed_items": int(missed),
                "notes": notes,
            }
            with st.expander("Add manual detection", expanded=False):
                st.caption("No image-click canvas in this POC — set coordinates, then confirm.")
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

        with tab_dup:
            st.write(f"Duplicates removed by AI: **{display_result.duplicates_removed}**")
            overlap_pairs = [
                d for d in review_dets if d.suspected_overlap
            ]
            if overlap_pairs:
                for d in overlap_pairs:
                    st.warning(
                        f"Detection {d.marker_number} — possible duplicate / overlap "
                        f"({d.class_name}, {d.confidence:.0%})"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(
                            f"Keep Detection {d.marker_number}",
                            key=f"dup_keep_{d.detection_id}",
                        ):
                            st.info(f"Detection {d.marker_number} kept.")
                    with c2:
                        if st.button(
                            f"Exclude Detection {d.marker_number}",
                            key=f"dup_excl_{d.detection_id}",
                        ):
                            excluded.add(d.detection_id)
                            edits["excluded_ids"] = list(excluded)
                            st.session_state.review_edits = edits
                            st.rerun()
            else:
                st.caption("No open duplicate candidates for this result.")
            if display_result.strategy_counts:
                st.caption("Dedup strategy counts")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Strategy": k, "Count": v}
                            for k, v in display_result.strategy_counts.items()
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        with tab_warn:
            shown = False
            for d in review_dets:
                if d.suspected_occlusion:
                    st.warning(
                        f"Detection {d.marker_number} — Partially outside / occluded"
                    )
                    shown = True
                if d.confidence < 0.35:
                    st.warning(
                        f"Detection {d.marker_number} — Low confidence ({d.confidence:.0%})"
                    )
                    shown = True
            for w in display_result.warnings or []:
                st.warning(w)
                shown = True
            if not shown:
                st.caption("No warnings.")

        with tab_det:
            if selected:
                st.write(f"**Detection {selected.marker_number}** · {selected.class_name}")
                st.write(f"Confidence: {selected.confidence:.0%}")
                st.write(f"Manual: {'Yes' if selected.is_manual else 'No'}")
                with st.expander("Technical Details", expanded=False):
                    st.write(f"Stable detection ID: `{selected.detection_id}`")
                    st.write(f"Model: {display_result.model_name}")
                    st.write(f"Raw class: {selected.class_name}")
                    st.write(f"Raw confidence: {selected.confidence:.4f}")
                    st.write(
                        f"Coordinates: ({selected.x1:.1f},{selected.y1:.1f})-"
                        f"({selected.x2:.1f},{selected.y2:.1f})"
                    )
                    st.write(
                        f"Center: ({selected.center_x:.1f}, {selected.center_y:.1f})"
                    )
                    st.write(f"Source image: {display_result.image_name}")
                    st.write(f"Response source: {display_result.source or '(n/a)'}")
            else:
                st.caption("Select a detection to view details.")
            with st.expander("Result technical summary", expanded=False):
                st.write(f"Avg confidence: {display_result.avg_confidence:.1%}")
                st.write(f"Processing time: {display_result.processing_time_seconds:.2f}s")
                st.write(f"API calls used: {display_result.api_calls_used}")
                st.write(f"Inference mode: {display_result.inference_mode}")
                st.json(display_result.summary_dict())

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("← Back", use_container_width=True, key="rev_back"):
            navigate_to("wizard", stage="analyze")
    with b2:
        if st.button("Re-run Analysis", use_container_width=True, key="rev_rerun"):
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
            use_container_width=True,
            disabled=save_disabled,
            key="rev_save",
        ):
            _save_inventory()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _render_wizard() -> None:
    stage = normalize_stage(st.session_state.get("wizard_stage") or "setup")
    st.session_state.wizard_stage = stage

    if stage == "review" and st.session_state.analysis_status not in {"complete", "partial"}:
        if st.session_state.save_status != "saved":
            st.warning("Complete analysis before reviewing.")
            stage = "analyze"
            st.session_state.wizard_stage = stage

    dispatch = {
        "setup": stage_setup,
        "photos": stage_photos,
        "analyze": stage_analyze,
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
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    config.reload_settings()
    _init_session()
    inject_css()
    ensure_data_dir()
    try:
        initialize_database()
    except DatabaseError:
        pass

    view = normalize_view(st.session_state.get("app_view") or "welcome")
    st.session_state.app_view = view

    if view == "welcome":
        view_welcome()
    elif view == "wizard":
        render_page_toolbar(
            mode="wizard",
            on_settings=lambda: open_settings(section="ai_configuration"),
            on_start_fresh=lambda: reset_active_analysis(go_home=True),
        )
        _render_wizard()
    elif view == "settings":
        view_settings()
    else:
        st.session_state.app_view = "welcome"
        view_welcome()


if __name__ == "__main__":
    main()
