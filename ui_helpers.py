"""UI helpers and page navigation for AI Inventory Counter.

Constants live in app_constants.py (no Streamlit).
This module must not be imported by app_constants.
Streamlit is imported lazily via `_st()` only inside functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from app_constants import (
    ADMIN_ONLY_VIEWS,
    DEFAULT_REVIEW,
    PHOTO_REL_DISPLAY,
    PHOTO_REL_INTERNAL_TO_DISPLAY,
    SETTINGS_LABEL_TO_SECTION,
    SETTINGS_SECTION_LABELS,
    SETTINGS_SECTIONS,
    STAGE_ALIASES,
    STAGE_LABELS,
    STAGES,
    VIEW_ALIASES,
    VIEWS,
    get_settings_section_from_label,
)

# Re-export so `from ui_helpers import SETTINGS_SECTION_LABELS` still works
# AFTER this module has finished importing app_constants (always safe).
__all__ = [
    "ADMIN_ONLY_VIEWS",
    "VIEWS",
    "DEFAULT_REVIEW",
    "PHOTO_REL_DISPLAY",
    "PHOTO_REL_INTERNAL_TO_DISPLAY",
    "SETTINGS_LABEL_TO_SECTION",
    "SETTINGS_SECTION_LABELS",
    "SETTINGS_SECTIONS",
    "STAGE_ALIASES",
    "STAGE_LABELS",
    "STAGES",
    "VIEW_ALIASES",
    "DEFAULT_UI_THEME",
    "default_form",
    "get_settings_section_from_label",
    "get_ui_theme",
    "inject_css",
    "leave_settings",
    "navigate_to",
    "normalize_stage",
    "normalize_view",
    "open_settings",
    "render_card",
    "render_empty_state",
    "render_nav_buttons",
    "render_page_toolbar",
    "render_status_badge",
    "render_settings_header",
    "render_stage_header",
    "render_stepper",
    "reset_active_analysis",
]

DEFAULT_UI_THEME = "dark"
_UI_THEMES = frozenset({"dark", "light"})


def _st():
    import streamlit as st

    return st


def default_form() -> dict[str, Any]:
    from inventory_config import FIXED_PHOTO_RELATIONSHIP

    return {
        "yard_choice": "LA Yard",
        "yard_custom": "",
        # Unset until the user picks an inventory type on Inventory Setup.
        "inventory_choice": "",
        "inventory_custom": "",
        "custom_item_name": "",
        "custom_item_alternatives": "",
        "effective_prompts": [],
        "counting_unit": "",
        "photo_relationship": FIXED_PHOTO_RELATIONSHIP,
        "selected_mode": "Single Model",
        "selected_models": [],
        "prompt": "",
        "prompt_preset": "",
        # Align with YOLO-World workflow confidence (~0.1 server-side); 0.40 hid live hits.
        "confidence_threshold": 0.25,
        "iou_threshold": 0.50,
        "inference_mode": "Whole Image",
        "tile_size": 800,
        "tile_overlap": 0.25,
        "deduplication_strategy": "Conservative",
        "agreement_label": "At least 2 models",
        "confirm_high_api": False,
        "class_override": "",
        "counting_strategy": "",
        "recommended_model_name": "",
        "recommended_setup_resolved": False,
        "recommended_setup_error": "",
    }


def normalize_view(view: str) -> str:
    mapped = VIEW_ALIASES.get(view, view)
    if mapped in VIEWS:
        return mapped
    return "welcome"


def normalize_stage(stage: str) -> str:
    mapped = STAGE_ALIASES.get(stage, stage)
    return mapped if mapped in STAGES else "setup"


def navigate_to(
    view: str,
    *,
    stage: str | None = None,
    settings_section: str | None = None,
) -> None:
    st = _st()
    view = normalize_view(view)
    st.session_state.app_view = view
    if stage is not None:
        st.session_state.wizard_stage = normalize_stage(stage)
    if settings_section is not None and settings_section in SETTINGS_SECTIONS:
        st.session_state.settings_section = settings_section
        st.session_state.settings_section_radio = SETTINGS_SECTION_LABELS[settings_section]
    st.rerun()


def open_settings(section: str = "ai_configuration") -> None:
    st = _st()
    if section not in SETTINGS_SECTIONS:
        section = "ai_configuration"

    current_page = normalize_view(st.session_state.get("app_view", "welcome"))
    if current_page != "settings":
        st.session_state.previous_page = current_page
        if current_page == "wizard":
            st.session_state.previous_wizard_step = st.session_state.get(
                "wizard_stage", "setup"
            )

    st.session_state.settings_section = section
    st.session_state.settings_section_radio = SETTINGS_SECTION_LABELS[section]
    st.session_state.app_view = "settings"
    st.rerun()


def leave_settings() -> None:
    st = _st()
    return_page = normalize_view(st.session_state.pop("previous_page", "welcome") or "welcome")
    if return_page not in {"welcome", "wizard"}:
        return_page = "welcome"

    if return_page == "wizard":
        return_step = st.session_state.pop("previous_wizard_step", None)
        st.session_state.wizard_stage = normalize_stage(return_step or "setup")
    else:
        st.session_state.pop("previous_wizard_step", None)

    st.session_state.app_view = return_page
    st.rerun()


def reset_active_analysis(
    *,
    go_home: bool = True,
    start_wizard: bool = False,
    rerun: bool = True,
) -> None:
    st = _st()
    st.session_state.form = default_form()
    st.session_state.uploaded_images = []
    st.session_state.uploader_nonce = int(st.session_state.get("uploader_nonce", 0)) + 1
    st.session_state.analysis_status = "idle"
    st.session_state.analysis_results = []
    st.session_state.analysis_failures = []
    st.session_state.analysis_meta = {}
    st.session_state.consensus_result = None
    st.session_state.comparison_summaries = []
    st.session_state.accepted_result_key = None
    st.session_state.review_state = dict(DEFAULT_REVIEW)
    st.session_state.review_edits = {
        "excluded_ids": [],
        "manual_detections": [],
        "class_overrides": {},
    }
    st.session_state.save_status = "idle"
    st.session_state.saved_record = None
    st.session_state.analyze_running = False
    st.session_state._analysis_executing = False
    st.session_state.analysis_run_id = None
    st.session_state.pending_review_payload = None
    st.session_state.selected_photo_index = 0
    st.session_state.open_advanced_settings = False
    st.session_state.inference_cache = {}
    st.session_state.previous_page = None
    st.session_state.previous_wizard_step = None
    st.session_state.model_trial_rows = []
    st.session_state.model_trial_suggestion = {}
    st.session_state.pending_camera = None
    st.session_state.selected_detection_id = None
    st.session_state.review_active_image = None
    st.session_state.review_active_model = None
    st.session_state.sample_selected_ids = []
    st.session_state.sample_preview_id = None
    st.session_state.compare_side_by_side = False
    st.session_state.run_context = None
    if go_home:
        st.session_state.app_view = "welcome"
        st.session_state.wizard_stage = "setup"
    elif start_wizard:
        st.session_state.app_view = "wizard"
        st.session_state.wizard_stage = "setup"
    if rerun:
        st.rerun()


def get_ui_theme() -> str:
    """App is dark-only (no theme toggle)."""
    return DEFAULT_UI_THEME


def render_stage_header(title: str, caption: str) -> None:
    """Compact professional section header for wizard + settings pages."""
    st = _st()
    st.markdown(
        f"""
        <div class="aic-settings-head">
          <h3>{title}</h3>
          <p>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_header(title: str, caption: str) -> None:
    """Alias for settings pages."""
    render_stage_header(title, caption)


def _theme_override_css(theme: str) -> str:
    # Dark theme only — light toggle removed.
    return """
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        section.main {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stHeader"] {
            background: rgba(14,17,23,0.92) !important;
        }
        [data-testid="stToolbar"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(250,250,250,0.05) !important;
            border-color: rgba(250,250,250,0.12) !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] > div {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        .aic-card, .aic-inv-card, .aic-empty, .aic-model-card {
            background: rgba(250,250,250,0.04) !important;
            border-color: rgba(250,250,250,0.14) !important;
        }
        .aic-toolbar-title, .aic-hero-title, .aic-inv-card-title {
            color: #fafafa !important;
        }
        .aic-hero-sub, .aic-muted, .aic-note, .aic-analyze-status {
            color: rgba(250,250,250,0.72) !important;
        }
        hr, [data-testid="stDivider"] {
            border-color: rgba(250,250,250,0.14) !important;
        }
        """


def inject_css() -> None:
    st = _st()
    theme = get_ui_theme()
    base_css = """
        <style>
        [data-testid="stAppViewContainer"] .main .block-container,
        section.main .block-container,
        .block-container {
            padding-top: 2.75rem;
            padding-bottom: 2rem;
            max-width: 1080px;
            overflow: visible;
        }
        .aic-card {
            border: 1px solid rgba(128,128,128,0.20);
            border-radius: 12px;
            padding: 1rem 1.15rem;
            background: rgba(128,128,128,0.05);
            margin-bottom: 0.75rem;
        }
        .aic-card-selected {
            border: 2px solid rgba(255,75,75,0.75);
            background: rgba(255,75,75,0.08);
        }
        .aic-muted { opacity: 0.72; font-size: 0.92rem; line-height: 1.45; }
        .aic-hero-title {
            font-size: 2.1rem; font-weight: 700; margin-bottom: 0.35rem;
        }
        .aic-hero-sub {
            font-size: 1.05rem; opacity: 0.78; max-width: 34rem;
            margin-bottom: 1.4rem; line-height: 1.5;
        }
        .aic-badge {
            display: inline-block; font-size: 0.72rem; font-weight: 600;
            padding: 0.15rem 0.5rem; border-radius: 999px;
            background: rgba(255, 170, 0, 0.18);
            border: 1px solid rgba(255, 170, 0, 0.45);
            margin-left: 0.35rem; vertical-align: middle;
        }
        .aic-stepper {
            display: flex; flex-wrap: wrap; gap: 0.35rem;
            margin: 0.25rem 0 0.55rem 0; font-size: 0.76rem;
            padding: 0.4rem 0.5rem;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.18);
            background: rgba(250,250,250,0.03);
        }
        .aic-pill {
            padding: 0.26rem 0.58rem; border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.28); opacity: 0.5;
        }
        .aic-pill.done {
            opacity: 0.9;
            background: rgba(250,250,250,0.06);
            border-color: rgba(128,128,128,0.28);
        }
        .aic-pill.current {
            opacity: 1; font-weight: 700;
            background: rgba(250,250,250,0.12);
            border-color: rgba(250,250,250,0.35);
        }
        .aic-empty {
            text-align: center; padding: 1.5rem 1rem;
            border: 1px dashed rgba(128,128,128,0.35);
            border-radius: 12px; background: rgba(128,128,128,0.04);
        }
        .aic-status-ok { color: #2ea043; font-weight: 600; }
        .aic-status-bad { color: #e5534b; font-weight: 600; }
        .aic-toolbar-title {
            font-size: 1.15rem; font-weight: 700; margin: 0.35rem 0;
            line-height: 1.3;
        }
        div[data-testid="stMetric"] {
            background: rgba(128,128,128,0.07);
            border: 1px solid rgba(128,128,128,0.16);
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
        }
        div[data-testid="stHorizontalBlock"] button {
            min-height: 2.4rem;
        }
        .aic-inv-card {
            position: relative;
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 12px;
            padding: 0.85rem 0.9rem;
            min-height: 4.35rem;
            background: rgba(128,128,128,0.05);
            margin-bottom: 0.45rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }
        .aic-inv-card--selected {
            border: 2px solid rgba(255,75,75,0.75);
            background: rgba(255,75,75,0.08);
        }
        .aic-inv-card--unavailable {
            cursor: not-allowed;
            user-select: none;
            pointer-events: none;
            opacity: 0.92;
        }
        .aic-inv-card-title {
            font-weight: 650;
            font-size: 0.95rem;
            line-height: 1.3;
            padding-right: 1.4rem;
        }
        .aic-inv-soon {
            font-size: 0.72rem;
            opacity: 0.72;
            margin-top: 0.2rem;
        }
        .aic-inv-unavailable {
            position: absolute;
            top: 0.4rem;
            right: 0.4rem;
            width: 1.15rem;
            height: 1.15rem;
            border-radius: 50%;
            background: #e5534b;
            color: #fff;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.15rem;
            text-align: center;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .aic-photos-status {
            border: 1px solid rgba(46,160,67,0.45);
            background: rgba(46,160,67,0.12);
            border-radius: 10px;
            padding: 0.45rem 0.65rem;
            font-size: 0.8rem;
            line-height: 1.35;
            margin: 0.15rem 0 0.35rem 0;
        }
        .aic-photos-status .dot {
            display: inline-block;
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: #2ea043;
            margin-right: 0.35rem;
            vertical-align: middle;
        }
        .aic-analyze-status {
            font-size: 0.88rem;
            opacity: 0.85;
            margin: 0.25rem 0 0.65rem 0;
        }
        .aic-note {
            font-size: 0.88rem; opacity: 0.78; margin: 0.25rem 0 0.65rem 0;
            line-height: 1.4;
        }
        .aic-img-card {
            border: 2px solid rgba(46,160,67,0.55);
            border-radius: 12px;
            padding: 0.45rem;
            background: rgba(46,160,67,0.06);
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            margin: 0.25rem 0 0.5rem 0;
        }
        .aic-img-card [data-testid="stImage"],
        .aic-img-card [data-testid="stImage"] > img,
        div[data-testid="stImage"] {
            background: transparent !important;
        }
        .aic-img-card img,
        div[data-testid="stImage"] img {
            object-fit: contain !important;
            width: 100% !important;
            height: auto !important;
            max-height: 68vh;
            background: transparent !important;
        }
        .aic-det-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.55rem;
            height: 1.55rem;
            border-radius: 999px;
            color: #fff;
            font-size: 0.78rem;
            font-weight: 700;
            margin-right: 0.35rem;
            border: 2px solid rgba(255,255,255,0.85);
            box-shadow: 0 0 0 1px rgba(0,0,0,0.25);
        }
        .aic-det-row-selected {
            border: 2px solid rgba(46,160,67,0.65);
            border-radius: 10px;
            padding: 0.35rem 0.5rem;
            background: rgba(46,160,67,0.10);
            margin-bottom: 0.35rem;
        }
        .aic-metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 0.3rem;
            margin: 0.2rem 0 0.45rem 0;
        }
        .aic-metric-tile {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 8px;
            padding: 0.3rem 0.45rem;
            background: rgba(250,250,250,0.04);
            font-size: 0.72rem;
            opacity: 0.95;
        }
        .aic-metric-tile b { display: block; font-size: 1.05rem; margin-top: 0.08rem; font-weight: 700; }
        .aic-review-workspace {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 10px;
            padding: 0.45rem 0.55rem 0.2rem 0.55rem;
            background: rgba(250,250,250,0.025);
            margin-top: 0.15rem;
        }
        .aic-review-meta {
            font-size: 0.84rem;
            opacity: 0.78;
            margin: 0 0 0.45rem 0;
            line-height: 1.4;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"],
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            min-height: 4.35rem;
            border-radius: 12px;
        }
        /* Inventory setup tiles keep taller buttons; review marker chips stay compact */
        .aic-marker-bar button {
            min-height: 2.1rem !important;
            border-radius: 999px !important;
        }
        /* Compact Add Photos layout */
        .aic-photo-strip {
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 12px;
            padding: 0.55rem 0.65rem;
            background: rgba(250,250,250,0.03);
            margin: 0.35rem 0 0.5rem 0;
        }
        .aic-photo-strip-title {
            font-size: 0.85rem;
            font-weight: 650;
            margin: 0 0 0.35rem 0;
        }
        .aic-sample-card {
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 10px;
            padding: 0.35rem;
            background: rgba(250,250,250,0.03);
            margin-bottom: 0.35rem;
            min-height: 0;
        }
        .aic-sample-card img {
            max-height: 110px !important;
            object-fit: cover !important;
            border-radius: 8px;
        }
        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] img,
        [data-testid="stCameraInput"] > div {
            max-height: 200px !important;
        }
        [data-testid="stCameraInput"] {
            max-width: 420px;
        }
        .aic-model-pick {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 12px;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.4rem;
            background: rgba(250,250,250,0.03);
        }
        .aic-model-pick-selected {
            border-color: rgba(255,75,75,0.55);
            box-shadow: 0 0 0 1px rgba(255,75,75,0.25);
        }
        .aic-rgb-accent {
            height: 2px;
            border-radius: 999px;
            margin: 0.1rem 0 0.4rem 0;
            background: linear-gradient(90deg, #ff4b4b 0%, #2196f3 55%, #2ea043 100%);
            opacity: 0.55;
        }
        .aic-review-image {
            margin: 0.15rem 0 0.35rem 0 !important;
            padding: 0.3rem !important;
        }
        /* Review canvas: keep the annotated image viewport-bound */
        [data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stImage"] img,
        .aic-img-card ~ div img {
            max-height: 46vh;
            object-fit: contain;
        }
        /* Professional settings / wizard chrome (RGB reserved for home dashboard) */
        .aic-settings-head {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
            margin: 0.1rem 0 0.55rem 0;
            background: rgba(250,250,250,0.03);
        }
        .aic-settings-head h3 {
            margin: 0 0 0.15rem 0;
            font-size: 1.12rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .aic-settings-head p {
            margin: 0;
            opacity: 0.72;
            font-size: 0.86rem;
            line-height: 1.35;
        }
        .aic-rgb-bar {
            display: none;
        }
        .aic-panel {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 10px;
            padding: 0.55rem 0.7rem;
            margin: 0 0 0.45rem 0;
            background: rgba(250,250,250,0.03);
        }
        .aic-panel-r, .aic-panel-b, .aic-panel-g {
            border-left: 2px solid rgba(128,128,128,0.35);
            background: rgba(250,250,250,0.03);
        }
        .aic-panel-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            opacity: 0.7;
            margin: 0 0 0.35rem 0;
        }
        .aic-chip-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.35rem;
            margin: 0.1rem 0 0.3rem 0;
        }
        .aic-chip-grid-4 {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
        .aic-chip {
            border-radius: 8px;
            padding: 0.35rem 0.45rem;
            border: 1px solid rgba(128,128,128,0.18);
            background: rgba(250,250,250,0.04);
            min-height: 2.6rem;
        }
        .aic-chip-r, .aic-chip-b, .aic-chip-g {
            border-color: rgba(128,128,128,0.18);
            background: rgba(250,250,250,0.04);
        }
        .aic-chip-label {
            display: block;
            font-size: 0.64rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.65;
            margin-bottom: 0.1rem;
        }
        .aic-chip-value {
            display: block;
            font-size: 0.9rem;
            font-weight: 650;
            line-height: 1.25;
            word-break: break-word;
        }
        .aic-kv-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.3rem 0.65rem;
        }
        .aic-kv {
            font-size: 0.84rem;
            line-height: 1.35;
        }
        .aic-kv b { opacity: 0.72; font-weight: 600; }
        .aic-hist-card {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 10px;
            padding: 0.5rem 0.65rem;
            margin-bottom: 0.35rem;
            background: rgba(250,250,250,0.03);
        }
        .aic-hist-card-top {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            align-items: baseline;
            margin-bottom: 0.2rem;
        }
        .aic-hist-card-top b { font-size: 0.92rem; }
        .aic-hist-meta {
            font-size: 0.78rem;
            opacity: 0.72;
            line-height: 1.35;
        }
        .aic-pill-rgb {
            display: inline-block;
            font-size: 0.66rem;
            font-weight: 650;
            padding: 0.1rem 0.4rem;
            border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.3);
            background: rgba(250,250,250,0.06);
            color: inherit;
        }
        @media (max-width: 760px) {
            .aic-chip-grid, .aic-chip-grid-4, .aic-kv-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        /* Home dashboard — RGB kept here only */
        .aic-dash-hero {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 16px;
            padding: 1rem 1.1rem 1.05rem 1.1rem;
            margin: 0.2rem 0 0.85rem 0;
            background:
                linear-gradient(135deg, rgba(255,75,75,0.14), transparent 42%),
                linear-gradient(225deg, rgba(33,150,243,0.14), transparent 48%),
                linear-gradient(15deg, rgba(46,160,67,0.12), transparent 40%),
                rgba(250,250,250,0.03);
        }
        .aic-dash-hero .aic-rgb-bar {
            display: block;
            height: 3px;
            border-radius: 999px;
            margin: 0 0 0.55rem 0;
            background: linear-gradient(90deg, #ff4b4b 0%, #2196f3 50%, #2ea043 100%);
        }
        .aic-dash-status .aic-chip-r {
            border-color: rgba(255,75,75,0.35);
            background: rgba(255,75,75,0.10);
        }
        .aic-dash-status .aic-chip-b {
            border-color: rgba(33,150,243,0.35);
            background: rgba(33,150,243,0.10);
        }
        .aic-dash-status .aic-chip-g {
            border-color: rgba(46,160,67,0.35);
            background: rgba(46,160,67,0.10);
        }
        .aic-dash-hero h1 {
            margin: 0 0 0.25rem 0;
            font-size: 1.65rem;
            font-weight: 750;
            letter-spacing: 0.01em;
        }
        .aic-dash-hero p {
            margin: 0;
            opacity: 0.8;
            font-size: 0.95rem;
            line-height: 1.45;
            max-width: 38rem;
        }
        .aic-dash-tile {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            min-height: 5.2rem;
            margin-bottom: 0.35rem;
            background: rgba(250,250,250,0.03);
        }
        .aic-dash-tile-r {
            border-left: 3px solid #ff4b4b;
            background: linear-gradient(90deg, rgba(255,75,75,0.12), rgba(250,250,250,0.02) 60%);
        }
        .aic-dash-tile-b {
            border-left: 3px solid #2196f3;
            background: linear-gradient(90deg, rgba(33,150,243,0.12), rgba(250,250,250,0.02) 60%);
        }
        .aic-dash-tile-g {
            border-left: 3px solid #2ea043;
            background: linear-gradient(90deg, rgba(46,160,67,0.12), rgba(250,250,250,0.02) 60%);
        }
        .aic-dash-tile h4 {
            margin: 0 0 0.25rem 0;
            font-size: 0.98rem;
            font-weight: 700;
        }
        .aic-dash-tile p {
            margin: 0;
            font-size: 0.82rem;
            opacity: 0.75;
            line-height: 1.35;
        }
    """
    st.markdown(
        base_css + _theme_override_css(theme) + "\n        </style>\n        ",
        unsafe_allow_html=True,
    )


def render_page_toolbar(
    *,
    mode: Literal["home", "wizard", "settings"],
    on_settings: Callable[[], None] | None = None,
    on_start_fresh: Callable[[], None] | None = None,
    on_back: Callable[[], None] | None = None,
) -> None:
    st = _st()
    if mode == "home":
        left, right = st.columns([6, 1.4], vertical_alignment="center")
        with left:
            st.markdown(
                '<p class="aic-toolbar-title">AI Inventory Counter</p>',
                unsafe_allow_html=True,
            )
        with right:
            if on_settings and st.button(
                "Settings",
                key="toolbar_settings_home",
                width="stretch",
            ):
                on_settings()
    elif mode == "wizard":
        left, mid, right = st.columns([5.2, 1.4, 1.3], vertical_alignment="center")
        with left:
            st.markdown(
                '<p class="aic-toolbar-title">AI Inventory Counter</p>',
                unsafe_allow_html=True,
            )
        with mid:
            if on_start_fresh and st.button(
                "Start Fresh",
                key="toolbar_start_fresh",
                width="stretch",
            ):
                on_start_fresh()
        with right:
            if on_settings and st.button(
                "Settings",
                key="toolbar_settings_wizard",
                width="stretch",
            ):
                on_settings()
    else:
        left, right = st.columns([5.5, 1.5], vertical_alignment="center")
        with left:
            if on_back and st.button(
                "← Back to Inventory Counter",
                key="toolbar_back_settings",
                width="stretch",
            ):
                on_back()
        with right:
            st.markdown(
                '<p class="aic-toolbar-title" style="text-align:right;">Settings</p>',
                unsafe_allow_html=True,
            )
    st.divider()


def render_stepper(current: str) -> None:
    st = _st()
    current = normalize_stage(current)
    pills = []
    reached_current = False
    for stage in STAGES:
        label = STAGE_LABELS[stage]
        cls = "aic-pill"
        if stage == current:
            cls += " current"
            reached_current = True
        elif not reached_current:
            cls += " done"
        pills.append(f'<span class="{cls}">{label}</span>')
    idx = STAGES.index(current) if current in STAGES else 0
    st.markdown(
        f'<div class="aic-stepper">{"".join(pills)}</div>'
        f'<p class="aic-muted" style="margin:0 0 0.55rem 0;font-size:0.82rem;">'
        f"Step {idx + 1} of {len(STAGES)} · {STAGE_LABELS.get(current, current)}"
        f"</p>",
        unsafe_allow_html=True,
    )


def render_nav_buttons(
    *,
    back_stage: str | None = None,
    next_label: str = "Continue",
    next_disabled: bool = False,
    on_next: Callable[[], None] | None = None,
    next_stage: str | None = None,
    key_prefix: str = "nav",
) -> None:
    st = _st()
    st.write("")
    left, _, right = st.columns([1, 2, 1])
    with left:
        if back_stage:
            if st.button("← Back", width="stretch", key=f"{key_prefix}_back"):
                navigate_to("wizard", stage=back_stage)
    with right:
        if on_next is not None or next_stage:
            if st.button(
                next_label,
                type="primary",
                width="stretch",
                disabled=next_disabled,
                key=f"{key_prefix}_next",
            ):
                if on_next is not None:
                    on_next()
                elif next_stage:
                    navigate_to("wizard", stage=next_stage)


def render_status_badge(
    ok: bool, ok_text: str = "Connected", bad_text: str = "Not Connected"
) -> str:
    if ok:
        return f'<span class="aic-status-ok">{ok_text}</span>'
    return f'<span class="aic-status-bad">{bad_text}</span>'


def render_empty_state(title: str, body: str) -> None:
    st = _st()
    st.markdown(
        f"""
        <div class="aic-empty">
          <div style="font-size:1.15rem;font-weight:650;margin-bottom:0.4rem;">{title}</div>
          <div class="aic-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card(html_body: str, *, selected: bool = False) -> None:
    st = _st()
    cls = "aic-card aic-card-selected" if selected else "aic-card"
    st.markdown(f'<div class="{cls}">{html_body}</div>', unsafe_allow_html=True)
