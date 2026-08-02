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
    settings_sections_for_role,
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
    "render_page_hero",
    "render_page_toolbar",
    "render_status_badge",
    "render_settings_header",
    "render_stage_header",
    "render_stepper",
    "reset_active_analysis",
    "settings_sections_for_role",
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
    # Legacy callers still pass settings_section — map to the matching panel view.
    if settings_section:
        view = settings_section
    view = normalize_view(view)
    st.session_state.app_view = view
    if stage is not None:
        st.session_state.wizard_stage = normalize_stage(stage)
    st.rerun()


def open_settings(section: str = "ai_configuration") -> None:
    """Compatibility helper — opens a top-level panel (Settings UI removed)."""
    if section not in SETTINGS_SECTIONS:
        section = "ai_configuration"
    navigate_to(section)


def leave_settings() -> None:
    navigate_to("welcome")


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


def render_page_hero(title: str, caption: str = "") -> None:
    """Shared page headline used on Home, Administration, panels, and wizard stages."""
    import html as _html

    st = _st()
    caption_html = f"<p>{_html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f"""
        <div class="aic-page-hero">
          <div class="aic-rgb-bar"></div>
          <h1>{_html.escape(title)}</h1>
          {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_header(title: str, caption: str) -> None:
    """Wizard / panel section header — same visual language as Home."""
    render_page_hero(title, caption)


def render_settings_header(title: str, caption: str) -> None:
    """Alias for panel pages."""
    render_page_hero(title, caption)


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
        .aic-conn-light {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 600;
            font-size: 0.98rem;
            line-height: 1.2;
        }
        .aic-glow-dot {
            width: 0.85rem;
            height: 0.85rem;
            border-radius: 50%;
            flex-shrink: 0;
            box-shadow: 0 0 0 0 transparent;
        }
        .aic-glow-ok {
            background: #22c55e;
            box-shadow: 0 0 8px 2px rgba(34, 197, 94, 0.85),
                        0 0 18px 4px rgba(34, 197, 94, 0.45);
            animation: aic-glow-pulse-ok 1.8s ease-in-out infinite;
        }
        .aic-glow-bad {
            background: #ef4444;
            box-shadow: 0 0 8px 2px rgba(239, 68, 68, 0.85),
                        0 0 18px 4px rgba(239, 68, 68, 0.45);
            animation: aic-glow-pulse-bad 1.8s ease-in-out infinite;
        }
        .aic-glow-warn {
            background: #f59e0b;
            box-shadow: 0 0 8px 2px rgba(245, 158, 11, 0.75),
                        0 0 16px 4px rgba(245, 158, 11, 0.35);
            animation: aic-glow-pulse-warn 1.8s ease-in-out infinite;
        }
        @keyframes aic-glow-pulse-ok {
            0%, 100% { box-shadow: 0 0 6px 1px rgba(34, 197, 94, 0.55), 0 0 12px 3px rgba(34, 197, 94, 0.25); }
            50% { box-shadow: 0 0 10px 3px rgba(34, 197, 94, 0.95), 0 0 22px 6px rgba(34, 197, 94, 0.5); }
        }
        @keyframes aic-glow-pulse-bad {
            0%, 100% { box-shadow: 0 0 6px 1px rgba(239, 68, 68, 0.55), 0 0 12px 3px rgba(239, 68, 68, 0.25); }
            50% { box-shadow: 0 0 10px 3px rgba(239, 68, 68, 0.95), 0 0 22px 6px rgba(239, 68, 68, 0.5); }
        }
        @keyframes aic-glow-pulse-warn {
            0%, 100% { box-shadow: 0 0 6px 1px rgba(245, 158, 11, 0.45), 0 0 12px 3px rgba(245, 158, 11, 0.2); }
            50% { box-shadow: 0 0 10px 3px rgba(245, 158, 11, 0.9), 0 0 20px 5px rgba(245, 158, 11, 0.45); }
        }
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
        /* Review markers are sibling element-containers in Streamlit (markdown
           does not wrap following widgets). Target with :has() + adjacent sibling. */
        .element-container:has(.aic-review-layout) + .element-container
            [data-testid="stHorizontalBlock"] {
            align-items: flex-start !important;
        }
        .aic-review-layout,
        .aic-review-canvas {
            height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            overflow: hidden !important;
            background: transparent !important;
            box-shadow: none !important;
        }
        .element-container:has(.aic-review-canvas) + .element-container {
            border: 1px solid rgba(128, 128, 128, 0.28) !important;
            border-radius: 12px !important;
            padding: 0.35rem !important;
            background: rgba(0, 0, 0, 0.18) !important;
            width: 100% !important;
            margin: 0.15rem 0 0.45rem 0 !important;
        }
        .element-container:has(.aic-review-canvas) + .element-container
            [data-testid="stImage"],
        .element-container:has(.aic-review-canvas) + .element-container
            [data-testid="stImage"] > div {
            width: 100% !important;
            max-width: 100% !important;
        }
        .element-container:has(.aic-review-canvas) + .element-container img {
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            max-height: min(82vh, 920px) !important;
            object-fit: contain !important;
            display: block !important;
            margin: 0 auto !important;
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
        /* Review stage: keep inspector controls compact so the canvas can dominate */
        .element-container:has(.aic-review-compact) ~ .element-container
            [data-testid="stHorizontalBlock"] button[kind="primary"],
        .element-container:has(.aic-review-compact) ~ .element-container
            [data-testid="stHorizontalBlock"] button[kind="secondary"] {
            min-height: 2.35rem !important;
            border-radius: 10px !important;
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
        /* Non-review image cards keep a moderate height. */
        .element-container:has(.aic-img-card):not(:has(.aic-review-canvas))
            + .element-container img {
            max-height: 68vh;
            object-fit: contain;
        }
        /* Shared page hero for every surface */
        .aic-page-hero,
        .aic-dash-hero,
        .aic-admin-hero,
        .aic-settings-head {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 16px;
            padding: 0.95rem 1.1rem 1rem 1.1rem;
            margin: 0 0 0.85rem 0;
            background:
                linear-gradient(135deg, rgba(255,75,75,0.12), transparent 42%),
                linear-gradient(225deg, rgba(33,150,243,0.12), transparent 48%),
                linear-gradient(15deg, rgba(46,160,67,0.10), transparent 40%),
                rgba(250,250,250,0.03);
        }
        .aic-page-hero .aic-rgb-bar,
        .aic-dash-hero .aic-rgb-bar,
        .aic-admin-hero .aic-rgb-bar,
        .aic-settings-head .aic-rgb-bar,
        section[data-testid="stSidebar"] .aic-side-brand .aic-rgb-bar {
            display: block;
            height: 3px;
            border-radius: 999px;
            margin: 0 0 0.55rem 0;
            background: linear-gradient(
                90deg,
                #ff4b4b 0%,
                #2196f3 25%,
                #2ea043 50%,
                #2196f3 75%,
                #ff4b4b 100%
            );
            background-size: 220% 100%;
            animation: aic-rgb-flow 5.5s linear infinite;
        }
        @keyframes aic-rgb-flow {
            0% { background-position: 0% 50%; }
            100% { background-position: 220% 50%; }
        }
        @keyframes aic-rgb-flow-vertical {
            0% { background-position: 50% 0%; }
            100% { background-position: 50% 220%; }
        }
        @media (prefers-reduced-motion: reduce) {
            .aic-page-hero .aic-rgb-bar,
            .aic-dash-hero .aic-rgb-bar,
            .aic-admin-hero .aic-rgb-bar,
            .aic-settings-head .aic-rgb-bar,
            section[data-testid="stSidebar"] .aic-side-brand .aic-rgb-bar,
            section[data-testid="stSidebar"]::before {
                animation: none !important;
                background-size: 100% 100% !important;
            }
            .aic-page-hero .aic-rgb-bar,
            .aic-dash-hero .aic-rgb-bar,
            .aic-admin-hero .aic-rgb-bar,
            .aic-settings-head .aic-rgb-bar,
            section[data-testid="stSidebar"] .aic-side-brand .aic-rgb-bar {
                background: linear-gradient(90deg, #ff4b4b 0%, #2196f3 50%, #2ea043 100%);
            }
            section[data-testid="stSidebar"]::before {
                background: linear-gradient(180deg, #ff4b4b 0%, #2196f3 50%, #2ea043 100%);
            }
        }
        .aic-page-hero h1,
        .aic-page-hero h2,
        .aic-dash-hero h1,
        .aic-admin-hero h1,
        .aic-admin-hero h2,
        .aic-settings-head h3 {
            margin: 0 0 0.25rem 0;
            font-size: 1.45rem;
            font-weight: 750;
            letter-spacing: 0.01em;
        }
        .aic-dash-hero h1 {
            font-size: 1.65rem;
        }
        .aic-page-hero p,
        .aic-dash-hero p,
        .aic-admin-hero p,
        .aic-settings-head p {
            margin: 0;
            opacity: 0.78;
            font-size: 0.92rem;
            line-height: 1.45;
            max-width: 42rem;
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
        section[data-testid="stSidebar"] {
            position: relative;
            min-width: 6.5rem !important;
            max-width: 6.5rem !important;
            overflow: visible !important;
        }
        /* Vertical RGB edge on the left panel */
        section[data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 3px;
            height: 100%;
            border-radius: 999px 0 0 999px;
            z-index: 6;
            pointer-events: none;
            background: linear-gradient(
                180deg,
                #ff4b4b 0%,
                #2196f3 25%,
                #2ea043 50%,
                #2196f3 75%,
                #ff4b4b 100%
            );
            background-size: 100% 220%;
            animation: aic-rgb-flow-vertical 5.5s linear infinite;
        }
        section[data-testid="stSidebar"] > div {
            padding-left: 0.45rem;
            padding-right: 0.45rem;
        }
        section[data-testid="stSidebar"] .aic-side-brand {
            margin: 0.15rem 0 0.65rem 0;
        }
        section[data-testid="stSidebar"] .aic-side-brand .aic-rgb-bar {
            margin-bottom: 0;
        }
        section[data-testid="stSidebar"] .stButton > button {
            min-height: 2.55rem;
            padding: 0.4rem 0.25rem;
            justify-content: center;
            font-size: 0.78rem;
        }
        /* Icon-only nav buttons use a zero-width label; keep Material glyphs centered. */
        section[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {
            font-size: 1.35rem;
            line-height: 1;
        }
        section[data-testid="stSidebar"] .aic-side-spacer {
            min-height: 0.75rem;
        }
        section[data-testid="stSidebar"] .aic-side-profile {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.35rem;
        }
        .aic-role-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.18rem 0.48rem;
            border-radius: 999px;
            border: 1px solid transparent;
            flex-shrink: 0;
        }
        .aic-role-badge.aic-role-admin {
            background: rgba(56, 189, 248, 0.16);
            border-color: rgba(56, 189, 248, 0.45);
            color: #7dd3fc;
        }
        .aic-role-badge.aic-role-user {
            background: rgba(163, 230, 53, 0.14);
            border-color: rgba(163, 230, 53, 0.4);
            color: #bef264;
        }
        /* Administration console */
        .aic-admin-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.15rem 0 0.85rem 0;
        }
        .aic-admin-metrics-3 {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .aic-admin-metric {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 12px;
            padding: 0.7rem 0.8rem;
            background: rgba(250,250,250,0.03);
            min-height: 4.4rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.15rem;
        }
        .aic-admin-metric .val {
            font-size: 1.45rem;
            font-weight: 750;
            line-height: 1.1;
            letter-spacing: 0.01em;
        }
        .aic-admin-metric .lbl {
            font-size: 0.78rem;
            opacity: 0.72;
            font-weight: 600;
        }
        .aic-admin-section {
            margin: 0.35rem 0 0.65rem 0;
        }
        .aic-admin-section h4 {
            margin: 0 0 0.2rem 0;
            font-size: 1.02rem;
            font-weight: 700;
        }
        .aic-admin-section p {
            margin: 0;
            font-size: 0.84rem;
            opacity: 0.72;
            line-height: 1.4;
        }
        .aic-admin-panel {
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin: 0 0 0.85rem 0;
            background: rgba(250,250,250,0.025);
        }
        .aic-admin-activity {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .aic-admin-activity-row {
            display: grid;
            grid-template-columns: 9.5rem 7rem 1fr auto;
            gap: 0.65rem;
            align-items: center;
            padding: 0.45rem 0.55rem;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.14);
            background: rgba(128,128,128,0.04);
            font-size: 0.82rem;
        }
        .aic-admin-activity-row .when { opacity: 0.7; font-variant-numeric: tabular-nums; }
        .aic-admin-activity-row .actor { font-weight: 650; }
        .aic-admin-activity-row .event { opacity: 0.88; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .aic-admin-activity-row .outcome {
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 0.12rem 0.4rem;
            border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.28);
        }
        .aic-admin-activity-row .outcome.ok {
            border-color: rgba(46,160,67,0.45);
            background: rgba(46,160,67,0.12);
            color: #7dcea0;
        }
        .aic-admin-activity-row .outcome.bad {
            border-color: rgba(229,83,75,0.45);
            background: rgba(229,83,75,0.12);
            color: #f0a8a4;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.25rem;
            flex-wrap: wrap;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            padding-top: 0.45rem;
            padding-bottom: 0.45rem;
        }
        @media (max-width: 900px) {
            .aic-admin-metrics, .aic-admin-metrics-3 {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .aic-admin-activity-row {
                grid-template-columns: 1fr 1fr;
                gap: 0.25rem 0.5rem;
            }
        }
    """
    st.markdown(
        base_css + _theme_override_css(theme) + "\n        </style>\n        ",
        unsafe_allow_html=True,
    )


def render_page_toolbar(
    *,
    mode: Literal["home", "wizard", "panel"] | str = "home",
    title: str | None = None,
    on_settings: Callable[[], None] | None = None,
    on_start_fresh: Callable[[], None] | None = None,
    on_back: Callable[[], None] | None = None,
) -> None:
    st = _st()
    # Navigation lives in the left sidebar; page titles use render_page_hero.
    del on_settings, on_back, title
    if mode == "wizard" and on_start_fresh:
        if st.button("Start Fresh", key="toolbar_start_fresh", width="stretch"):
            on_start_fresh()



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
