"""UI helpers and page navigation for AI Inventory Counter.

Constants live in app_constants.py (no Streamlit).
This module must not be imported by app_constants.
Streamlit is imported lazily via `_st()` only inside functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from app_constants import (
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
    get_settings_section_from_label,
)

# Re-export so `from ui_helpers import SETTINGS_SECTION_LABELS` still works
# AFTER this module has finished importing app_constants (always safe).
__all__ = [
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
    "render_stepper",
    "render_theme_toggle",
    "reset_active_analysis",
    "set_ui_theme",
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
        # Unset until the user clicks Fence Panels on Inventory Setup.
        "inventory_choice": "",
        "inventory_custom": "",
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
    if mapped in {"welcome", "wizard", "settings"}:
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


def reset_active_analysis(*, go_home: bool = True, start_wizard: bool = False) -> None:
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
    if go_home:
        st.session_state.app_view = "welcome"
        st.session_state.wizard_stage = "setup"
    elif start_wizard:
        st.session_state.app_view = "wizard"
        st.session_state.wizard_stage = "setup"
    st.rerun()


def get_ui_theme() -> str:
    st = _st()
    theme = str(st.session_state.get("aic_theme") or DEFAULT_UI_THEME).lower()
    return theme if theme in _UI_THEMES else DEFAULT_UI_THEME


def set_ui_theme(theme: str) -> None:
    st = _st()
    next_theme = str(theme or DEFAULT_UI_THEME).lower()
    st.session_state.aic_theme = next_theme if next_theme in _UI_THEMES else DEFAULT_UI_THEME


def render_theme_toggle(*, key: str = "aic_theme_toggle_btn") -> None:
    """Compact Dark/Light toggle; does not reuse the aic_theme session key."""
    st = _st()
    current = get_ui_theme()
    label = "Light theme" if current == "dark" else "Dark theme"
    if st.button(label, key=key, width="stretch"):
        set_ui_theme("light" if current == "dark" else "dark")
        st.rerun()


def _theme_override_css(theme: str) -> str:
    if theme == "light":
        return """
        /* Light theme overrides (session toggle) */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        section.main {
            background-color: #ffffff !important;
            color: #31333F !important;
        }
        [data-testid="stHeader"] {
            background: rgba(255,255,255,0.92) !important;
        }
        [data-testid="stToolbar"] {
            background: transparent !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #f0f2f6 !important;
            color: #31333F !important;
        }
        .stMarkdown, .stCaption, .stText, label, p, span, li {
            color: inherit;
        }
        div[data-testid="stMetric"] {
            background: rgba(49,51,63,0.04) !important;
            border-color: rgba(49,51,63,0.12) !important;
        }
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #31333F !important;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background-color: #f0f2f6 !important;
            color: #31333F !important;
        }
        .aic-card, .aic-inv-card, .aic-empty, .aic-model-card {
            background: rgba(49,51,63,0.04) !important;
            border-color: rgba(49,51,63,0.14) !important;
        }
        .aic-toolbar-title, .aic-hero-title, .aic-inv-card-title {
            color: #31333F !important;
        }
        .aic-hero-sub, .aic-muted, .aic-note, .aic-analyze-status {
            color: rgba(49,51,63,0.72) !important;
        }
        hr, [data-testid="stDivider"] {
            border-color: rgba(49,51,63,0.14) !important;
        }
        """
    return """
        /* Dark theme overrides (session toggle / default) */
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
            margin: 0.55rem 0 1rem 0; font-size: 0.78rem;
        }
        .aic-pill {
            padding: 0.28rem 0.62rem; border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.28); opacity: 0.5;
        }
        .aic-pill.done { opacity: 0.88; background: rgba(46,160,67,0.14); }
        .aic-pill.current {
            opacity: 1; font-weight: 650;
            background: rgba(255,75,75,0.14);
            border-color: rgba(255,75,75,0.45);
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
            grid-template-columns: 1fr 1fr;
            gap: 0.4rem;
            margin: 0.35rem 0 0.65rem 0;
        }
        .aic-metric-tile {
            border: 1px solid rgba(46,160,67,0.28);
            border-radius: 10px;
            padding: 0.4rem 0.55rem;
            background: rgba(46,160,67,0.06);
            font-size: 0.82rem;
        }
        .aic-metric-tile b { display: block; font-size: 1.15rem; margin-top: 0.1rem; }
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
            border: 1px solid rgba(33,150,243,0.28);
            border-radius: 12px;
            padding: 0.55rem 0.65rem;
            background: linear-gradient(
                135deg,
                rgba(255,75,75,0.06),
                rgba(33,150,243,0.08),
                rgba(46,160,67,0.08)
            );
            margin: 0.35rem 0 0.5rem 0;
        }
        .aic-photo-strip-title {
            font-size: 0.85rem;
            font-weight: 650;
            margin: 0 0 0.35rem 0;
        }
        .aic-sample-card {
            border: 1px solid rgba(33,150,243,0.22);
            border-radius: 10px;
            padding: 0.35rem;
            background: rgba(33,150,243,0.06);
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
            background: linear-gradient(
                90deg,
                rgba(255,75,75,0.05),
                rgba(33,150,243,0.06),
                rgba(46,160,67,0.05)
            );
        }
        .aic-model-pick-selected {
            border-color: rgba(255,75,75,0.55);
            box-shadow: 0 0 0 1px rgba(255,75,75,0.25);
        }
        .aic-rgb-accent {
            height: 3px;
            border-radius: 999px;
            margin: 0.15rem 0 0.55rem 0;
            background: linear-gradient(90deg, #ff4b4b 0%, #2196f3 50%, #2ea043 100%);
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
        left, theme_col, right = st.columns([5.2, 1.5, 1.4], vertical_alignment="center")
        with left:
            st.markdown(
                '<p class="aic-toolbar-title">AI Inventory Counter</p>',
                unsafe_allow_html=True,
            )
        with theme_col:
            render_theme_toggle(key="aic_theme_toggle_home")
        with right:
            if on_settings and st.button(
                "Settings",
                key="toolbar_settings_home",
                width="stretch",
            ):
                on_settings()
    elif mode == "wizard":
        left, mid, theme_col, right = st.columns(
            [4.2, 1.35, 1.45, 1.25], vertical_alignment="center"
        )
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
        with theme_col:
            render_theme_toggle(key="aic_theme_toggle_wizard")
        with right:
            if on_settings and st.button(
                "Settings",
                key="toolbar_settings_wizard",
                width="stretch",
            ):
                on_settings()
    else:
        left, theme_col, right = st.columns([4.6, 1.5, 1.6], vertical_alignment="center")
        with left:
            if on_back and st.button(
                "← Back to Inventory Counter",
                key="toolbar_back_settings",
                width="stretch",
            ):
                on_back()
        with theme_col:
            render_theme_toggle(key="aic_theme_toggle_settings")
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
    st.markdown(f'<div class="aic-stepper">{"".join(pills)}</div>', unsafe_allow_html=True)
    idx = STAGES.index(current) if current in STAGES else 0
    st.caption(f"Step {idx + 1} of {len(STAGES)} · {STAGE_LABELS.get(current, current)}")


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
