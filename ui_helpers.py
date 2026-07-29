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
    "default_form",
    "get_settings_section_from_label",
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
    "reset_active_analysis",
]


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


def inject_css() -> None:
    st = _st()
    st.markdown(
        """
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
        </style>
        """,
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
                use_container_width=True,
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
                use_container_width=True,
            ):
                on_start_fresh()
        with right:
            if on_settings and st.button(
                "Settings",
                key="toolbar_settings_wizard",
                use_container_width=True,
            ):
                on_settings()
    else:
        left, right = st.columns([5.5, 1.5], vertical_alignment="center")
        with left:
            if on_back and st.button(
                "← Back to Inventory Counter",
                key="toolbar_back_settings",
                use_container_width=True,
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
            if st.button("← Back", use_container_width=True, key=f"{key_prefix}_back"):
                navigate_to("wizard", stage=back_stage)
    with right:
        if on_next is not None or next_stage:
            if st.button(
                next_label,
                type="primary",
                use_container_width=True,
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
