"""Streamlit UI for experimental local Shape Detection.

Kept out of app.py — the main app only authenticates, routes, and calls
``render_shape_detection_page``.
"""

from __future__ import annotations

import copy
import io
import uuid
from typing import Any

import json

import pandas as pd
import streamlit as st

import auth_session
from shape_detection import (
    EXPERIMENTAL_NOTICE,
    MSG_NO_IMAGE,
    annotate_circles,
    decode_bgr,
    encode_image,
    generate_synthetic_circle_sample,
    run_shape_detection,
    sanitize_upload_filename,
    validate_shape_image_bytes,
    ShapeDetectionError,
)
from shape_detection_models import (
    MODE_LABELS,
    REVIEW_STATUS_LABELS,
    TARGET_LABELS,
    ShapeDetectionResult,
    ShapeDetectionSettings,
    apply_mode_presets,
    balanced_defaults,
    build_cache_key,
)
from shape_detection_storage import (
    ShapeAuthError,
    ShapeStorageError,
    export_csv,
    export_json,
    get_shape_test,
    get_shape_test_items,
    list_shape_tests,
    load_annotated_image_bytes,
    result_from_saved_run,
    save_shape_test,
    shape_detection_allowed,
)
from shape_registry import (
    UNSUPPORTED_SHAPE_MESSAGE,
    ShapeResolutionError,
    coming_soon_shapes,
    preset_options,
    resolve_shape,
)
from ui_helpers import navigate_to, render_page_hero

# Session-state namespace
SS_PREFIX = "shape_detection_"
SS_IMAGE = "shape_detection_image"
SS_IMAGE_META = "shape_detection_image_meta"
SS_RESULT = "shape_detection_result"
SS_SETTINGS = "shape_detection_settings"
SS_SELECTED = "shape_detection_selected_item"
SS_REVIEW = "shape_detection_review"
SS_RUN_ID = "shape_detection_run_id"
SS_EXECUTING = "shape_detection_executing"
SS_CACHE = "shape_detection_cache"
SS_SOURCE = "shape_detection_source_type"
SS_FILENAME = "shape_detection_filename"
SS_ANNOTATION = "shape_detection_annotation_style"
SS_VIEW_MODE = "shape_detection_view_mode"
SS_SHAPE_TEXT = "shape_detection_shape_text"
SS_ANN_BYTES = "shape_detection_annotated_bytes"
SS_TECH = "shape_detection_tech_error"


def clear_shape_detection_state() -> None:
    """Drop all Shape Detection session keys. Does not touch auth or OpenRouter."""
    for key in list(st.session_state.keys()):
        if str(key).startswith("shape_detection"):
            st.session_state.pop(key, None)


def _settings() -> ShapeDetectionSettings:
    raw = st.session_state.get(SS_SETTINGS)
    if isinstance(raw, ShapeDetectionSettings):
        return raw
    if isinstance(raw, dict):
        base = balanced_defaults()
        for k, v in raw.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base
    s = balanced_defaults()
    st.session_state[SS_SETTINGS] = s
    return s


def _store_settings(settings: ShapeDetectionSettings) -> None:
    st.session_state[SS_SETTINGS] = settings


def _result() -> ShapeDetectionResult | None:
    value = st.session_state.get(SS_RESULT)
    return value if isinstance(value, ShapeDetectionResult) else None


def _apply_review_to_result(result: ShapeDetectionResult) -> ShapeDetectionResult:
    """Merge data_editor / review map into detection included flags."""
    review = st.session_state.get(SS_REVIEW) or {}
    if not isinstance(review, dict):
        return result
    for det in result.detections:
        row = review.get(det.id) or review.get(str(det.sequence_number))
        if not isinstance(row, dict):
            continue
        if "included" in row:
            det.included = bool(row["included"])
        if "review_status" in row:
            status = str(row["review_status"]).strip().lower().replace(" ", "_")
            if status in REVIEW_STATUS_LABELS:
                det.review_status = status  # type: ignore[assignment]
            else:
                # Accept display labels
                for key, label in REVIEW_STATUS_LABELS.items():
                    if label.lower() == str(row["review_status"]).lower():
                        det.review_status = key  # type: ignore[assignment]
                        break
        if det.review_status in {"false_positive", "duplicate", "ignore"}:
            det.included = False
    result.manually_added_count = int(
        st.session_state.get("shape_detection_manual_add", 0) or 0
    )
    result.manual_notes = str(
        st.session_state.get("shape_detection_manual_notes", "") or ""
    )
    return result


def render_shape_detection_page(user=None) -> None:
    user = user or auth_session.current_user()
    allowed, message = shape_detection_allowed(user)
    if not allowed:
        render_page_hero("Shape Detection", "Testing Phase")
        st.error(message)
        if st.button("Back to Dashboard", key="shape_back_denied"):
            clear_shape_detection_state()
            navigate_to("welcome")
            st.rerun()
        return

    render_page_hero(
        "Shape Detection",
        "Testing Phase",
    )
    st.caption(
        "Detect likely visible shapes using local computer vision. "
        "No API key or paid inference is required."
    )
    st.info(
        "Detect likely visible shapes and objects. "
        "Review the results before using the final count."
    )

    top = st.columns([1, 1, 2])
    with top[0]:
        if st.button("Back to Dashboard", key="shape_back_dashboard"):
            # Leaving the page must not clear OpenRouter or inventory wizard.
            navigate_to("welcome")
            st.rerun()
    with top[1]:
        if st.button("Reset", key="shape_reset"):
            clear_shape_detection_state()
            st.rerun()

    tabs = st.tabs(["Setup", "Results", "Review", "History"])
    with tabs[0]:
        _render_setup(user)
    with tabs[1]:
        _render_results()
    with tabs[2]:
        _render_review()
    with tabs[3]:
        _render_history(user)


def _render_setup(user) -> None:
    st.markdown("##### 1. Detection Target")
    st.markdown("**What shape do you want to detect?**")
    presets = preset_options() or ["Circles"]
    c1, c2 = st.columns(2)
    with c1:
        preset = st.selectbox("Preset", presets, key="shape_detection_preset")
    with c2:
        shape_text = st.text_input(
            "Or type a shape",
            value=st.session_state.get(SS_SHAPE_TEXT, ""),
            key="shape_detection_shape_input",
            placeholder="e.g. circles, rectangles, triangles…",
        )
        st.session_state[SS_SHAPE_TEXT] = shape_text

    requested = (shape_text or preset or "").strip()
    shape_ok = False
    resolved_key = ""
    try:
        if requested:
            spec = resolve_shape(requested)
            shape_ok = True
            resolved_key = spec.key
            st.caption(f"Will use: **{spec.display_name}**")
    except ShapeResolutionError as exc:
        st.warning(str(exc))
        soon = ", ".join(s.display_name for s in coming_soon_shapes())
        if soon:
            st.caption(f"Coming soon: {soon}")

    settings = _settings()
    if resolved_key == "circle":
        st.markdown("**What kinds of circles should be detected?**")
        target_labels = list(TARGET_LABELS.values())
        target_keys = list(TARGET_LABELS.keys())
        target_idx = (
            target_keys.index(settings.target_type)
            if settings.target_type in target_keys
            else 2
        )
        target_label = st.radio(
            "Target type",
            target_labels,
            index=target_idx,
            horizontal=True,
            key="shape_detection_target_radio",
            label_visibility="collapsed",
        )
        settings.target_type = target_keys[target_labels.index(target_label)]  # type: ignore[assignment]

    st.markdown("##### 2. Image")
    src = st.radio(
        "Image source",
        ["Upload Image", "Camera", "Built-in Test Sample"],
        horizontal=True,
        key="shape_detection_image_source",
    )
    image_bytes: bytes | None = st.session_state.get(SS_IMAGE)
    if src == "Upload Image":
        up = st.file_uploader(
            "Upload JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            key="shape_detection_uploader",
        )
        if up is not None:
            try:
                data = up.getvalue()
                meta = validate_shape_image_bytes(data)
                fname = sanitize_upload_filename(up.name)
                st.session_state[SS_IMAGE] = data
                st.session_state[SS_IMAGE_META] = meta
                st.session_state[SS_SOURCE] = "upload"
                st.session_state[SS_FILENAME] = fname
                # New image — clear prior result but keep settings
                st.session_state.pop(SS_RESULT, None)
                st.session_state.pop(SS_ANN_BYTES, None)
                image_bytes = data
            except ShapeDetectionError as exc:
                st.error(str(exc))
    elif src == "Camera":
        cam = st.camera_input("Capture", key="shape_detection_camera")
        if cam is not None:
            try:
                data = cam.getvalue()
                meta = validate_shape_image_bytes(data)
                st.session_state[SS_IMAGE] = data
                st.session_state[SS_IMAGE_META] = meta
                st.session_state[SS_SOURCE] = "camera"
                st.session_state[SS_FILENAME] = "camera_capture"
                st.session_state.pop(SS_RESULT, None)
                st.session_state.pop(SS_ANN_BYTES, None)
                image_bytes = data
            except ShapeDetectionError as exc:
                st.error(str(exc))
    else:
        st.caption(
            "Synthetic sample with filled and outlined circles "
            "(generated locally — not downloaded)."
        )
        if st.button("Load built-in test sample", key="shape_load_sample"):
            data, expected = generate_synthetic_circle_sample()
            meta = validate_shape_image_bytes(data)
            st.session_state[SS_IMAGE] = data
            st.session_state[SS_IMAGE_META] = meta
            st.session_state[SS_SOURCE] = "synthetic"
            st.session_state[SS_FILENAME] = "synthetic_circles"
            st.session_state["shape_detection_sample_expected"] = expected
            st.session_state.pop(SS_RESULT, None)
            st.session_state.pop(SS_ANN_BYTES, None)
            st.rerun()
        image_bytes = st.session_state.get(SS_IMAGE)

    if image_bytes:
        meta = st.session_state.get(SS_IMAGE_META) or {}
        st.image(image_bytes, caption="Preview (detection has not run yet)", width="stretch")
        st.caption(
            f"{meta.get('width', '?')}×{meta.get('height', '?')} · "
            f"{meta.get('size_bytes', 0) / 1024:.0f} KB"
        )
        if st.session_state.get("shape_detection_sample_expected"):
            st.caption(
                f"Known synthetic count: {st.session_state['shape_detection_sample_expected']} "
                "(for your reference only)."
            )
    else:
        st.caption(MSG_NO_IMAGE)

    st.markdown("##### 3. Detection Settings")
    mode_labels = list(MODE_LABELS.values())
    mode_keys = list(MODE_LABELS.keys())
    mode_idx = mode_keys.index(settings.mode) if settings.mode in mode_keys else 1
    mode_label = st.selectbox(
        "Detection mode",
        mode_labels,
        index=mode_idx,
        key="shape_detection_mode",
        help="Strict = fewer false positives. Sensitive = more candidates.",
    )
    settings.mode = mode_keys[mode_labels.index(mode_label)]  # type: ignore[assignment]
    settings = apply_mode_presets(settings)

    size_mode = st.radio(
        "Size range",
        ["Auto", "Custom"],
        horizontal=True,
        index=0 if settings.size_mode == "auto" else 1,
        key="shape_detection_size_mode",
    )
    settings.size_mode = "auto" if size_mode == "Auto" else "custom"  # type: ignore[assignment]
    if settings.size_mode == "custom":
        sc1, sc2 = st.columns(2)
        with sc1:
            settings.min_diameter_pct = float(
                st.number_input(
                    "Minimum diameter (% of short side)",
                    min_value=0.5,
                    max_value=90.0,
                    value=float(settings.min_diameter_pct),
                    step=0.5,
                    key="shape_min_diam_pct",
                )
            )
        with sc2:
            settings.max_diameter_pct = float(
                st.number_input(
                    "Maximum diameter (% of short side)",
                    min_value=1.0,
                    max_value=95.0,
                    value=float(settings.max_diameter_pct),
                    step=0.5,
                    key="shape_max_diam_pct",
                )
            )
        if settings.max_diameter_pct <= settings.min_diameter_pct:
            st.error("Maximum diameter must be greater than minimum.")

    with st.expander("Advanced Settings", expanded=False):
        settings.include_partial = st.checkbox(
            "Include partially visible circles at image edges",
            value=bool(settings.include_partial),
            key="shape_include_partial",
        )
        settings.count_concentric_separately = st.checkbox(
            "Count concentric circles separately",
            value=bool(settings.count_concentric_separately),
            key="shape_concentric",
            help="Off (default): ring-like inner/outer edges count as one object.",
        )
        settings.use_hough = st.checkbox(
            "Use Hough circle detector",
            value=bool(settings.use_hough),
            key="shape_use_hough",
            help="Edge-sensitive circle transform (good for rings and round objects).",
        )
        settings.use_contour = st.checkbox(
            "Use contour circularity detector",
            value=bool(settings.use_contour),
            key="shape_use_contour",
            help="Finds blob/outline shapes with near-circular geometry.",
        )
        settings.min_center_distance_pct = float(
            st.slider(
                "Minimum center distance (% of short side)",
                1.0,
                20.0,
                float(settings.min_center_distance_pct),
                key="shape_min_center_dist",
            )
        )
        settings.edge_sensitivity = float(
            st.slider(
                "Edge sensitivity",
                40.0,
                200.0,
                float(settings.edge_sensitivity),
                key="shape_edge_sens",
                help="Higher values require stronger edges (fewer detections).",
            )
        )
        settings.hough_accumulator = float(
            st.slider(
                "Hough accumulator sensitivity",
                8.0,
                60.0,
                float(settings.hough_accumulator),
                key="shape_hough_acc",
                help="Lower values accept weaker circle evidence (more detections).",
            )
        )
        settings.contour_circularity = float(
            st.slider(
                "Contour circularity threshold",
                0.50,
                0.95,
                float(settings.contour_circularity),
                key="shape_circ_thr",
                help="4π·area / perimeter² — higher means rounder shapes only.",
            )
        )
        if st.button("Reset to Balanced Defaults", key="shape_reset_defaults"):
            _store_settings(balanced_defaults())
            st.rerun()

    _store_settings(settings)

    st.markdown("##### 4. Run Detection")
    needs_circle_engine = resolved_key == "circle"
    can_run = (
        bool(image_bytes)
        and shape_ok
        and not st.session_state.get(SS_EXECUTING)
        and (
            (not needs_circle_engine)
            or settings.use_hough
            or settings.use_contour
        )
        and (
            settings.size_mode == "auto"
            or settings.max_diameter_pct > settings.min_diameter_pct
        )
    )
    run_label = "Run Detection"
    if shape_ok and requested:
        try:
            run_label = f"Detect {resolve_shape(requested).display_name}"
        except ShapeResolutionError:
            run_label = "Run Detection"
    if st.button(
        run_label,
        type="primary",
        disabled=not can_run,
        key="shape_detect_btn",
    ):
        _execute_detection(requested, image_bytes, settings)


def _execute_detection(
    requested: str,
    image_bytes: bytes | None,
    settings: ShapeDetectionSettings,
) -> None:
    if st.session_state.get(SS_EXECUTING):
        return
    if not image_bytes:
        st.error(MSG_NO_IMAGE)
        return
    try:
        resolve_shape(requested)
    except ShapeResolutionError as exc:
        st.error(str(exc))
        return

    run_id = str(uuid.uuid4())
    st.session_state[SS_RUN_ID] = run_id
    st.session_state[SS_EXECUTING] = True
    progress = st.progress(0, text="Preparing image")
    steps = [
        "Preparing image",
        "Finding circular boundaries",
        "Evaluating circular shapes",
        "Removing duplicate detections",
        "Preparing results",
    ]
    step_state = {"i": 0}

    def on_progress(label: str) -> None:
        if label in steps:
            step_state["i"] = steps.index(label)
        progress.progress(
            min(1.0, (step_state["i"] + 1) / len(steps)),
            text=label,
        )

    try:
        cache = st.session_state.setdefault(SS_CACHE, {})
        meta = st.session_state.get(SS_IMAGE_META) or validate_shape_image_bytes(
            image_bytes
        )
        key = build_cache_key(meta["hash"], resolve_shape(requested).key, settings)
        if key in cache and isinstance(cache[key], ShapeDetectionResult):
            result = copy.deepcopy(cache[key])
            progress.progress(1.0, text="Preparing results")
        else:
            result = run_shape_detection(
                image_bytes,
                requested_shape=requested,
                settings=settings,
                progress=on_progress,
            )
            # Per-session cache only (never shared across users)
            cache[key] = copy.deepcopy(result)
            # Bound cache size
            while len(cache) > 8:
                cache.pop(next(iter(cache)))

        st.session_state[SS_RESULT] = result
        st.session_state[SS_REVIEW] = {
            d.id: {
                "included": d.included,
                "review_status": d.review_status,
            }
            for d in result.detections
        }
        st.session_state.pop(SS_ANN_BYTES, None)
        st.session_state.pop(SS_TECH, None)
        if result.warning:
            st.warning(result.warning)
        else:
            label = result.normalized_shape or "shape"
            st.success(
                f"Detected {result.included_count} {label}(s)."
            )
    except ShapeDetectionError as exc:
        st.session_state[SS_TECH] = exc.technical or str(exc)
        st.error(str(exc))
        if exc.technical:
            with st.expander("Technical Details"):
                st.code(exc.technical)
    finally:
        st.session_state[SS_EXECUTING] = False
        progress.empty()


def _annotated_bytes(
    result: ShapeDetectionResult,
    style: str,
    *,
    solo: bool = False,
) -> bytes:
    image_bytes = st.session_state.get(SS_IMAGE)
    if not image_bytes:
        raise ShapeDetectionError(MSG_NO_IMAGE)
    selected = st.session_state.get(SS_SELECTED)
    cache_key = (
        f"{result.image_hash}:{style}:{result.included_count}:{selected}:solo={solo}"
    )
    cached = st.session_state.get(SS_ANN_BYTES)
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return cached["bytes"]
    image = decode_bgr(image_bytes)
    ann = annotate_circles(
        image,
        result.detections,
        style=style,
        selected_id=selected,
        solo=solo,
    )
    raw = encode_image(ann, fmt="png")
    st.session_state[SS_ANN_BYTES] = {"key": cache_key, "bytes": raw}
    return raw


def _render_count_preview(result: ShapeDetectionResult) -> None:
    """Compact numbered strip — one chip per included detection."""
    included = [d for d in result.detections if d.included]
    st.markdown("##### Count preview")
    shape_name = (result.normalized_shape or "shape").title()
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:0.75rem;margin:0 0 0.55rem 0'>"
        f"<div style='font-size:2.35rem;font-weight:750;line-height:1'>{result.final_count}</div>"
        f"<div style='opacity:0.75;font-size:0.95rem'>{shape_name} · "
        f"{result.included_count} included"
        f"{f' · +{result.manually_added_count} manual' if result.manually_added_count else ''}"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if not included:
        st.caption("No included detections to preview.")
        return

    st.caption("Tap a number to select it. Use Solo view to inspect that item alone.")
    # Wrap chips in rows of up to 8
    per_row = 8
    for row_start in range(0, len(included), per_row):
        chunk = included[row_start : row_start + per_row]
        cols = st.columns(len(chunk))
        for col, det in zip(cols, chunk):
            with col:
                selected = st.session_state.get(SS_SELECTED) == det.id
                label = f"#{det.sequence_number}"
                if st.button(
                    label,
                    key=f"shape_chip_{det.id}",
                    type="primary" if selected else "secondary",
                    width="stretch",
                    help=f"{det.shape} · {det.size_label()}",
                ):
                    st.session_state[SS_SELECTED] = det.id
                    st.session_state.pop(SS_ANN_BYTES, None)
                    st.rerun()
                st.caption(det.size_label())


def _render_results() -> None:
    result = _result()
    if result is None:
        st.caption("Run detection on the Setup tab to see results.")
        return
    result = _apply_review_to_result(result)

    _render_count_preview(result)
    st.caption(EXPERIMENTAL_NOTICE)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Included", result.included_count)
    m2.metric("Excluded", result.excluded_count)
    m3.metric("Partial", result.partial_count)
    m4.metric("Manual adds", result.manually_added_count)
    m5, m6, m7 = st.columns(3)
    m5.metric("Hough / line", result.hough_count)
    m6.metric("Contour / fit", result.contour_count)
    m7.metric("Time (s)", f"{result.processing_time_seconds:.2f}")
    st.caption(
        f"Processed {result.processed_width}×{result.processed_height} · "
        f"Original {result.original_width}×{result.original_height}"
    )

    st.markdown("##### Visual Review")
    view = st.radio(
        "Image view",
        ["All numbered", "Solo selected", "Outlines", "Original"],
        horizontal=True,
        key="shape_view_mode",
        help=(
            "Solo selected shows only the chosen item — no other numbers or markers."
        ),
    )

    included = [d for d in result.detections if d.included]
    if included and not st.session_state.get(SS_SELECTED):
        st.session_state[SS_SELECTED] = included[0].id
    # Keep selection valid
    ids = {d.id for d in result.detections}
    if st.session_state.get(SS_SELECTED) not in ids and included:
        st.session_state[SS_SELECTED] = included[0].id

    if included:
        options = {d.id: f"#{d.sequence_number} · {d.shape} · {d.size_label()}" for d in included}
        current = st.session_state.get(SS_SELECTED)
        keys = list(options.keys())
        idx = keys.index(current) if current in keys else 0
        picked = st.selectbox(
            "Focus item",
            keys,
            index=idx,
            format_func=lambda i: options[i],
            key="shape_focus_select",
        )
        if picked != st.session_state.get(SS_SELECTED):
            st.session_state[SS_SELECTED] = picked
            st.session_state.pop(SS_ANN_BYTES, None)

    try:
        if view == "Original":
            st.image(st.session_state.get(SS_IMAGE), width="stretch")
        elif view == "Solo selected":
            st.caption("Solo mode — one shape only, no numbers.")
            st.image(
                _annotated_bytes(result, "outlines", solo=True),
                width="stretch",
            )
        elif view == "Outlines":
            st.image(_annotated_bytes(result, "outlines"), width="stretch")
        else:
            st.image(_annotated_bytes(result, "numbered"), width="stretch")
    except ShapeDetectionError as exc:
        st.error(str(exc))

    selected = next(
        (d for d in result.detections if d.id == st.session_state.get(SS_SELECTED)),
        None,
    )
    if selected:
        with st.expander("Selected item details", expanded=view == "Solo selected"):
            st.write(
                {
                    "number": selected.sequence_number,
                    "shape": selected.shape,
                    "size": selected.size_label(),
                    "center": (
                        round(selected.center_x, 1),
                        round(selected.center_y, 1),
                    ),
                    "partial": selected.partial,
                    "methods": selected.detection_methods,
                    "shape_quality": selected.quality_score,
                    "included": selected.included,
                }
            )

    st.markdown("##### Export")
    e1, e2, e3 = st.columns(3)
    with e1:
        try:
            png = _annotated_bytes(result, "numbered")
            st.download_button(
                "Annotated PNG",
                data=png,
                file_name="shape_detection_annotated.png",
                mime="image/png",
                key="shape_dl_png",
            )
        except ShapeDetectionError:
            st.caption("Annotated image unavailable.")
    with e2:
        import csv
        from io import StringIO

        buf = StringIO()
        w = csv.writer(buf)
        w.writerow(
            [
                "sequence_number",
                "shape",
                "center_x",
                "center_y",
                "radius",
                "diameter",
                "width",
                "height",
                "partial",
                "methods",
                "shape_quality",
                "included",
                "review_status",
            ]
        )
        for d in result.detections:
            w.writerow(
                [
                    d.sequence_number,
                    d.shape,
                    d.center_x,
                    d.center_y,
                    d.radius,
                    d.diameter,
                    d.width,
                    d.height,
                    d.partial,
                    ";".join(d.detection_methods),
                    d.quality_score,
                    d.included,
                    d.review_status,
                ]
            )
        st.download_button(
            "CSV",
            data=buf.getvalue(),
            file_name="shape_detection.csv",
            mime="text/csv",
            key="shape_dl_csv",
        )
    with e3:
        st.download_button(
            "JSON",
            data=json.dumps(result.public_export_dict(), indent=2),
            file_name="shape_detection.json",
            mime="application/json",
            key="shape_dl_json",
        )


def _render_review() -> None:
    result = _result()
    if result is None:
        st.caption("No detections to review yet.")
        return
    result = _apply_review_to_result(result)

    rows = []
    for d in result.detections:
        rows.append(
            {
                "Number": d.sequence_number,
                "Include": d.included,
                "Shape": d.shape,
                "Diameter": round(d.diameter, 1),
                "Radius": round(d.radius, 1),
                "Partial": d.partial,
                "Detection methods": ", ".join(d.detection_methods),
                "Shape quality": round(d.quality_score, 3),
                "Review status": REVIEW_STATUS_LABELS.get(
                    d.review_status, d.review_status
                ),
                "_id": d.id,
            }
        )
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df.drop(columns=["_id"]) if "_id" in df.columns else df,
        column_config={
            "Include": st.column_config.CheckboxColumn(),
            "Partial": st.column_config.CheckboxColumn(disabled=True),
            "Review status": st.column_config.SelectboxColumn(
                options=list(REVIEW_STATUS_LABELS.values())
            ),
            "Shape quality": st.column_config.NumberColumn(
                help="Geometric shape quality — not model confidence."
            ),
        },
        hide_index=True,
        width="stretch",
        key="shape_review_editor",
    )
    # Sync editor back
    review_map: dict[str, Any] = {}
    for i, d in enumerate(result.detections):
        if i >= len(edited):
            break
        row = edited.iloc[i]
        status_label = str(row.get("Review status") or "Unreviewed")
        status_key = "unreviewed"
        for k, lab in REVIEW_STATUS_LABELS.items():
            if lab == status_label:
                status_key = k
                break
        included = bool(row.get("Include"))
        if status_key in {"false_positive", "duplicate", "ignore"}:
            included = False
        review_map[d.id] = {"included": included, "review_status": status_key}
        d.included = included
        d.review_status = status_key  # type: ignore[assignment]
    st.session_state[SS_REVIEW] = review_map

    ids = [d.id for d in result.detections]
    labels = [f"#{d.sequence_number}" for d in result.detections]
    if ids:
        choice = st.selectbox(
            "Inspect detection",
            options=list(range(len(ids))),
            format_func=lambda i: labels[i],
            key="shape_inspect_idx",
        )
        det = result.detections[int(choice)]
        st.session_state[SS_SELECTED] = det.id
        with st.expander("Detection details", expanded=True):
            st.write(
                {
                    "center": (round(det.center_x, 1), round(det.center_y, 1)),
                    "radius": round(det.radius, 1),
                    "diameter": round(det.diameter, 1),
                    "bounding_box": det.bounding_box.as_dict(),
                    "partial": det.partial,
                    "hough": "hough" in det.detection_methods,
                    "contour": "contour" in det.detection_methods,
                    "shape_quality": det.quality_score,
                    "review_status": det.review_status,
                }
            )

    st.markdown("##### Manual Count Adjustment")
    st.session_state["shape_detection_manual_add"] = int(
        st.number_input(
            "Additional missed circles",
            min_value=0,
            value=int(st.session_state.get("shape_detection_manual_add", 0) or 0),
            step=1,
            key="shape_manual_add",
        )
    )
    st.session_state["shape_detection_manual_notes"] = st.text_input(
        "Reason or notes",
        value=str(st.session_state.get("shape_detection_manual_notes", "") or ""),
        key="shape_manual_notes",
    )
    result.manually_added_count = int(
        st.session_state["shape_detection_manual_add"]
    )
    st.info(
        f"Detected included circles ({result.included_count}) "
        f"+ manually added ({result.manually_added_count}) "
        f"= **final circle count {result.final_count}**"
    )

    user = auth_session.current_user()
    if st.button("Save Shape Test", type="primary", key="shape_save_btn"):
        try:
            ann = None
            try:
                ann = _annotated_bytes(result, "numbered")
            except ShapeDetectionError:
                ann = None
            run_id = save_shape_test(
                result,
                user=user,
                source_type=str(st.session_state.get(SS_SOURCE) or "upload"),
                original_filename=str(st.session_state.get(SS_FILENAME) or ""),
                annotated_bytes=ann,
                notes=str(st.session_state.get("shape_detection_manual_notes") or ""),
            )
            st.success(f"Saved shape test #{run_id}.")
        except (ShapeAuthError, ShapeStorageError) as exc:
            st.error(str(exc))


def _render_history(user) -> None:
    st.markdown("##### Shape Test History")
    st.caption("Opening a saved result does not rerun detection.")

    f1, f2, f3 = st.columns(3)
    with f1:
        mode_filter = st.selectbox(
            "Detection mode",
            ["Any", "strict", "balanced", "sensitive"],
            key="shape_hist_mode",
        )
    with f2:
        min_final = st.number_input(
            "Min final count", min_value=0, value=0, key="shape_hist_min"
        )
    with f3:
        partial_opt = st.selectbox(
            "Partial detections",
            ["Any", "Present", "None"],
            key="shape_hist_partial",
        )

    owner_filter = None
    if user and user.is_admin:
        owner_raw = st.text_input(
            "Filter by user id (admin)",
            value="",
            key="shape_hist_user",
        )
        if owner_raw.strip().isdigit():
            owner_filter = int(owner_raw.strip())

    try:
        rows = list_shape_tests(
            user,
            detection_mode=None if mode_filter == "Any" else mode_filter,
            min_final_count=int(min_final) if min_final else None,
            partial_present=(
                True
                if partial_opt == "Present"
                else False
                if partial_opt == "None"
                else None
            ),
            owner_user_id=owner_filter,
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Could not load shape test history.")
        with st.expander("Technical Details"):
            st.code(type(exc).__name__)
        return

    if not rows:
        st.caption("No saved shape tests yet.")
        return

    for row in rows:
        title = (
            f"#{row['id']} · {row.get('normalized_shape')} · "
            f"final {row.get('final_count')} · {row.get('created_at')}"
        )
        if user and user.is_admin:
            title += f" · {row.get('owner_username_snapshot') or 'unowned'}"
        with st.expander(title, expanded=False):
            st.write(
                {
                    "shape": row.get("requested_shape"),
                    "image": row.get("original_filename_sanitized"),
                    "mode": row.get("detection_mode"),
                    "detected": row.get("detected_count"),
                    "manual": row.get("manually_added_count"),
                    "final": row.get("final_count"),
                    "time_s": row.get("processing_time"),
                }
            )
            if st.button("Open Result", key=f"shape_open_{row['id']}"):
                try:
                    run = get_shape_test(int(row["id"]), user)
                    items = get_shape_test_items(int(row["id"]), user)
                    loaded = result_from_saved_run(run, items)
                    st.session_state[SS_RESULT] = loaded
                    st.session_state[SS_REVIEW] = {
                        d.id: {
                            "included": d.included,
                            "review_status": d.review_status,
                        }
                        for d in loaded.detections
                    }
                    ann = load_annotated_image_bytes(run)
                    if ann:
                        st.session_state[SS_IMAGE] = ann
                        st.session_state[SS_IMAGE_META] = {
                            "width": run.get("original_width"),
                            "height": run.get("original_height"),
                            "hash": run.get("image_hash"),
                            "size_bytes": len(ann),
                        }
                        st.session_state[SS_SOURCE] = "history"
                    else:
                        st.warning(
                            "Annotated image file is missing; showing saved counts only."
                        )
                    st.success("Loaded saved result (detection was not rerun).")
                except ShapeAuthError as exc:
                    st.error(str(exc))

            try:
                items = get_shape_test_items(int(row["id"]), user)
                st.download_button(
                    "Export CSV",
                    data=export_csv(row, items),
                    file_name=f"shape_test_{row['id']}.csv",
                    mime="text/csv",
                    key=f"shape_hist_csv_{row['id']}",
                )
                st.download_button(
                    "Export JSON",
                    data=export_json(row, items),
                    file_name=f"shape_test_{row['id']}.json",
                    mime="application/json",
                    key=f"shape_hist_json_{row['id']}",
                )
            except ShapeAuthError as exc:
                st.error(str(exc))
