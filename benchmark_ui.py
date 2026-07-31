"""Streamlit UI for Detection Benchmark (Settings → AI Configuration).

Isolated from the inventory wizard: never reads/writes analysis_results,
run_context, or wizard uploads.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

import pandas as pd
import streamlit as st

from benchmark import (
    DATASET_GUIDANCE,
    MAX_PROMPT_SETS,
    PROMPT_QUALITY_GUIDANCE,
    RECOMMENDED_BENCHMARK_INVENTORIES,
    THRESHOLD_WARNING,
    BenchmarkRunOutcome,
    build_prompt_comparison_row,
    evaluation_label,
    filter_benchmark_history,
    image_content_hash,
    load_benchmark_results,
    parse_prompt_sets,
    save_benchmark_result,
    update_profile_prompt_terms,
    validate_expected_count,
)
from inventory_profiles import (
    effective_prompts_for_inventory,
    enabled_profiles,
    is_custom_inventory,
    prompts_to_csv,
)
from sample_images import get_sample_by_id, list_enabled_samples, read_sample_bytes


def _wizard_snapshot() -> dict[str, Any]:
    """Capture wizard keys to assert isolation (tests / debug)."""
    keys = (
        "analysis_results",
        "run_context",
        "uploaded_images",
        "stage",
        "inventory_choice",
        "form",
    )
    return {k: st.session_state.get(k) for k in keys}


def _init_benchmark_state() -> None:
    st.session_state.setdefault("benchmark_outcomes", [])
    st.session_state.setdefault("benchmark_active_idx", 0)
    st.session_state.setdefault("benchmark_meta", {})
    st.session_state.setdefault("benchmark_image_bytes", None)
    st.session_state.setdefault("benchmark_image_name", None)
    st.session_state.setdefault("benchmark_image_source", None)
    st.session_state.setdefault("benchmark_image_hash", None)
    st.session_state.setdefault("benchmark_promote_choice", None)
    st.session_state.setdefault("benchmark_mode", "Single Image")
    st.session_state.setdefault("batch_image_bytes", {})
    st.session_state.setdefault("batch_annotated", {})
    st.session_state.setdefault("batch_session", None)
    st.session_state.setdefault("batch_progress", {})
    st.session_state.setdefault("batch_cancel", False)
    st.session_state.setdefault("batch_run_cache", {})
    st.session_state.setdefault("batch_force_rerun", False)


def render_detection_benchmark_section(
    *,
    run_yolo_world: Callable[..., BenchmarkRunOutcome],
    yolo_model_key: str,
    api_ready: bool,
    demo_mode: bool,
) -> None:
    """Compact Detection Benchmark panel (not a wizard)."""
    _init_benchmark_state()
    wizard_before = _wizard_snapshot()

    st.markdown(
        '<div class="aic-panel aic-panel-b"><div class="aic-panel-title">'
        "Detection Benchmark</div>"
        "<p class=\"aic-muted\" style=\"margin:0;\">"
        "Image-specific YOLO-World validation. Separate from the inventory wizard — "
        "does not change active analysis, uploads, or run context."
        "</p></div>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Benchmark mode",
        options=["Single Image", "Batch Benchmark"],
        horizontal=True,
        key="benchmark_mode",
        help="Single Image keeps the original workflow. Batch adds multi-image "
        "threshold sweeps and aggregate comparison.",
    )

    with st.expander("Prompt quality & dataset guidance", expanded=False):
        st.markdown(PROMPT_QUALITY_GUIDANCE)
        st.markdown(THRESHOLD_WARNING)
        st.markdown(DATASET_GUIDANCE)
        st.caption(
            "Recommended objects: "
            + ", ".join(RECOMMENDED_BENCHMARK_INVENTORIES)
            + ". Traffic Cones remain untested until a genuine cone image is supplied."
        )

    if mode == "Batch Benchmark":
        from benchmark_batch_ui import render_batch_benchmark

        render_batch_benchmark(
            run_yolo_world=run_yolo_world,
            yolo_model_key=yolo_model_key,
            api_ready=api_ready,
            demo_mode=demo_mode,
        )
    else:
        _render_single_image_benchmark(
            run_yolo_world=run_yolo_world,
            yolo_model_key=yolo_model_key,
            api_ready=api_ready,
            demo_mode=demo_mode,
        )

    wizard_after = _wizard_snapshot()
    if wizard_before != wizard_after:
        st.warning(
            "Benchmark UI unexpectedly changed wizard session keys. "
            "Please report this as a bug."
        )


def _render_single_image_benchmark(
    *,
    run_yolo_world: Callable[..., BenchmarkRunOutcome],
    yolo_model_key: str,
    api_ready: bool,
    demo_mode: bool,
) -> None:
    """Original single-image benchmark (preserved)."""
    profiles = enabled_profiles()
    profile_keys = [p["key"] for p in profiles]
    labels = {p["key"]: (p.get("display_name") or p["key"]) for p in profiles}

    form_col, img_col = st.columns([1.15, 1], gap="medium")

    with form_col:
        inv = st.selectbox(
            "Inventory",
            options=profile_keys,
            format_func=lambda k: labels.get(k, k),
            key="benchmark_inventory",
        )
        custom_name = ""
        custom_alt = ""
        if is_custom_inventory(inv):
            custom_name = st.text_input("Custom item name", key="benchmark_custom_name")
            custom_alt = st.text_input(
                "Alternate terms",
                key="benchmark_custom_alt",
                help="Comma-separated alternatives for Custom Item.",
            )

        default_prompts, _ = effective_prompts_for_inventory(
            inv,
            custom_item_name=custom_name or None,
            custom_alternatives=custom_alt or None,
        )
        st.caption("Profile default: " + (prompts_to_csv(default_prompts) or "(none)"))

        st.markdown("**Prompt sets** (max 3; each runs independently)")
        n_sets = st.number_input(
            "Number of prompt sets",
            min_value=1,
            max_value=MAX_PROMPT_SETS,
            value=1,
            step=1,
            key="benchmark_n_sets",
        )
        prompt_raw: list[str] = []
        for i in range(int(n_sets)):
            key = f"benchmark_prompt_{i}"
            if key not in st.session_state:
                st.session_state[key] = (
                    prompts_to_csv(default_prompts) if i == 0 else ""
                )
            prompt_raw.append(
                st.text_area(
                    f"Prompt set {chr(65 + i)}",
                    key=key,
                    height=68,
                    help="Comma or newline separated terms for this test only.",
                )
            )

        if "benchmark_expected_count" not in st.session_state:
            st.session_state.benchmark_expected_count = int(
                st.session_state.get("benchmark_expected_prefill", 0) or 0
            )
        expected_raw = st.number_input(
            "Expected object count (ground truth)",
            min_value=0,
            step=1,
            key="benchmark_expected_count",
        )
        object_def = st.text_input(
            "Object definition (optional)",
            key="benchmark_object_def",
            placeholder="Count each individual traffic cone, including partially visible cones.",
        )
        notes = st.text_input("Notes (optional)", key="benchmark_notes")

    with img_col:
        st.markdown("**Test image**")
        samples = list_enabled_samples(inventory_key=None)
        sample_ids = [s.id for s in samples]
        sample_titles = {
            s.id: f"{s.title}"
            + (
                f" (GT={s.benchmark.get('expected_count')})"
                if getattr(s, "benchmark", None) and s.benchmark.get("expected_count") is not None
                else ""
            )
            for s in samples
        }
        sample_choice = st.selectbox(
            "Built-in sample",
            options=["(none)"] + sample_ids,
            format_func=lambda i: sample_titles.get(i, i),
            key="benchmark_sample_id",
        )
        if sample_choice != "(none)":
            sample = get_sample_by_id(sample_choice)
            if sample and getattr(sample, "benchmark", None):
                b = sample.benchmark
                if b.get("verified") and b.get("expected_count") is not None:
                    st.caption(
                        f"Verified sample GT: {b.get('expected_count')} — "
                        f"{b.get('object_definition') or ''}"
                    )
                    st.session_state.benchmark_expected_prefill = int(b["expected_count"])
                if b.get("inventory_key"):
                    st.caption(f"Sample inventory hint: {b.get('inventory_key')}")

        upload = st.file_uploader(
            "Or upload dedicated test image",
            type=["jpg", "jpeg", "png", "webp"],
            key="benchmark_uploader",
            accept_multiple_files=False,
        )
        st.caption("Model for this phase: **YOLO-World** only.")
        bench_conf = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.20,
            step=0.05,
            key="benchmark_confidence",
            help="Detections below this threshold are dropped before review.",
        )

        if demo_mode:
            st.info("Benchmark requires DEMO_MODE=false for live YOLO-World.")
        elif not api_ready:
            st.warning("ROBOFLOW_API_KEY missing — live benchmark disabled.")

        run_clicked = st.button(
            "Run Benchmark",
            type="primary",
            key="benchmark_run_btn",
            width="stretch",
            disabled=demo_mode or not api_ready,
        )

    if run_clicked:
        expected, exp_errs = validate_expected_count(expected_raw)
        sets, set_errs = parse_prompt_sets(prompt_raw)
        image_bytes = None
        image_name = None
        image_source = None
        if upload is not None:
            upload.seek(0)
            image_bytes = upload.read()
            image_name = upload.name or "benchmark_upload.jpg"
            image_source = "upload"
        elif sample_choice != "(none)":
            sample = get_sample_by_id(sample_choice)
            if sample is not None:
                image_bytes = read_sample_bytes(sample)
                image_name = sample.filename
                image_source = f"sample:{sample.id}"
        errs = list(exp_errs) + list(set_errs)
        if image_bytes is None:
            errs.append("Select a sample image or upload a test image.")
        if errs:
            for e in errs:
                st.error(e)
        else:
            assert image_bytes is not None and expected is not None
            st.session_state.benchmark_image_bytes = image_bytes
            st.session_state.benchmark_image_name = image_name
            st.session_state.benchmark_image_source = image_source
            st.session_state.benchmark_image_hash = image_content_hash(image_bytes)
            st.session_state.benchmark_meta = {
                "inventory_key": inv,
                "custom_item_name": custom_name or None,
                "expected_count": expected,
                "object_definition": object_def,
                "notes": notes,
                "model_key": yolo_model_key,
            }
            outcomes: list[BenchmarkRunOutcome] = []
            progress = st.progress(0.0, text="Running prompt sets…")
            for i, prompts in enumerate(sets):
                label = f"Set {chr(65 + i)}"
                progress.progress((i) / max(1, len(sets)), text=f"Running {label}…")
                try:
                    outcome = run_yolo_world(
                        image_bytes=image_bytes,
                        image_name=image_name or "benchmark.jpg",
                        prompts=prompts,
                        prompt_set_label=label,
                        confidence_threshold=float(bench_conf),
                    )
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    outcome = BenchmarkRunOutcome(
                        prompt_set_label=label,
                        prompt_set=list(prompts),
                        success=False,
                        execution_failed=True,
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                outcome.apply_review(expected_count=expected)
                outcomes.append(outcome)
            progress.progress(1.0, text="Done")
            st.session_state.benchmark_outcomes = outcomes
            st.session_state.benchmark_active_idx = 0
            st.session_state.benchmark_promote_choice = None
            st.rerun()

    outcomes: list[BenchmarkRunOutcome] = list(st.session_state.get("benchmark_outcomes") or [])
    meta = dict(st.session_state.get("benchmark_meta") or {})

    tab_res, tab_viz, tab_cmp, tab_hist = st.tabs(
        ["Results", "Visual Review", "Prompt Comparison", "History"]
    )

    with tab_res:
        if not outcomes:
            st.caption("Run a benchmark to see results.")
        else:
            idx = int(st.session_state.get("benchmark_active_idx") or 0)
            idx = max(0, min(idx, len(outcomes) - 1))
            labels_opts = [o.prompt_set_label for o in outcomes]
            idx = labels_opts.index(
                st.selectbox(
                    "Active prompt set",
                    options=labels_opts,
                    index=idx,
                    key="benchmark_active_select",
                )
            )
            st.session_state.benchmark_active_idx = idx
            o = outcomes[idx]
            o.apply_review(expected_count=meta.get("expected_count"))
            _render_result_summary(o, meta)
            if o.annotated_image_bytes:
                st.image(
                    o.annotated_image_bytes,
                    caption=f"Annotated — {o.prompt_set_label}",
                    width="stretch",
                )
            with st.expander("Technical details", expanded=False):
                st.json(o.technical or {})

    with tab_viz:
        if not outcomes:
            st.caption("No detections to review yet.")
        else:
            idx = int(st.session_state.get("benchmark_active_idx") or 0)
            idx = max(0, min(idx, len(outcomes) - 1))
            o = outcomes[idx]
            _render_visual_review(o, meta, outcomes_key="benchmark_outcomes", idx=idx)

    with tab_cmp:
        _render_prompt_comparison(outcomes, meta)

    with tab_hist:
        _render_benchmark_history()


def _render_result_summary(o: BenchmarkRunOutcome, meta: dict[str, Any]) -> None:
    expected = meta.get("expected_count")
    diff = (
        (o.final_count - int(expected))
        if expected is not None
        else None
    )
    st.markdown(
        f"""
        <div class="aic-chip-grid aic-chip-grid-4">
          <div class="aic-chip aic-chip-b"><span class="aic-chip-label">Evaluation</span>
            <span class="aic-chip-value">{evaluation_label(o.metrics.evaluation)}</span></div>
          <div class="aic-chip aic-chip-g"><span class="aic-chip-label">AI count</span>
            <span class="aic-chip-value">{o.final_count}</span></div>
          <div class="aic-chip aic-chip-r"><span class="aic-chip-label">Expected</span>
            <span class="aic-chip-value">{expected if expected is not None else "—"}</span></div>
          <div class="aic-chip aic-chip-b"><span class="aic-chip-label">Difference</span>
            <span class="aic-chip-value">{diff if diff is not None else "—"}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Metrics below are for **this image and prompt set only** — "
        "not universal model accuracy."
    )
    st.markdown(
        f"""
        <div class="aic-kv-grid">
          <div class="aic-kv"><b>Inventory</b><br/>{meta.get("inventory_key") or "—"}</div>
          <div class="aic-kv"><b>Prompts</b><br/>{prompts_to_csv(o.prompt_set) or "—"}</div>
          <div class="aic-kv"><b>Invocation mode</b><br/>{o.invocation_mode or "—"}</div>
          <div class="aic-kv"><b>Matched step</b><br/>{o.matched_step_id or "—"}</div>
          <div class="aic-kv"><b>Injected field</b><br/>{o.field_injected or "—"}</div>
          <div class="aic-kv"><b>Fallback used</b><br/>{"yes" if o.fallback_used else "no"}</div>
          <div class="aic-kv"><b>Raw / normalized / final</b><br/>{o.raw_count} / {o.normalized_count} / {o.final_count}</div>
          <div class="aic-kv"><b>Returned classes</b><br/>{", ".join(o.returned_classes) or "(none)"}</div>
          <div class="aic-kv"><b>Warnings</b><br/>{o.warning_count}</div>
          <div class="aic-kv"><b>Processing time</b><br/>{o.processing_time:.2f}s</div>
          <div class="aic-kv"><b>Precision / Recall</b><br/>
            {(f"{o.metrics.precision:.2f}" if o.metrics.precision is not None else "—")} /
            {(f"{o.metrics.recall:.2f}" if o.metrics.recall is not None else "—")}
          </div>
          <div class="aic-kv"><b>Count accuracy</b><br/>
            {(f"{o.metrics.count_accuracy:.2f}" if o.metrics.count_accuracy is not None else "—")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if o.error_message:
        st.error(o.error_message)
    if o.fallback_used:
        st.error(
            "Unmodified Workflow fallback was used — this is not a valid dynamic benchmark."
        )


def _render_visual_review(
    o: BenchmarkRunOutcome,
    meta: dict[str, Any],
    *,
    outcomes_key: str,
    idx: int,
) -> None:
    st.caption(f"Reviewing {o.prompt_set_label} — number detections match the annotation.")
    if o.annotated_image_bytes:
        st.image(o.annotated_image_bytes, width="stretch")
    if not o.detections:
        st.info("No detections to label. Enter missed objects if ground truth > 0.")
    labels: list[str] = list(o.detection_labels)
    while len(labels) < len(o.detections):
        labels.append("correct")
    label_options = [
        "correct",
        "false_positive",
        "wrong_class",
        "duplicate",
        "ignore",
    ]
    for i, det in enumerate(o.detections):
        cls = det.get("class_name") or det.get("class") or "?"
        conf = det.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else "?"
        labels[i] = st.selectbox(
            f"#{i + 1} {cls} ({conf_s})",
            options=label_options,
            index=label_options.index(labels[i]) if labels[i] in label_options else 0,
            key=f"benchmark_det_label_{idx}_{i}",
        )
    missed = st.number_input(
        "Missed object count (false negatives)",
        min_value=0,
        value=int(o.missed_count or 0),
        step=1,
        key=f"benchmark_missed_{idx}",
    )
    if st.button("Apply review metrics", key=f"benchmark_apply_review_{idx}"):
        o.detection_labels = labels[: len(o.detections)]
        o.missed_count = int(missed)
        o.apply_review(expected_count=meta.get("expected_count"))
        outcomes = list(st.session_state.get(outcomes_key) or [])
        if 0 <= idx < len(outcomes):
            outcomes[idx] = o
            st.session_state[outcomes_key] = outcomes
        st.success(
            f"TP={o.metrics.true_positives} FP={o.metrics.false_positives} "
            f"FN={o.metrics.false_negatives} | "
            f"P={o.metrics.precision} R={o.metrics.recall}"
        )
        st.rerun()

    if st.button("Save this prompt-set result", key=f"benchmark_save_{idx}"):
        o.apply_review(
            expected_count=meta.get("expected_count"),
            labels=labels[: len(o.detections)],
            missed_count=int(missed),
        )
        record = o.to_storage_dict(
            inventory_key=str(meta.get("inventory_key") or ""),
            custom_item_name=meta.get("custom_item_name"),
            expected_count=meta.get("expected_count"),
            image_hash=str(st.session_state.get("benchmark_image_hash") or ""),
            image_source=str(st.session_state.get("benchmark_image_source") or ""),
            image_name=str(st.session_state.get("benchmark_image_name") or ""),
            model_key=str(meta.get("model_key") or "YOLO-World"),
            notes=str(meta.get("notes") or ""),
            object_definition=str(meta.get("object_definition") or ""),
        )
        saved = save_benchmark_result(record)
        st.success(f"Saved benchmark {saved.get('benchmark_id')}")


def _render_prompt_comparison(
    outcomes: list[BenchmarkRunOutcome],
    meta: dict[str, Any],
) -> None:
    if not outcomes:
        st.caption("Run multiple prompt sets to compare.")
        return
    expected = meta.get("expected_count")
    rows = []
    for o in outcomes:
        o.apply_review(expected_count=expected)
        rows.append(
            build_prompt_comparison_row(
                prompt_set_label=o.prompt_set_label,
                prompt_set=o.prompt_set,
                ai_count=o.final_count,
                expected_count=expected,
                metrics=o.metrics,
                processing_time=o.processing_time,
                returned_classes=o.returned_classes,
                status=evaluation_label(o.metrics.evaluation),
            )
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=220)
    st.caption(f"1 image × {len(outcomes)} prompt set(s) = {len(outcomes)} inference run(s).")

    choice = st.selectbox(
        "Select prompt set to promote",
        options=["(none)"] + [o.prompt_set_label for o in outcomes],
        key="benchmark_promote_select",
    )
    inv = meta.get("inventory_key")
    can_promote = (
        choice != "(none)"
        and inv
        and not is_custom_inventory(str(inv))
    )
    if st.button(
        "Use These Prompts for Inventory Profile",
        key="benchmark_promote_btn",
        disabled=not can_promote,
    ):
        selected = next(o for o in outcomes if o.prompt_set_label == choice)
        ok, msg = update_profile_prompt_terms(str(inv), list(selected.prompt_set))
        if ok:
            st.success(msg)
            st.session_state.benchmark_promote_choice = choice
        else:
            st.error(msg)
    if is_custom_inventory(str(inv or "")):
        st.caption("Custom Item prompts are entered at setup and are not stored as a preset.")


def _render_benchmark_history() -> None:
    st.caption(
        "Stored in `data/benchmarks.json`. On Streamlit Community Cloud this file is "
        "ephemeral unless you add external storage."
    )
    results = load_benchmark_results()
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        inventories = sorted({str(r.get("inventory_key") or "") for r in results if r.get("inventory_key")})
        inv_f = st.selectbox(
            "Inventory",
            options=["(all)"] + inventories,
            key="benchmark_hist_inv",
        )
    with f2:
        exact = st.checkbox("Exact match", key="benchmark_hist_exact")
    with f3:
        over = st.checkbox("Overcount", key="benchmark_hist_over")
    with f4:
        under = st.checkbox("Undercount", key="benchmark_hist_under")
    with f5:
        failed = st.checkbox("Failed", key="benchmark_hist_fail")

    filtered = filter_benchmark_history(
        results,
        inventory_key=inv_f,
        exact_match=exact,
        overcount=over,
        undercount=under,
        failed=failed,
    )
    if not filtered:
        st.caption("No benchmark results yet.")
        return
    table = []
    for r in filtered[:50]:
        table.append(
            {
                "Date": str(r.get("timestamp") or "")[:19],
                "Inventory": r.get("inventory_key"),
                "Image": r.get("image_name") or (str(r.get("image_hash") or "")[:10]),
                "Prompt set": r.get("prompt_set_label")
                or prompts_to_csv(list(r.get("prompt_set") or [])),
                "Expected": r.get("expected_count"),
                "AI count": r.get("final_count"),
                "Precision": r.get("precision"),
                "Recall": r.get("recall"),
                "Count diff": (
                    (int(r["final_count"]) - int(r["expected_count"]))
                    if r.get("final_count") is not None
                    and r.get("expected_count") is not None
                    else None
                ),
                "ID": r.get("benchmark_id"),
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", height=260)
    open_id = st.selectbox(
        "Open result",
        options=["(none)"] + [str(r.get("benchmark_id")) for r in filtered[:50]],
        key="benchmark_hist_open",
    )
    if open_id != "(none)":
        match = next((r for r in filtered if str(r.get("benchmark_id")) == open_id), None)
        if match:
            with st.expander("Result detail", expanded=True):
                st.json(match)
