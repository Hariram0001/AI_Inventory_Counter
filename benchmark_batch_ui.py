"""Batch Detection Benchmark UI (multi-image + threshold sweep).

Imported by benchmark_ui; keeps Single Image mode untouched.
"""

from __future__ import annotations

import traceback
import uuid
from typing import Any, Callable

import pandas as pd
import streamlit as st

from benchmark import (
    DEFAULT_SWEEP_THRESHOLDS,
    INFERENCE_CONFIRM_THRESHOLD,
    MAX_BATCH_IMAGES,
    MAX_PROMPT_SETS,
    MAX_THRESHOLDS,
    RANKING_OBJECTIVES,
    BatchImageSpec,
    BenchmarkRunCache,
    BenchmarkRunOutcome,
    NamedPromptSet,
    aggregate_prompt_threshold,
    apply_visual_review_to_run,
    build_batch_session,
    build_cache_key,
    build_comparison_matrix,
    calculate_inference_run_count,
    compute_benchmark_metrics,
    dedupe_batch_images,
    enumerate_batch_combinations,
    export_session_csv,
    export_session_json,
    format_upload_size,
    load_batch_sessions,
    normalize_thresholds,
    outcome_to_run_dict,
    owned_by_user,
    parse_named_prompt_sets,
    recommend_configuration,
    save_batch_session,
    stamp_owner,
    update_profile_prompt_terms,
    validate_batch_ground_truth,
)
from inventory_profiles import (
    effective_prompts_for_inventory,
    enabled_profiles,
    is_custom_inventory,
    prompts_to_csv,
)
from sample_images import get_sample_by_id, list_enabled_samples, read_sample_bytes


def _owned(session: dict[str, Any]) -> dict[str, Any]:
    """Attribute a batch session to whoever is signed in."""
    import auth_session

    user = auth_session.current_user()
    return stamp_owner(
        session,
        user_id=None if user is None else user.user_id,
        username="" if user is None else user.username,
    )


def _my_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch sessions for the signed-in user; administrators see all."""
    import auth_session

    user = auth_session.current_user()
    return owned_by_user(
        sessions,
        user_id=None if user is None else user.user_id,
        is_admin=bool(user is not None and user.is_admin),
    )


def render_batch_benchmark(
    *,
    run_yolo_world: Callable[..., BenchmarkRunOutcome],
    yolo_model_key: str,
    api_ready: bool,
    demo_mode: bool,
) -> None:
    """Multi-image batch benchmark with threshold sweep."""
    profiles = enabled_profiles()
    profile_keys = [p["key"] for p in profiles]
    labels = {p["key"]: (p.get("display_name") or p["key"]) for p in profiles}
    samples = list_enabled_samples(inventory_key=None)

    tab_setup, tab_prog, tab_cmp, tab_viz, tab_hist = st.tabs(
        ["Setup", "Progress", "Comparison", "Visual Review", "History"]
    )

    with tab_setup:
        _render_batch_setup(
            run_yolo_world=run_yolo_world,
            yolo_model_key=yolo_model_key,
            api_ready=api_ready,
            demo_mode=demo_mode,
            profile_keys=profile_keys,
            labels=labels,
            samples=samples,
        )

    session = st.session_state.get("batch_session")
    progress = st.session_state.get("batch_progress") or {}

    with tab_prog:
        if progress:
            st.write(
                f"Completed {progress.get('completed', 0)} / {progress.get('total', 0)} · "
                f"Failures {progress.get('failures', 0)}"
            )
            st.caption(
                f"Last: image={progress.get('image')} · "
                f"prompt={progress.get('prompt_set')} · "
                f"thr={progress.get('threshold')}"
            )
            st.progress(
                float(progress.get("completed", 0))
                / max(1, float(progress.get("total", 1)))
            )
        else:
            st.caption("No batch run in progress.")
        if st.button("Cancel Remaining Runs", key="batch_cancel_btn"):
            st.session_state.batch_cancel = True
            st.warning("Cancel requested — stops before the next combination.")

    with tab_cmp:
        _render_batch_comparison(session)

    with tab_viz:
        _render_batch_visual_review(session)

    with tab_hist:
        _render_batch_history()


def _render_batch_setup(
    *,
    run_yolo_world: Callable[..., BenchmarkRunOutcome],
    yolo_model_key: str,
    api_ready: bool,
    demo_mode: bool,
    profile_keys: list[str],
    labels: dict[str, str],
    samples: list[Any],
) -> None:
    inv = st.selectbox(
        "Inventory",
        options=profile_keys,
        format_func=lambda k: labels.get(k, k),
        key="batch_inventory",
    )
    custom_name = ""
    custom_alt = ""
    if is_custom_inventory(inv):
        custom_name = st.text_area(
            "Items to detect",
            key="batch_custom_name",
            height=90,
            placeholder="traffic cone\nbarrel\npallet",
            help="One or more items — one per line, or comma-separated.",
        )
        custom_alt = st.text_input(
            "Extra synonyms (optional)",
            key="batch_custom_alt",
        )

    default_prompts, _ = effective_prompts_for_inventory(
        inv,
        custom_item_name=custom_name or None,
        custom_alternatives=custom_alt or None,
    )

    st.markdown(f"**Images** (max {MAX_BATCH_IMAGES})")
    sample_ids = st.multiselect(
        "Built-in samples",
        options=[s.id for s in samples],
        format_func=lambda i: next(
            (
                f"{s.title}"
                + (
                    f" (GT={s.benchmark.get('expected_count')})"
                    if s.benchmark and s.benchmark.get("expected_count") is not None
                    else ""
                )
                for s in samples
                if s.id == i
            ),
            i,
        ),
        key="batch_sample_ids",
    )
    uploads = st.file_uploader(
        "Upload test images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    raw_items: list[dict[str, Any]] = []
    for sid in sample_ids:
        sample = get_sample_by_id(sid)
        if sample is None:
            continue
        data = read_sample_bytes(sample)
        bmeta = sample.benchmark or {}
        raw_items.append(
            {
                "image_id": sample.id,
                "image_name": sample.filename,
                "image_source": f"sample:{sample.id}",
                "image_bytes": data,
                "expected_count": bmeta.get("expected_count"),
                "object_definition": bmeta.get("object_definition") or "",
                "include_in_aggregate": True,
            }
        )
    if uploads:
        for up in uploads:
            up.seek(0)
            raw_items.append(
                {
                    "image_id": up.name,
                    "image_name": up.name,
                    "image_source": "upload",
                    "image_bytes": up.read(),
                    "include_in_aggregate": True,
                }
            )

    specs, warn, bytes_map = dedupe_batch_images(raw_items, max_images=MAX_BATCH_IMAGES)
    for w in warn:
        st.caption(w)
    total_bytes = sum(s.size_bytes for s in specs)
    st.caption(
        f"{len(specs)} image(s) · {format_upload_size(total_bytes)} total"
        + (f" · {len(warn)} notice(s)" if warn else "")
    )

    if specs:
        gt_rows = []
        for s in specs:
            prev = (st.session_state.get("batch_gt_edits") or {}).get(s.image_hash, {})
            gt_rows.append(
                {
                    "image_hash": s.image_hash,
                    "Image": s.image_name,
                    "Expected count": int(
                        prev.get(
                            "expected_count",
                            s.expected_count if s.expected_count is not None else 0,
                        )
                    ),
                    "Object definition": prev.get(
                        "object_definition", s.object_definition
                    ),
                    "Notes": prev.get("notes", s.notes),
                    "Include": bool(prev.get("include", s.include_in_aggregate)),
                }
            )
        edited = st.data_editor(
            pd.DataFrame(gt_rows),
            hide_index=True,
            width="stretch",
            height=min(280, 80 + 36 * max(1, len(gt_rows))),
            column_config={
                "image_hash": None,
                "Expected count": st.column_config.NumberColumn(min_value=0, step=1),
                "Include": st.column_config.CheckboxColumn(),
            },
            key="batch_gt_editor",
        )
        edits: dict[str, Any] = {}
        for _, row in edited.iterrows():
            h = str(row["image_hash"])
            edits[h] = {
                "expected_count": int(row["Expected count"]),
                "object_definition": str(row.get("Object definition") or ""),
                "notes": str(row.get("Notes") or ""),
                "include": bool(row.get("Include")),
            }
            for s in specs:
                if s.image_hash == h:
                    s.expected_count = int(row["Expected count"])
                    s.object_definition = str(row.get("Object definition") or "")
                    s.notes = str(row.get("Notes") or "")
                    s.include_in_aggregate = bool(row.get("Include"))
        st.session_state.batch_gt_edits = edits
        st.session_state.batch_image_bytes = bytes_map
        st.session_state.batch_specs = [s.to_dict() for s in specs]

    st.markdown("**Prompt sets** (max 3)")
    n_sets = st.number_input(
        "Number of prompt sets",
        min_value=1,
        max_value=MAX_PROMPT_SETS,
        value=min(2, MAX_PROMPT_SETS),
        key="batch_n_sets",
    )
    named_raw: list[dict[str, Any]] = []
    for i in range(int(n_sets)):
        c1, c2 = st.columns([1, 3])
        with c1:
            name = st.text_input(
                f"Name {chr(65 + i)}",
                value=["Basic", "Specific", "Counting unit"][i]
                if i < 3
                else f"Set {chr(65 + i)}",
                key=f"batch_ps_name_{i}",
            )
            enabled = st.checkbox("Enabled", value=True, key=f"batch_ps_en_{i}")
        with c2:
            default = prompts_to_csv(default_prompts) if i == 0 else ""
            key = f"batch_ps_terms_{i}"
            if key not in st.session_state:
                st.session_state[key] = default
            terms = st.text_area(f"Prompts {chr(65 + i)}", key=key, height=68)
        named_raw.append(
            {"name": name or f"Set {chr(65 + i)}", "prompts": terms, "enabled": enabled}
        )

    st.markdown("**Confidence thresholds**")
    thr_mode = st.radio(
        "Threshold mode",
        options=["Fixed threshold", "Threshold sweep"],
        horizontal=True,
        key="batch_thr_mode",
    )
    if thr_mode == "Fixed threshold":
        fixed = st.slider(
            "Confidence", 0.01, 0.95, 0.20, 0.01, key="batch_fixed_thr"
        )
        thr_values = [float(fixed)]
    else:
        default_csv = ", ".join(str(x) for x in DEFAULT_SWEEP_THRESHOLDS)
        thr_text = st.text_input(
            "Thresholds (comma-separated)",
            value=default_csv,
            key="batch_thr_csv",
            help=f"Min 0.01, max 0.95, at most {MAX_THRESHOLDS} values.",
        )
        parts = [p.strip() for p in thr_text.replace(";", ",").split(",") if p.strip()]
        thr_values, thr_errs = normalize_thresholds(parts)
        for e in thr_errs:
            if "At least" not in e:
                st.caption(e)
        if not thr_values:
            thr_values = list(DEFAULT_SWEEP_THRESHOLDS)

    prompt_sets, ps_errs = parse_named_prompt_sets(named_raw)
    enabled_sets = [p for p in prompt_sets if p.enabled]
    included = [s for s in specs if s.include_in_aggregate]
    planned = calculate_inference_run_count(
        image_count=len(included),
        prompt_set_count=len(enabled_sets),
        threshold_count=len(thr_values),
    )
    st.info(
        f"**Planned inference runs:** {len(included)} images × "
        f"{len(enabled_sets)} prompt sets × {len(thr_values)} thresholds "
        f"= **{planned}**"
    )
    force = st.checkbox("Force rerun (ignore cache)", key="batch_force_rerun_cb")
    need_confirm = planned > INFERENCE_CONFIRM_THRESHOLD
    confirmed = True
    if need_confirm:
        confirmed = st.checkbox(
            f"I confirm running {planned} inference calls (>{INFERENCE_CONFIRM_THRESHOLD})",
            key="batch_confirm_runs",
        )

    if demo_mode:
        st.info("Batch benchmark requires DEMO_MODE=false.")
    elif not api_ready:
        st.warning("ROBOFLOW_API_KEY missing.")

    run_batch = st.button(
        "Run Batch Benchmark",
        type="primary",
        key="batch_run_btn",
        disabled=demo_mode or not api_ready or (need_confirm and not confirmed),
    )

    if run_batch:
        gt_errs = validate_batch_ground_truth(specs)
        errs = list(ps_errs) + list(gt_errs)
        if not specs:
            errs.append("Add at least one image.")
        if errs:
            for e in errs:
                st.error(e)
            return
        st.session_state.batch_cancel = False
        execute_batch_runs(
            run_yolo_world=run_yolo_world,
            yolo_model_key=yolo_model_key,
            inventory_key=inv,
            custom_item_name=custom_name or None,
            specs=specs,
            bytes_map=bytes_map,
            prompt_sets=prompt_sets,
            thresholds=thr_values,
            force_rerun=bool(force),
        )
        st.rerun()


def _render_batch_comparison(session: dict[str, Any] | None) -> None:
    if not session:
        st.caption("Run a batch to see comparison.")
        return
    aggregates = list(session.get("aggregates") or [])
    matrix = build_comparison_matrix(aggregates)
    st.dataframe(pd.DataFrame(matrix), hide_index=True, width="stretch", height=260)
    st.caption(
        "Cell metrics are for this benchmark dataset only — not universal accuracy."
    )
    ps_opts = sorted({a.get("prompt_set_label") for a in aggregates})
    thr_opts = sorted({float(a.get("confidence_threshold")) for a in aggregates})
    if not ps_opts or not thr_opts:
        return
    d1, d2 = st.columns(2)
    with d1:
        sel_ps = st.selectbox("Prompt set", options=ps_opts, key="batch_drill_ps")
    with d2:
        sel_thr = st.selectbox("Threshold", options=thr_opts, key="batch_drill_thr")
    detail = [
        r
        for r in (session.get("runs") or [])
        if r.get("prompt_set_label") == sel_ps
        and abs(float(r.get("confidence_threshold") or 0) - float(sel_thr)) < 1e-9
    ]
    if detail:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Image": r.get("image_name"),
                        "Expected": r.get("expected_count"),
                        "AI": r.get("final_count"),
                        "Error": r.get("count_error"),
                        "Exact": r.get("exact_match"),
                        "Reviewed": r.get("reviewed"),
                        "P": r.get("precision") if r.get("reviewed") else "not_reviewed",
                        "R": r.get("recall") if r.get("reviewed") else "not_reviewed",
                        "Cached": r.get("cached"),
                        "Failed": r.get("execution_failed"),
                        "Classes": ", ".join(r.get("returned_classes") or []),
                    }
                    for r in detail
                ]
            ),
            hide_index=True,
            width="stretch",
            height=240,
        )

    objective = st.selectbox(
        "Ranking objective",
        options=list(RANKING_OBJECTIVES),
        format_func=lambda o: {
            "lowest_mae": "Lowest mean absolute count error",
            "highest_exact_match": "Highest exact-match rate",
            "highest_recall": "Highest recall",
            "highest_precision": "Highest precision",
            "balanced_f1": "Balanced precision and recall",
        }.get(o, o),
        key="batch_rank_obj",
    )
    rec = recommend_configuration(aggregates, objective=objective)
    if rec is None:
        st.warning("Recommendation disabled — too few valid runs for this dataset.")
    else:
        st.success(
            f"**{rec['label']}** — {rec.get('prompt_set_label')} @ "
            f"{rec.get('confidence_threshold')} · "
            f"images={rec.get('supporting_image_count')} · "
            f"exact-match={rec.get('exact_match_rate')} · "
            f"MAE={rec.get('mean_absolute_count_error')} · "
            f"failures={rec.get('failed_runs')}"
        )
        st.caption(
            "Precision/recall available: "
            + ("yes" if rec.get("precision_recall_available") else "no (review needed)")
        )
        can_apply = (
            not is_custom_inventory(str(session.get("inventory_key") or ""))
            and rec.get("prompt_set_label")
        )
        if st.button(
            "Apply Prompt Set and Threshold to Inventory Profile",
            key="batch_apply_profile",
            disabled=not can_apply,
        ):
            prompts: list[str] = []
            for p in session.get("prompt_sets") or []:
                if p.get("name") == rec.get("prompt_set_label"):
                    prompts = list(p.get("prompts") or [])
                    break
            ok, msg = update_profile_prompt_terms(
                str(session.get("inventory_key")),
                prompts,
                default_confidence=float(rec.get("confidence_threshold")),
                justification_benchmark_id=str(session.get("session_id")),
            )
            if ok:
                st.success(msg)
                session["profile_update"] = {
                    "prompt_set": rec.get("prompt_set_label"),
                    "confidence_threshold": rec.get("confidence_threshold"),
                    "benchmark_session_id": session.get("session_id"),
                }
                st.session_state.batch_session = session
            else:
                st.error(msg)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Export JSON",
            data=export_session_json(session),
            file_name=f"benchmark_session_{str(session.get('session_id', 'x'))[:8]}.json",
            mime="application/json",
            key="batch_export_json",
        )
    with c2:
        st.download_button(
            "Export CSV summary",
            data=export_session_csv(session),
            file_name=f"benchmark_session_{str(session.get('session_id', 'x'))[:8]}.csv",
            mime="text/csv",
            key="batch_export_csv",
        )


def _render_batch_visual_review(session: dict[str, Any] | None) -> None:
    if not session:
        st.caption("No runs to review.")
        return
    runs = list(session.get("runs") or [])
    labels_run = [
        f"{r.get('image_name')} | {r.get('prompt_set_label')} @ "
        f"{r.get('confidence_threshold')}"
        + (" [cached]" if r.get("cached") else "")
        for r in runs
    ]
    if not labels_run:
        st.caption("Empty run list.")
        return
    idx = st.selectbox(
        "Open run",
        options=list(range(len(labels_run))),
        format_func=lambda i: labels_run[i],
        key="batch_viz_idx",
    )
    run = dict(runs[idx])
    ann = (st.session_state.get("batch_annotated") or {}).get(run.get("run_id"))
    src_bytes = (st.session_state.get("batch_image_bytes") or {}).get(
        run.get("image_hash")
    )
    cols = st.columns(2)
    with cols[0]:
        if src_bytes:
            st.image(src_bytes, caption="Source", width="stretch")
    with cols[1]:
        if ann:
            st.image(ann, caption="Annotated", width="stretch")
        else:
            st.caption("No annotated preview in session (re-run to capture).")
    st.write(
        {
            "expected": run.get("expected_count"),
            "ai_count": run.get("final_count"),
            "classes": run.get("returned_classes"),
            "threshold": run.get("confidence_threshold"),
            "prompt_set": run.get("prompt_set_label"),
            "reviewed": run.get("reviewed"),
            "fallback_used": run.get("fallback_used"),
            "invocation_mode": run.get("invocation_mode"),
        }
    )
    dets = list(run.get("detections") or [])
    det_labels = list(run.get("detection_labels") or [])
    while len(det_labels) < len(dets):
        det_labels.append("correct")
    label_options = [
        "correct",
        "false_positive",
        "wrong_class",
        "duplicate",
        "ignore",
    ]
    for i, det in enumerate(dets):
        cls = det.get("class_name") or det.get("class") or "?"
        conf = det.get("confidence")
        conf_s = f"{float(conf):.0%}" if conf is not None else "?"
        det_labels[i] = st.selectbox(
            f"#{i + 1} {cls} ({conf_s})",
            options=label_options,
            index=label_options.index(det_labels[i])
            if det_labels[i] in label_options
            else 0,
            key=f"batch_det_{run.get('run_id')}_{i}",
        )
    missed = st.number_input(
        "Missed object count",
        min_value=0,
        value=int(run.get("missed_count") or 0),
        key=f"batch_missed_{run.get('run_id')}",
    )
    if st.button("Save visual review for this run", key="batch_save_review"):
        updated = apply_visual_review_to_run(
            run,
            labels=det_labels[: len(dets)],
            missed_count=int(missed),
        )
        runs[idx] = updated
        session["runs"] = runs
        aggregates = []
        for p in session.get("prompt_sets") or []:
            if not p.get("enabled", True):
                continue
            for thr in session.get("confidence_thresholds") or []:
                aggregates.append(
                    aggregate_prompt_threshold(
                        runs,
                        prompt_set_label=str(p.get("name")),
                        confidence_threshold=float(thr),
                    )
                )
        session["aggregates"] = aggregates
        st.session_state.batch_session = session
        st.success(
            f"Reviewed — TP={updated.get('true_positives')} "
            f"FP={updated.get('false_positives')} "
            f"FN={updated.get('false_negatives')}"
        )
        st.rerun()
    with st.expander("Technical details", expanded=False):
        st.json(run.get("technical") or {})


def _render_batch_history() -> None:
    import auth_session

    viewer = auth_session.current_user()
    show_owner = bool(viewer is not None and viewer.is_admin)
    st.caption(
        "Batch sessions: `data/benchmark_sessions.json` (ephemeral on Streamlit Cloud). "
        "Per-run rows also append to `data/benchmarks.json` for compatibility. "
        + ("You see every user's sessions." if show_owner else "Only your own sessions are listed.")
    )
    sessions = _my_sessions(load_batch_sessions())
    if not sessions:
        st.caption("You have not saved any batch sessions yet.")
        return
    rows = []
    for s in reversed(sessions[-30:]):
        rows.append(
            {
                "Date": str(s.get("timestamp") or "")[:19],
                **({"User": s.get("username") or "—"} if show_owner else {}),
                "Inventory": s.get("inventory_key"),
                "Images": len(s.get("images") or []),
                "Runs": len(s.get("runs") or []),
                "Session": str(s.get("session_id") or "")[:8],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=220)
    open_sid = st.selectbox(
        "Open session",
        options=["(none)"] + [str(s.get("session_id")) for s in reversed(sessions[-30:])],
        key="batch_hist_open",
    )
    if open_sid != "(none)":
        match = next(
            (s for s in sessions if str(s.get("session_id")) == open_sid),
            None,
        )
        if match:
            st.session_state.batch_session = match
            st.json(
                {
                    "session_id": match.get("session_id"),
                    "inventory_key": match.get("inventory_key"),
                    "thresholds": match.get("confidence_thresholds"),
                    "prompt_sets": match.get("prompt_sets"),
                    "aggregates": match.get("aggregates"),
                }
            )


def execute_batch_runs(
    *,
    run_yolo_world: Callable[..., BenchmarkRunOutcome],
    yolo_model_key: str,
    inventory_key: str,
    custom_item_name: str | None,
    specs: list[BatchImageSpec],
    bytes_map: dict[str, bytes],
    prompt_sets: list[NamedPromptSet],
    thresholds: list[float],
    force_rerun: bool,
) -> None:
    """Execute image × prompt × threshold combinations with cache + isolation."""
    combos = enumerate_batch_combinations(specs, prompt_sets, thresholds)
    total = len(combos)
    cache = BenchmarkRunCache(st.session_state.get("batch_run_cache") or {})
    workflow_id = ""
    if ":" in (yolo_model_key or ""):
        workflow_id = (yolo_model_key or "").split(":", 1)[-1]

    session_id = str(uuid.uuid4())
    runs: list[dict[str, Any]] = []
    annotated: dict[str, bytes] = dict(st.session_state.get("batch_annotated") or {})
    failures = 0
    progress_box = st.empty()
    bar = st.progress(0.0)

    for i, (im, ps, thr) in enumerate(combos):
        if st.session_state.get("batch_cancel"):
            progress_box.warning(f"Cancelled after {i} / {total} runs.")
            break
        st.session_state.batch_progress = {
            "completed": i,
            "total": total,
            "image": im.image_name,
            "prompt_set": ps.name,
            "threshold": thr,
            "failures": failures,
        }
        progress_box.caption(
            f"Running {i + 1}/{total}: {im.image_name} · {ps.name} @ {thr}"
        )
        bar.progress(i / max(1, total))

        owner = None
        try:
            import auth_session as _auth_session

            current = _auth_session.current_user()
            owner = int(current.user_id) if current is not None else None
        except Exception:  # noqa: BLE001
            owner = None
        key = build_cache_key(
            image_hash=im.image_hash,
            model_key=yolo_model_key,
            prompts=ps.prompts,
            confidence_threshold=thr,
            workflow_id=workflow_id,
            user_id=owner,
        )
        cached_hit = None if force_rerun else cache.get(key)
        if cached_hit is not None:
            run = dict(cached_hit)
            run["run_id"] = str(uuid.uuid4())
            run["session_id"] = session_id
            run["cached"] = True
            run["prompt_set_label"] = ps.name
            run["prompt_set"] = list(ps.prompts)
            run["confidence_threshold"] = thr
            run["image_name"] = im.image_name
            run["image_hash"] = im.image_hash
            run["image_source"] = im.image_source
            run["include_in_aggregate"] = im.include_in_aggregate
            run["expected_count"] = im.expected_count
            run["object_definition"] = im.object_definition
            run["reviewed"] = False
            run["precision_status"] = "not_reviewed"
            run["recall_status"] = "not_reviewed"
            run["true_positives"] = None
            run["false_positives"] = None
            run["false_negatives"] = None
            run["precision"] = None
            run["recall"] = None
            m = compute_benchmark_metrics(
                ai_count=int(run.get("final_count") or 0),
                expected_count=im.expected_count,
                execution_failed=bool(run.get("execution_failed")),
            )
            run["count_error"] = m.count_error
            run["count_accuracy"] = m.count_accuracy
            run["evaluation"] = m.evaluation
            run["exact_match"] = m.evaluation == "exact_count_match"
            run["difference"] = (
                (int(run.get("final_count") or 0) - int(im.expected_count))
                if im.expected_count is not None
                else None
            )
            runs.append(run)
            continue

        data = bytes_map.get(im.image_hash)
        if not data:
            failures += 1
            runs.append(
                {
                    "run_id": str(uuid.uuid4()),
                    "session_id": session_id,
                    "image_name": im.image_name,
                    "image_hash": im.image_hash,
                    "prompt_set_label": ps.name,
                    "prompt_set": list(ps.prompts),
                    "confidence_threshold": thr,
                    "expected_count": im.expected_count,
                    "final_count": 0,
                    "execution_failed": True,
                    "success": False,
                    "error_message": "Missing image bytes",
                    "include_in_aggregate": im.include_in_aggregate,
                    "reviewed": False,
                    "fallback_used": False,
                }
            )
            continue
        try:
            outcome = run_yolo_world(
                image_bytes=data,
                image_name=im.image_name,
                prompts=list(ps.prompts),
                prompt_set_label=ps.name,
                confidence_threshold=float(thr),
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            outcome = BenchmarkRunOutcome(
                prompt_set_label=ps.name,
                prompt_set=list(ps.prompts),
                success=False,
                execution_failed=True,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        if outcome.execution_failed:
            failures += 1
        run = outcome_to_run_dict(
            outcome,
            image=im,
            confidence_threshold=thr,
            model_key=yolo_model_key,
            session_id=session_id,
            cached=False,
            reviewed=False,
        )
        if outcome.annotated_image_bytes:
            annotated[run["run_id"]] = outcome.annotated_image_bytes
        cache.put(
            key,
            {
                **run,
                "success": outcome.success,
                "execution_failed": outcome.execution_failed,
            },
        )
        runs.append(run)

    bar.progress(1.0)
    st.session_state.batch_progress = {
        "completed": len(runs),
        "total": total,
        "failures": failures,
        "image": "",
        "prompt_set": "",
        "threshold": None,
    }
    st.session_state.batch_run_cache = cache.to_dict().get("entries") or {}
    st.session_state.batch_annotated = annotated
    session = build_batch_session(
        inventory_key=inventory_key,
        model_key=yolo_model_key,
        images=specs,
        prompt_sets=prompt_sets,
        thresholds=thresholds,
        runs=runs,
        custom_item_name=custom_item_name,
        session_id=session_id,
    )
    session = _owned(session)
    save_batch_session(session)
    st.session_state.batch_session = session
