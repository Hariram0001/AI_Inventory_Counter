"""Roboflow-style Model Catalog UI (Settings → AI Configuration)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

import config
from model_adapters import InferenceOptions, get_adapter, model_key
from model_catalog import (
    SOURCE_DEMO,
    SOURCE_FOUNDATION,
    SOURCE_LOCAL,
    SOURCE_UNIVERSE,
    SOURCE_WORKSPACE,
    STATUS_FAILED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    VALIDATION_LEVEL_LIVE,
    CatalogEntry,
    add_approved_public_model,
    filter_catalog_entries,
    get_all_catalog_models,
    last_sync_report,
    load_catalog_entries,
    load_future_capabilities,
    remove_from_catalog,
    save_catalog_entries,
    set_catalog_entry_enabled,
    sync_workspace_models,
    validate_model,
)
from model_registry import load_models_from_file, save_models_to_file
from schemas import ModelConfig


def inject_catalog_css() -> None:
    st.markdown(
        """
        <style>
        .aic-badge {
            display: inline-block; font-size: 0.68rem; font-weight: 650;
            padding: 0.12rem 0.45rem; border-radius: 999px; margin-right: 0.25rem;
            border: 1px solid rgba(128,128,128,0.28);
        }
        .aic-badge-workspace { background: rgba(46,160,67,0.18); border-color: rgba(46,160,67,0.45); }
        .aic-badge-foundation { background: rgba(33,150,243,0.16); border-color: rgba(33,150,243,0.42); }
        .aic-badge-public { background: rgba(255,75,75,0.14); border-color: rgba(255,75,75,0.4); }
        .aic-badge-demo { background: rgba(255,170,0,0.18); border-color: rgba(255,170,0,0.45); }
        .aic-badge-unavailable { background: rgba(128,128,128,0.12); opacity: 0.85; }
        .aic-model-card {
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 10px;
            padding: 0.55rem 0.65rem;
            margin-bottom: 0.35rem;
            background: rgba(250,250,250,0.03);
        }
        .aic-model-card-selected {
            border-color: rgba(250,250,250,0.35);
            box-shadow: 0 0 0 1px rgba(250,250,250,0.12);
        }
        .aic-catalog-layout {
            border: 1px solid rgba(128,128,128,0.16);
            border-radius: 10px;
            padding: 0.55rem;
            background: rgba(250,250,250,0.02);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _source_badge(source: str, demo_only: bool = False) -> str:
    if demo_only or source == SOURCE_DEMO:
        return '<span class="aic-badge aic-badge-demo">Demo</span>'
    if source == SOURCE_LOCAL:
        return '<span class="aic-badge aic-badge-demo">Local</span>'
    mapping = {
        SOURCE_WORKSPACE: ("Workspace", "aic-badge-workspace"),
        SOURCE_FOUNDATION: ("Foundation", "aic-badge-foundation"),
        SOURCE_UNIVERSE: ("Public", "aic-badge-public"),
    }
    label, cls = mapping.get(source, ("Unavailable", "aic-badge-unavailable"))
    return f'<span class="aic-badge {cls}">{label}</span>'


def _status_label(entry: CatalogEntry) -> str:
    if entry.stale:
        return "Unavailable during last sync"
    if entry.status == STATUS_UNAVAILABLE or entry.adapter_type == "none":
        return "Deployment unavailable"
    if entry.validated or entry.status == STATUS_READY:
        return "Ready"
    if entry.last_test_status in {"Failed", "failed"}:
        return "Failed validation"
    return "Needs configuration"


def render_model_catalog_section(
    *,
    run_model_test: Any = None,
    get_test_image_bytes: Any = None,
) -> None:
    """Full Model Catalog browser for Settings → AI Configuration."""
    inject_catalog_css()
    st.markdown("##### Model Catalog")
    st.caption(
        "Discover workspace models, browse foundation adapters, and approve public model IDs. "
        "API keys are never shown or stored in the catalog."
    )

    if "catalog_selected_key" not in st.session_state:
        st.session_state.catalog_selected_key = None
    if "catalog_source_tab" not in st.session_state:
        st.session_state.catalog_source_tab = "My Workspace"

    # Toolbar
    t1, t2, t3, t4 = st.columns([1.4, 1, 1, 1])
    with t1:
        search = st.text_input("Search models", key="catalog_search", placeholder="Name, class, project…")
    with t2:
        task_filter = st.selectbox(
            "Task",
            [
                "All",
                "object_detection",
                "instance_segmentation",
                "classification",
                "multimodal",
            ],
            key="catalog_task_filter",
        )
    with t3:
        status_filter = st.selectbox(
            "Status",
            [
                "All",
                "Ready",
                "Enabled",
                "Needs Configuration",
                "Failed Validation",
                "Compatible with Fence Panels",
            ],
            key="catalog_status_filter",
        )
    with t4:
        st.write("")
        if st.button("Refresh Workspace", type="primary", key="catalog_sync_btn", width="stretch"):
            with st.spinner("Syncing workspace models…"):
                report = sync_workspace_models()
            st.session_state.catalog_last_sync = report
            if report.get("ok"):
                st.success(
                    f"Sync OK — {report.get('projects_found', 0)} projects · "
                    f"{report.get('models_registered', 0)} trained models · "
                    f"{report.get('versions_skipped_unusable', 0)} unusable versions skipped"
                )
            else:
                st.error("; ".join(report.get("errors") or ["Sync failed."]))
            st.rerun()

    report = st.session_state.get("catalog_last_sync") or last_sync_report()
    if report:
        st.caption(
            f"Last sync: {report.get('finished_at') or report.get('started_at') or '—'} · "
            f"method: {report.get('method') or '—'} · "
            f"errors: {len(report.get('errors') or [])}"
        )

    tab_ws, tab_f, tab_p = st.tabs(["My Workspace", "Foundation Models", "Public Models"])
    entries = get_all_catalog_models()

    status_map = {
        "All": "all",
        "Ready": "ready",
        "Enabled": "enabled",
        "Needs Configuration": "needs_configuration",
        "Failed Validation": "failed_validation",
        "Compatible with Fence Panels": "compatible_fence",
    }
    status_key = status_map.get(status_filter, "all")
    task_key = None if task_filter == "All" else task_filter
    compatible = status_key == "compatible_fence"
    enabled_only = status_key == "enabled"

    with tab_ws:
        _render_source_cards(
            filter_catalog_entries(
                entries,
                search=search,
                source=SOURCE_WORKSPACE,
                task_type=task_key,
                status=status_key if status_key not in {"compatible_fence", "enabled"} else "all",
                compatible_fence=compatible,
                enabled_only=enabled_only,
            ),
            empty=(
                "No trained object-detection projects were found in the configured "
                "Roboflow workspace. Click Refresh Workspace after training a model, "
                "or register a public model ID."
            ),
        )
    with tab_f:
        _render_source_cards(
            filter_catalog_entries(
                [
                    e
                    for e in entries
                    if e.source in {SOURCE_FOUNDATION, SOURCE_LOCAL}
                    and e.adapter_type != "none"
                ],
                search=search,
                task_type=task_key,
                status=status_key if status_key not in {"compatible_fence", "enabled"} else "all",
                compatible_fence=compatible,
                enabled_only=enabled_only,
            ),
            empty="No foundation or local models registered.",
        )
        with st.expander("Future Capabilities (informational)", expanded=False):
            st.caption(
                "These capabilities are not selectable for object counting in this POC."
            )
            for item in load_future_capabilities():
                st.markdown(
                    f"- **{item['name']}** ({item['task']}) — {item['note']}"
                )
    with tab_p:
        st.markdown("**Add Public Object-Detection Model**")
        st.caption(
            "Paste a real Roboflow model ID (project/version). "
            "Universe models are not bulk-imported. Discovery ≠ live validation."
        )
        c1, c2 = st.columns(2)
        with c1:
            pub_id = st.text_input("Model ID", key="catalog_pub_id", placeholder="my-project/3")
            pub_name = st.text_input("Display name", key="catalog_pub_name")
            pub_desc = st.text_input("Description (optional)", key="catalog_pub_desc")
            pub_classes = st.text_input(
                "Supported classes (comma-separated)",
                key="catalog_pub_classes",
                placeholder="cardboard-box, wooden-pallet",
            )
        with c2:
            pub_task = st.selectbox(
                "Task type",
                ["object_detection"],
                key="catalog_pub_task",
            )
            pub_inv = st.multiselect(
                "Compatible inventory profiles",
                [
                    "Fence Panel",
                    "Boxes",
                    "Pallets",
                    "Poles",
                    "Gates",
                    "Chairs",
                    "Traffic Cones",
                ],
                default=[],
                key="catalog_pub_inv",
            )
            pub_conf = st.number_input(
                "Default confidence",
                min_value=0.05,
                max_value=0.95,
                value=0.20,
                step=0.05,
                key="catalog_pub_conf",
            )
            pub_license = st.text_input("License (optional)", key="catalog_pub_license")
        if st.button("Validate & Add", key="catalog_pub_add"):
            if not config.api_key_configured():
                st.error("API key not configured.")
            else:
                class_list = [
                    c.strip() for c in (pub_classes or "").split(",") if c.strip()
                ]
                with st.spinner("Validating model metadata…"):
                    ok, msg, entry = add_approved_public_model(
                        model_id=pub_id,
                        display_name=pub_name or pub_id,
                        task_type=pub_task,
                        supported_classes=class_list or None,
                        supported_inventory_types=pub_inv,
                        license_info=pub_license or None,
                        description=pub_desc or None,
                        default_confidence=float(pub_conf),
                        require_metadata_validation=True,
                        require_live_validation=False,
                    )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        # Also show local/demo in public tab? No — keep under foundation/workspace only.
        # Show demo entries when DEMO_MODE for transparency
        demo_entries = [
            e
            for e in entries
            if e.source in {SOURCE_DEMO, SOURCE_LOCAL} or e.demo_only
        ]
        if config.DEMO_MODE and demo_entries:
            st.markdown("**Demo / local (DEMO_MODE)**")
            _render_source_cards(demo_entries, empty="")
        _render_source_cards(
            filter_catalog_entries(
                [e for e in entries if e.source == SOURCE_UNIVERSE],
                search=search,
                task_type=task_key,
                compatible_fence=compatible,
                enabled_only=enabled_only,
            ),
            empty="No approved public models yet.",
        )

    # Details panel
    sel = st.session_state.get("catalog_selected_key")
    if sel:
        entry = next((e for e in get_all_catalog_models() if e.key == sel), None)
        if entry:
            _render_details_panel(
                entry,
                run_model_test=run_model_test,
                get_test_image_bytes=get_test_image_bytes,
            )


def _render_source_cards(entries: list[CatalogEntry], *, empty: str) -> None:
    if not entries:
        if empty:
            st.info(empty)
        return
    for entry in entries:
        selected = st.session_state.get("catalog_selected_key") == entry.key
        cls = "aic-model-card aic-model-card-selected" if selected else "aic-model-card"
        classes_preview = ", ".join((entry.supported_classes or [])[:6]) or (
            "(dynamic / open)" if entry.dynamic_classes or entry.dynamic_prompts else "(none)"
        )
        if entry.supported_classes and len(entry.supported_classes) > 6:
            classes_preview += "…"
        inv = ", ".join(entry.supported_inventory_types or []) or (
            "(any / dynamic)"
            if entry.dynamic_classes or entry.dynamic_prompts
            else "(none declared)"
        )
        exec_label = (entry.execution_type or entry.kind or "—").replace("_", " ")
        st.markdown(
            f"""
            <div class="{cls}">
              {_source_badge(entry.source, entry.demo_only)}
              <b>{entry.display_name}</b><br/>
              <span class="aic-muted">{entry.task_type.replace('_', ' ')}
              · {exec_label}
              · {_status_label(entry)}</span><br/>
              <span class="aic-muted">Classes: {classes_preview}<br/>
              Dynamic prompts: {"Yes" if entry.dynamic_classes or entry.dynamic_prompts or entry.supports_prompt else "No"}
              · Inventories: {inv}<br/>
              Enabled: {"Yes" if entry.enabled and not entry.stale else "No"}
              · Last tested: {entry.last_tested_at or "—"}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Details", key=f"cat_det_{entry.key}", width="stretch"):
                st.session_state.catalog_selected_key = entry.key
                st.rerun()
        with b2:
            new_en = st.toggle(
                "Enabled",
                value=bool(entry.enabled) and not entry.stale and entry.adapter_type != "none",
                key=f"cat_en_{entry.key}",
                disabled=entry.adapter_type == "none" or entry.stale,
            )
            if new_en != bool(entry.enabled) and entry.adapter_type != "none" and not entry.stale:
                set_catalog_entry_enabled(entry.key, new_en)
                st.rerun()
        with b3:
            if st.button("Test", key=f"cat_test_{entry.key}", width="stretch"):
                st.session_state.catalog_selected_key = entry.key
                st.session_state.catalog_pending_test = entry.key
                st.rerun()
        with b4:
            if st.button("Use", key=f"cat_use_{entry.key}", width="stretch"):
                if entry.adapter_type == "none" or entry.stale:
                    st.warning("Deployment unavailable for this model.")
                else:
                    set_catalog_entry_enabled(entry.key, True)
                    st.session_state.form = dict(st.session_state.get("form") or {})
                    st.session_state.form["selected_models"] = [entry.display_name]
                    st.session_state.form["selected_mode"] = "Single Model"
                    st.success(f"Selected {entry.display_name} for Analysis.")


def _render_details_panel(
    entry: CatalogEntry,
    *,
    run_model_test: Any = None,
    get_test_image_bytes: Any = None,
) -> None:
    st.markdown("---")
    st.markdown(f"#### {entry.display_name}")
    st.markdown(
        f"{_source_badge(entry.source, entry.demo_only)} "
        f"**Status:** {_status_label(entry)}",
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"""
            - Source: {entry.source}
            - Task: {entry.task_type}
            - Execution: {(entry.execution_type or "—").replace("_", " ")}
            - Architecture: {entry.architecture or "—"}
            - Deployment: {entry.deployment or "—"}
            - Validation: {entry.validation_status or entry.status}
            """
        )
    with d2:
        st.markdown(
            f"""
            - Dynamic prompts: {"Yes" if entry.dynamic_classes or entry.dynamic_prompts or entry.supports_prompt else "No"}
            - Classes: {", ".join(entry.supported_classes) or "(dynamic/open)"}
            - Inventories: {", ".join(entry.supported_inventory_types) or "(any / dynamic)"}
            - Default confidence: {entry.default_confidence}
            - Default IoU: {entry.default_iou}
            - Last tested: {entry.last_tested_at or "—"}
            - Last test status: {entry.last_test_status or "—"}
            - License: {entry.license or "—"}
            """
        )
    with st.expander("Technical IDs", expanded=False):
        st.markdown(
            f"""
            - Key: `{entry.key}`
            - Workflow ID: `{entry.workflow_id or "—"}`
            - Project ID: `{entry.project_id or "—"}`
            - Model ID: `{entry.model_id or "—"}`
            - Version: `{entry.version or "—"}`
            - Adapter: `{entry.adapter_type}`
            - Workspace: `{entry.workspace or entry.workspace_id or "—"}`
            """
        )
    if entry.validation_message or entry.sync_note:
        st.caption(entry.validation_message or entry.sync_note)

    v = validate_model(entry.key)
    if not v.get("ok"):
        st.warning(v.get("message") or "Not ready")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("Use Model", key="cat_panel_use", type="primary"):
            if entry.adapter_type != "none" and not entry.stale:
                set_catalog_entry_enabled(entry.key, True)
                st.session_state.form = dict(st.session_state.get("form") or {})
                st.session_state.form["selected_models"] = [entry.display_name]
                st.success(f"Using {entry.display_name}")
    with a2:
        if st.button("Test Model", key="cat_panel_test"):
            st.session_state.catalog_pending_test = entry.key
    with a3:
        if st.button("Disable" if entry.enabled else "Enable", key="cat_panel_en"):
            set_catalog_entry_enabled(entry.key, not entry.enabled)
            st.rerun()
    with a4:
        if st.button("Remove from Catalog", key="cat_panel_rm"):
            remove_from_catalog(entry.key)
            st.session_state.catalog_selected_key = None
            st.rerun()

    # Dedicated test (does not modify inventory wizard uploads)
    pending = st.session_state.get("catalog_pending_test")
    if pending == entry.key or st.session_state.get("catalog_force_test_ui"):
        st.markdown("**Model test**")
        st.caption(
            "Dedicated probe only — does not change inventory photos. "
            "Upload or capture a test image, or use the Settings probe image."
        )
        test_up = st.file_uploader(
            "Upload Test Image",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"cat_test_upload_{entry.key}",
        )
        if test_up is not None:
            test_up.seek(0)
            st.session_state.ai_config_test_image_bytes = test_up.read()
            st.session_state.ai_config_test_image_name = test_up.name
        cam = st.camera_input("Use Camera", key=f"cat_test_cam_{entry.key}")
        if cam is not None:
            cam.seek(0)
            st.session_state.ai_config_test_image_bytes = bytes(cam.getvalue())
            st.session_state.ai_config_test_image_name = "catalog_camera_test.jpg"
        if run_model_test:
            if st.button("Test Model", key="cat_run_test_now", type="primary"):
                cfg = entry.to_model_config()
                if entry.adapter_type == "none":
                    st.error("Deployment unavailable")
                else:
                    with st.spinner("Testing…"):
                        result = run_model_test(cfg)
                    result["model_key"] = entry.key
                    _stamp_test_result(entry, result)
                    st.session_state.catalog_test_result = result
                    st.rerun()
        result = st.session_state.get("catalog_test_result")
        if isinstance(result, dict) and result.get("model_key") in {entry.key, entry.display_name}:
            if result.get("ok"):
                st.success("Test passed")
            else:
                st.error(result.get("message") or "Test failed")
            st.markdown(
                f"""
                - Authentication: {result.get("auth")}
                - Execution: {"OK" if result.get("ok") else "Failed"}
                - Response source: {result.get("response_source")}
                - Raw count: {result.get("raw_prediction_count")}
                - Normalized count: {result.get("normalized_prediction_count")}
                - Classes: {", ".join(result.get("detected_classes") or []) or "(none)"}
                - Latency: {float(result.get("processing_time") or 0):.2f}s
                - Parser: {result.get("parser_status")}
                """
            )
            if result.get("error_message"):
                st.caption(f"Error: {result.get('error_message')}")
            with st.expander("View Sanitized Result", expanded=False):
                safe = {
                    k: v
                    for k, v in result.items()
                    if k not in {"annotated_preview"}
                    and "api_key" not in str(k).lower()
                }
                st.json(safe)
            if result.get("annotated_preview"):
                st.image(result["annotated_preview"], width="stretch")


def _stamp_test_result(entry: CatalogEntry, result: dict[str, Any]) -> None:
    entries = load_catalog_entries()
    for e in entries:
        if e.key == entry.key:
            e.last_tested_at = datetime.now(timezone.utc).isoformat()
            e.last_test_status = "OK" if result.get("ok") else "Failed"
            e.validated = bool(result.get("ok"))
            e.validation_level = VALIDATION_LEVEL_LIVE
            if result.get("ok"):
                e.status = STATUS_READY
                e.validation_status = "ready"
                e.validation_message = "Live inference test succeeded."
                # Zero detections still prove executability
                if int(result.get("normalized_prediction_count") or 0) == 0 and int(
                    result.get("raw_prediction_count") or 0
                ) == 0:
                    e.validation_message = (
                        "Live test succeeded with zero detections "
                        "(executable; not an accuracy proof)."
                    )
            else:
                e.status = STATUS_FAILED
                e.validation_status = "failed"
                e.validation_message = str(result.get("message") or result.get("error_message") or "Failed")
            e.normalize_schema_fields()
            break
    save_catalog_entries(entries)
    # Mirror validation onto models.json — enable only after successful live test
    models = load_models_from_file()
    for m in models:
        if (m.key or model_key(m)) == entry.key and result.get("ok"):
            m.enabled = True
    save_models_to_file(models)
    # Keep approved public store in sync
    try:
        from model_catalog import load_approved_public_models, save_approved_public_models

        pubs = load_approved_public_models()
        changed = False
        for p in pubs:
            if p.key == entry.key:
                p.validated = bool(result.get("ok"))
                p.status = STATUS_READY if result.get("ok") else STATUS_FAILED
                p.last_tested_at = datetime.now(timezone.utc).isoformat()
                p.last_test_status = "OK" if result.get("ok") else "Failed"
                p.normalize_schema_fields()
                changed = True
        if changed:
            save_approved_public_models(pubs)
    except Exception:  # noqa: BLE001
        pass


def format_model_option(m: ModelConfig, entries_by_name: dict[str, CatalogEntry] | None = None) -> str:
    entry = (entries_by_name or {}).get(m.name)
    source = (entry.source if entry else (m.provider or m.kind) or "").title()
    task = (entry.task_type if entry else "object_detection").replace("_", " ")
    status = _status_label(entry) if entry else ("Ready" if m.enabled else "Needs configuration")
    return f"{m.name} · {source} · {task} · {status}"


def format_model_info_markdown(
    m: ModelConfig, entry: CatalogEntry | None = None
) -> str:
    """Human-readable About text for Analyze model Info buttons."""
    source = (entry.source if entry else (m.provider or m.kind) or "unknown").replace("_", " ")
    provider = (entry.provider if entry else m.provider) or "—"
    task = (entry.task_type if entry else "object_detection").replace("_", " ")
    adapter = (entry.adapter_type if entry else m.kind) or "—"
    architecture = (entry.architecture if entry else None) or (m.kind or "—")
    status = _status_label(entry) if entry else ("Ready" if m.enabled else "Needs configuration")
    workspace = (entry.workspace if entry else m.workspace_name) or "—"
    workflow = (entry.workflow_id if entry else m.workflow_id) or "—"
    model_id = (entry.model_id if entry else m.model_id) or "—"
    deployment = (entry.deployment if entry else None) or "—"
    classes = list((entry.supported_classes if entry else m.allowed_classes) or [])
    class_txt = ", ".join(classes[:12]) if classes else "Prompt-driven / open vocabulary"
    if classes and len(classes) > 12:
        class_txt += "…"
    note = (entry.sync_note if entry else "") or ""
    demo = bool(getattr(m, "demo_only", False)) or (
        callable(getattr(m, "is_demo_model_id", None)) and m.is_demo_model_id()
    )
    if demo:
        purpose = "Demo fixture that returns stored sample predictions (not live inference)."
    elif (adapter or "").startswith("local") or (m.kind or "").lower() == "local":
        purpose = (
            "Runs entirely on this machine with a classical image heuristic "
            "(pointed fence pickets). No Roboflow API call."
        )
    elif (adapter or "") == "roboflow_workflow" or (m.kind or "").lower() == "workflow":
        purpose = (
            "Hosted Roboflow Workflow via inference-sdk. "
            "Open-vocabulary detection using your text prompts."
        )
    elif (adapter or "") == "roboflow_model":
        purpose = (
            "Hosted Roboflow object-detection model version, invoked through inference-sdk."
        )
    else:
        purpose = "Configured inventory detection model for this analysis run."

    origin = "Imported from Roboflow workspace / foundation catalog"
    if source.lower() == "local":
        origin = "Bundled in this app (`picket_counter.py`)"
    elif source.lower() == "demo":
        origin = "Bundled demo fixture (`sample_responses/`)"
    elif source.lower() == "foundation":
        origin = "Roboflow foundation / Workflow catalog"
    elif source.lower() == "workspace":
        origin = f"Synced from Roboflow workspace `{workspace}`"
    elif source.lower() in {"universe", "public"}:
        origin = "Approved Roboflow Universe / public model ID"

    lines = [
        f"**{m.name}**",
        "",
        f"**What it does:** {purpose}",
        f"**Source:** {source.title()}",
        f"**Imported from:** {origin}",
        f"**Provider:** {provider}",
        f"**Architecture:** {architecture}",
        f"**Task:** {task}",
        f"**Adapter:** `{adapter}`",
        f"**Status:** {status}",
        f"**Workspace:** `{workspace}`",
        f"**Workflow ID:** `{workflow}`",
        f"**Model ID:** `{model_id}`",
        f"**Deployment:** {deployment}",
        f"**Classes:** {class_txt}",
    ]
    if note:
        lines.extend(["", f"**Note:** {note}"])
    return "\n".join(lines)
