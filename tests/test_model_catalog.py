"""Model Catalog: sync, filtering, resilience (no internet in normal pytest)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_catalog import (
    SOURCE_FOUNDATION,
    SOURCE_WORKSPACE,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    CatalogEntry,
    filter_catalog_entries,
    get_selectable_models,
    history_model_label,
    load_registered_foundation_models,
    normalize_workspace_version,
    remove_stale_model_selection,
    sync_workspace_models,
    validate_model,
    validate_universe_model_id,
)
from model_registry import get_selectable_analysis_models, load_models_from_file
from schemas import ModelConfig


FIXTURE_PROJECTS = [
    {
        "id": "hariram-s-mzhvc/fence-panels-abc",
        "type": "object-detection",
        "name": "Fence Panels",
        "url": "fence-panels-abc",
        "classes": {"fence-panel": 12, "post": 4},
        "versions": 2,
    },
    {
        "id": "hariram-s-mzhvc/empty-project",
        "type": "object-detection",
        "name": "Empty Project",
        "url": "empty-project",
        "classes": {},
        "versions": 1,
    },
]


def test_foundation_yolo_world_ready_others_unavailable():
    from model_catalog import ADAPTER_YOLO_WORLD, load_future_capabilities

    foundations = load_registered_foundation_models()
    yw = next(e for e in foundations if e.display_name == "YOLO-World")
    assert yw.status == STATUS_READY
    assert yw.dynamic_classes is True
    assert yw.dynamic_prompts is True
    assert yw.workflow_id == "custom-workflow"
    assert yw.adapter_type == ADAPTER_YOLO_WORLD
    assert yw.validated is True
    assert "Fence" not in yw.display_name or yw.display_name == "YOLO-World"
    # Non-counting / unverified architectures are not registered as Ready models
    assert all(e.display_name == "YOLO-World" for e in foundations)
    future = load_future_capabilities()
    assert future
    assert all("not" in f["note"].lower() or "informational" in f["note"].lower() for f in future)


def test_yolo_world_generic_naming_in_models_json():
    models = load_models_from_file()
    names = [m.name for m in models]
    assert "YOLO-World" in names
    assert "YOLO-World Fence Panel" not in names


def test_local_picket_included_in_live_selector():
    models = load_models_from_file()
    selectable = get_selectable_analysis_models(models, "Fence Panel", allow_demo=False)
    names = {m.name for m in selectable}
    assert "Local Picket Counter" in names
    assert "Demo Fence Detector" not in names
    assert "YOLO-World" in names


def test_local_picket_enabled_not_demo_only():
    models = load_models_from_file()
    local = next(m for m in models if m.name == "Local Picket Counter")
    assert local.demo_only is False
    assert local.enabled is True
    assert (local.kind or "").lower() == "local"


def test_normalize_workspace_version_skips_untrained():
    project = {
        "id": "ws/proj",
        "name": "Proj",
        "type": "object-detection",
        "classes": {"a": 1},
    }
    assert normalize_workspace_version("ws", project, {"version": 1}) is None
    entry = normalize_workspace_version(
        "ws",
        project,
        {"version": 2, "id": "ws/proj/2", "model": {"type": "yolov8"}},
    )
    assert entry is not None
    assert entry.model_id == "proj/2"
    assert entry.source == SOURCE_WORKSPACE


def test_normalize_fence_compatible_inventory():
    project = {
        "id": "ws/fence-x",
        "name": "Fence X",
        "type": "object-detection",
        "classes": {"fence-panel": 3},
    }
    entry = normalize_workspace_version(
        "ws",
        project,
        {"version": 1, "model": {"type": "yolov8"}},
    )
    assert entry is not None
    assert "Fence Panel" in entry.supported_inventory_types


def test_sync_workspace_models_with_fixtures(tmp_path, monkeypatch):
    import model_catalog as mc

    monkeypatch.setattr(mc, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(mc, "SYNC_REPORT_PATH", tmp_path / "sync.json")
    monkeypatch.setattr(mc, "PUBLIC_MODELS_PATH", tmp_path / "public.json")
    # Avoid rewriting real models.json
    monkeypatch.setattr(mc, "_sync_models_json_from_catalog", lambda entries: None)

    def fake_fetch_projects(workspace, **kwargs):
        return FIXTURE_PROJECTS, {"ok": True, "project_count": 2, "workspace": workspace}

    def fake_fetch_versions(workspace, slug, **kwargs):
        if slug == "fence-panels-abc":
            return (
                FIXTURE_PROJECTS[0],
                [
                    {"version": 1, "id": f"{workspace}/{slug}/1", "model": None},
                    {
                        "version": 2,
                        "id": f"{workspace}/{slug}/2",
                        "model": {"type": "yolov8"},
                    },
                ],
                None,
            )
        return FIXTURE_PROJECTS[1], [{"version": 1, "model": None}], None

    with patch.object(mc, "fetch_workspace_projects", side_effect=fake_fetch_projects):
        with patch.object(mc, "fetch_project_versions", side_effect=fake_fetch_versions):
            report = sync_workspace_models(persist=True)

    assert report["ok"] is True
    assert report["projects_found"] == 2
    assert report["models_registered"] == 1
    assert report["versions_skipped_unusable"] >= 1
    assert "api_key" not in json.dumps(report).lower() or "***" in json.dumps(report)


def test_sync_failure_preserves_catalog(tmp_path, monkeypatch):
    import model_catalog as mc

    monkeypatch.setattr(mc, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(mc, "SYNC_REPORT_PATH", tmp_path / "sync.json")
    prior = [
        CatalogEntry(
            key="model:keep/1",
            display_name="Keep",
            source=SOURCE_WORKSPACE,
            model_id="keep/1",
            enabled=True,
            kind="model",
            status=STATUS_READY,
        )
    ]
    mc.save_catalog_entries(prior)

    with patch.object(
        mc,
        "fetch_workspace_projects",
        return_value=([], {"ok": False, "error": "HTTP 401"}),
    ):
        report = sync_workspace_models(persist=True)
    assert report["ok"] is False
    # Prior entries still loadable
    loaded = mc.load_catalog_entries()
    assert any(e.key == "model:keep/1" for e in loaded)


def test_invalid_universe_model_id():
    ok, msg, _ = validate_universe_model_id("not-valid")
    assert ok is False
    assert "project" in msg.lower() or "version" in msg.lower()


def test_filter_catalog_and_fence_compat():
    entries = [
        CatalogEntry(
            key="a",
            display_name="Dyn",
            source=SOURCE_FOUNDATION,
            dynamic_classes=True,
            status=STATUS_READY,
        ),
        CatalogEntry(
            key="b",
            display_name="Other",
            source=SOURCE_WORKSPACE,
            supported_classes=["car"],
            status=STATUS_READY,
        ),
    ]
    fence = filter_catalog_entries(entries, compatible_fence=True)
    assert len(fence) == 1
    assert fence[0].display_name == "Dyn"


def test_remove_stale_selection_fallback():
    cleaned, note = remove_stale_model_selection(
        ["Gone Model", "Also Gone"], inventory_key="Fence Panel"
    )
    assert cleaned  # falls back to YOLO-World or first live
    assert note
    assert "Gone Model" not in cleaned


def test_validate_model_missing():
    result = validate_model("definitely-not-a-real-key-zzz")
    assert result["ok"] is False
    assert "no longer configured" in result["message"].lower()


def test_history_model_label():
    assert "YOLO-World" in history_model_label("YOLO-World")
    label = history_model_label("Deleted Custom Model XYZ")
    assert "no longer configured" in label.lower()


def test_catalog_ui_has_tabs():
    import catalog_ui

    src = inspect.getsource(catalog_ui.render_model_catalog_section)
    assert "My Workspace" in src
    assert "Foundation Models" in src
    assert "Public Models" in src
    assert "Refresh Workspace" in src


def test_analysis_uses_stale_removal():
    import app as app_module

    src = inspect.getsource(app_module.stage_analyze)
    assert "remove_stale_model_selection" in src
    assert "No compatible validated model is available" in src
    assert "Find Best Model" not in src
    # Compare peers include confirmed local inference + Roboflow (not demos)
    assert "compare_peer_models" in src
    assert "Only one compatible validated model is currently available" in src


def test_comparison_workspace_labels():
    import app as app_module

    src = inspect.getsource(app_module.stage_review)
    assert "Most detections" in src
    assert "not accuracy" in src.lower()
    assert "Side by Side" in src
    assert "Use This Result" in src


def test_canonical_result_schema():
    from model_adapters import ModelInferenceResult
    from schemas import Detection

    mir = ModelInferenceResult(
        model_key="workflow:x/y",
        model_display_name="YOLO-World",
        provider="Roboflow",
        success=True,
        response_source="live_roboflow",
        processing_time_seconds=0.5,
        raw_count=1,
        final_count=1,
        avg_confidence=0.5,
        max_confidence=0.5,
        classes=["fence panel"],
        effective_prompt=["fence panel"],
        effective_threshold=0.25,
        model_source="foundation",
        detections=[
            Detection(
                detection_id="1",
                class_name="fence panel",
                confidence=0.5,
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                center_x=5,
                center_y=5,
                width=10,
                height=10,
                source_model="YOLO-World",
                source_image="a.jpg",
            )
        ],
    )
    d = mir.to_canonical_dict()
    for key in (
        "model_key",
        "model_display_name",
        "model_source",
        "task_type",
        "success",
        "raw_count",
        "final_count",
        "effective_prompt",
        "effective_threshold",
    ):
        assert key in d


def test_no_api_key_in_catalog_dict():
    e = CatalogEntry(
        key="k",
        display_name="n",
        source="foundation",
        extra={"api_key": "SECRET", "ROBOFLOW_API_KEY": "SECRET"},
    )
    d = e.to_dict()
    assert "api_key" not in d
    assert "ROBOFLOW_API_KEY" not in d


def test_local_picket_audit_source():
    import picket_counter

    src = inspect.getsource(picket_counter)
    assert "classical" in src.lower() or "tip" in src.lower() or "peak" in src.lower()
    # Not Roboflow
    assert "InferenceHTTPClient" not in src
