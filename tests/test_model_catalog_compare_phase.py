"""Offline tests for Model Catalog schema, filtering, and Compare gating."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from comparison_helpers import (
    COMPARE_MAX_MODELS,
    COMPARE_MIN_MODELS,
    compare_peer_models,
    validate_compare_selection,
)
from model_adapters import UnsupportedModelAdapter, get_adapter, resolve_adapter_type
from model_catalog import (
    ADAPTER_ROBOFLOW_OD,
    ADAPTER_YOLO_WORLD,
    STATUS_METADATA_ONLY,
    STATUS_READY,
    CatalogEntry,
    add_approved_public_model,
    catalog_diagnostics_summary,
    get_selectable_models,
    infer_compatible_inventories,
    load_registered_foundation_models,
    migrate_catalog_schema,
    save_catalog_entries,
)
from model_registry import get_selectable_analysis_models, load_models_from_file, save_models_to_file
from schemas import ModelConfig
from secret_scan import find_persisted_secrets


def test_schema_migration_preserves_yolo_world(tmp_path, monkeypatch):
    import model_catalog as mc

    monkeypatch.setattr(mc, "CATALOG_PATH", tmp_path / "catalog.json")
    legacy = [
        CatalogEntry(
            key="workflow:hariram-s-mzhvc/custom-workflow",
            display_name="YOLO-World",
            source="foundation",
            adapter_type="roboflow_workflow",
            workflow_id="custom-workflow",
            dynamic_classes=True,
            enabled=True,
            validated=True,
            status=STATUS_READY,
            kind="workflow",
        ),
        CatalogEntry(
            key="foundation:rf-detr",
            display_name="RF-DETR",
            source="foundation",
            adapter_type="none",
            status="unavailable",
            kind="model",
        ),
    ]
    migrated = migrate_catalog_schema(legacy)
    names = {e.display_name for e in migrated}
    assert "YOLO-World" in names
    assert "RF-DETR" not in names
    yw = next(e for e in migrated if e.display_name == "YOLO-World")
    assert yw.adapter_type == ADAPTER_YOLO_WORLD
    assert yw.dynamic_prompts is True
    assert yw.validation_status == "ready"


def test_atomic_catalog_save_backup(tmp_path, monkeypatch):
    import model_catalog as mc

    path = tmp_path / "catalog.json"
    monkeypatch.setattr(mc, "CATALOG_PATH", path)
    entries = load_registered_foundation_models()
    save_catalog_entries(entries, backup=False)
    assert path.exists()
    save_catalog_entries(entries, backup=True)
    assert path.with_suffix(".json.bak").exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == 2
    assert not find_persisted_secrets(payload)


def test_infer_inventories_boxes_not_custom():
    inv = infer_compatible_inventories(["cardboard-box", "wooden-pallet"])
    assert "Boxes" in inv
    assert "Pallets" in inv
    assert "Custom Item" not in inv
    assert "Traffic Cones" not in inv


def test_fixed_class_excluded_from_incompatible_inventory(tmp_path, monkeypatch):
    import model_catalog as mc

    monkeypatch.setattr(mc, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(mc, "PUBLIC_MODELS_PATH", tmp_path / "public.json")
    models_path = tmp_path / "models.json"
    monkeypatch.setattr(mc, "MODELS_JSON_PATH", models_path)
    import model_registry as mr

    monkeypatch.setattr(mr, "MODELS_JSON_PATH", models_path)

    yw = load_registered_foundation_models()[0]
    fixed = CatalogEntry(
        key="model:boxes-only/1",
        display_name="Boxes Only",
        source="workspace",
        adapter_type=ADAPTER_ROBOFLOW_OD,
        model_id="boxes-only/1",
        kind="model",
        enabled=True,
        validated=True,
        status=STATUS_READY,
        supported_classes=["cardboard-box"],
        supported_inventory_types=["Boxes"],
        dynamic_classes=False,
    )
    save_catalog_entries([yw, fixed], backup=False)
    save_models_to_file([yw.to_model_config(), fixed.to_model_config()], path=models_path, backup=False)

    boxes = get_selectable_models("Boxes", allow_demo=False)
    cones = get_selectable_models("Traffic Cones", allow_demo=False)
    custom = get_selectable_models("Custom Item", allow_demo=False, custom_item=True)
    assert any(m.name == "Boxes Only" for m in boxes)
    assert not any(m.name == "Boxes Only" for m in cones)
    assert not any(m.name == "Boxes Only" for m in custom)
    assert any(m.name == "YOLO-World" for m in custom)


def test_public_duplicate_rejected(tmp_path, monkeypatch):
    import model_catalog as mc

    monkeypatch.setattr(mc, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(mc, "PUBLIC_MODELS_PATH", tmp_path / "public.json")
    monkeypatch.setattr(mc, "_merge_entry_into_models_json", lambda e: None)
    monkeypatch.setattr(mc, "load_catalog_entries", lambda: [])

    with patch.object(
        mc,
        "validate_universe_model_id",
        return_value=(
            True,
            "ok",
            {
                "workspace": "ws",
                "project": "proj",
                "version": "1",
                "classes": ["box"],
                "task_type": "object_detection",
                "has_model": True,
            },
        ),
    ):
        ok1, _, e1 = add_approved_public_model(
            model_id="proj/1",
            display_name="Pub",
            supported_classes=["box"],
            supported_inventory_types=["Boxes"],
            require_metadata_validation=True,
        )
        assert ok1 and e1 is not None
        assert e1.status == STATUS_METADATA_ONLY
        assert e1.validated is False
        assert e1.enabled is False
        ok2, msg, _ = add_approved_public_model(
            model_id="proj/1",
            display_name="Pub2",
            supported_classes=["box"],
            require_metadata_validation=True,
        )
        assert ok2 is False
        assert "already" in msg.lower()


def test_adapter_factory_routing():
    yw = ModelConfig(
        name="YOLO-World",
        kind="workflow",
        enabled=True,
        workspace_name="hariram-s-mzhvc",
        workflow_id="custom-workflow",
        supports_prompt=True,
        dynamic_classes=True,
        key="workflow:hariram-s-mzhvc/custom-workflow",
    )
    assert resolve_adapter_type(yw) == ADAPTER_YOLO_WORLD
    adapter = get_adapter(yw)
    assert type(adapter).__name__ == "RoboflowWorkflowAdapter"

    bad = ModelConfig(name="Nope", kind="model", enabled=True, model_id="")
    assert isinstance(get_adapter(bad), UnsupportedModelAdapter)


def test_compare_requires_two_max_three():
    names = ["A", "B", "C", "D"]
    assert validate_compare_selection(["A"], names)
    assert not validate_compare_selection(["A", "B"], names)
    assert validate_compare_selection(names, names)
    assert COMPARE_MIN_MODELS == 2
    assert COMPARE_MAX_MODELS == 3


def test_fence_compare_has_two_peers_today():
    models = load_models_from_file()
    selectable = get_selectable_analysis_models(models, "Fence Panel", allow_demo=False)
    peers = compare_peer_models(selectable)
    assert len(peers) >= 2
    names = {m.name for m in peers}
    assert "YOLO-World" in names
    assert "Local Picket Counter" in names


def test_boxes_only_one_peer_disables_compare():
    models = load_models_from_file()
    selectable = get_selectable_analysis_models(models, "Boxes", allow_demo=False)
    peers = compare_peer_models(selectable)
    assert "YOLO-World" in {m.name for m in peers}
    assert "Local Picket Counter" not in {m.name for m in peers}
    assert len(peers) < COMPARE_MIN_MODELS


def test_zero_detections_distinct_from_failure():
    from comparison_helpers import human_status

    assert human_status(success=True, final_count=0, error_type=None) == (
        "Success with zero detections"
    )
    assert human_status(success=False, final_count=None, error_type="api_error") != (
        "Success with zero detections"
    )


def test_catalog_diagnostics_no_secrets():
    summary = catalog_diagnostics_summary()
    blob = json.dumps(summary)
    assert "ROBOFLOW_API_KEY" not in blob
    assert "api_key=" not in blob.lower()


def test_metadata_validation_rejects_non_od(tmp_path, monkeypatch):
    import model_catalog as mc

    monkeypatch.setattr(mc, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(mc, "PUBLIC_MODELS_PATH", tmp_path / "public.json")
    monkeypatch.setattr(mc, "_merge_entry_into_models_json", lambda e: None)
    monkeypatch.setattr(mc, "load_catalog_entries", lambda: [])
    with patch.object(
        mc,
        "validate_universe_model_id",
        return_value=(
            True,
            "ok",
            {
                "workspace": "ws",
                "project": "c",
                "version": "1",
                "classes": ["x"],
                "task_type": "classification",
                "has_model": True,
            },
        ),
    ):
        ok, msg, _ = add_approved_public_model(
            model_id="c/1",
            display_name="Cls",
            task_type="classification",
            supported_classes=["x"],
        )
        assert ok is False
        assert "object-detection" in msg.lower()
