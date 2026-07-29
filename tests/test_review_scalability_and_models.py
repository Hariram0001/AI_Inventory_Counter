"""Scalable review navigation, confidence labeling, and model registry resilience."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import app as app_module
import config
from confidence_ui import (
    CONFIDENCE_HELP,
    CONFIDENCE_LABEL,
    confidence_band,
    format_confidence_percent,
)
from detection_viz import assign_marker_numbers
from inventory_config import resolve_recommended_model
from model_registry import (
    get_selectable_analysis_models,
    load_models_from_file,
    normalize_model_name,
    sanitize_selected_model_names,
)
from review_navigation import (
    build_synthetic_detections,
    filter_detections,
    format_detection_option,
    paginate,
    step_detection_id,
)


def test_150_detections_no_per_detection_buttons():
    dets = assign_marker_numbers(build_synthetic_detections(150))
    assert len(dets) == 150
    src = inspect.getsource(app_module.stage_review)
    assert "rev_chip_" not in src
    assert "Previous" in src and "Next" in src
    assert "Jump to" in src
    assert "rev_det_filter" in src
    # Navigator helpers scale without creating 150 widgets in pure logic
    page, _, pages = paginate(dets, 0, 15)
    assert len(page) == 15
    assert pages == 10
    nxt = step_detection_id(dets, dets[0].detection_id, delta=1)
    assert nxt == dets[1].detection_id


def test_jump_and_filters_work():
    dets = assign_marker_numbers(build_synthetic_detections(40))
    warns = filter_detections(dets, "warnings")
    assert all(
        d.suspected_overlap or d.suspected_occlusion or d.confidence < 0.5 for d in warns
    )
    manuals = filter_detections(dets, "manual")
    assert manuals == []
    label = format_detection_option(dets[0])
    assert label.startswith("#1")
    assert "synth-" not in label
    assert "—" in label


def test_confidence_labeled_not_accuracy():
    assert format_confidence_percent(0.55) == "55%"
    assert format_confidence_percent(0.8234) == "82.3%"
    assert confidence_band(0.8) == "High"
    assert confidence_band(0.55) == "Medium"
    assert confidence_band(0.2) == "Low"
    assert "accuracy" not in CONFIDENCE_HELP.lower() or "not a measured accuracy" in CONFIDENCE_HELP
    src = inspect.getsource(app_module.stage_review)
    assert "CONFIDENCE_LABEL" in src or CONFIDENCE_LABEL in src
    assert CONFIDENCE_HELP.split()[0]  # imported helper used in UI
    assert "CONFIDENCE_HELP" in src
    assert "Classification rate" not in src
    assert "Detection accuracy" not in src


def test_yolo_world_generic_name_and_inventory_prompt():
    models = load_models_from_file()
    names = {m.name for m in models}
    assert "YOLO-World" in names
    assert "YOLO-World Fence Panel" not in names
    yolo = next(m for m in models if m.name == "YOLO-World")
    assert yolo.supports_prompt or yolo.dynamic_classes
    assert yolo.workflow_id == "custom-workflow"
    prompt = config.inventory_detection_prompt("Fence Panel")
    assert "fence panel" in prompt.lower()
    resolved = resolve_recommended_model(
        "Fence Panel", models, config.INVENTORY_MODEL_RECOMMENDATIONS, allow_demo=False
    )
    assert resolved["ok"]
    assert resolved["model_name"] == "YOLO-World"
    assert "fence panel" in (resolved["prompt"] or "").lower()


def test_local_picket_optional_in_live_selector():
    selectable = get_selectable_analysis_models(
        load_models_from_file(), "Fence Panel", allow_demo=False
    )
    names = {m.name for m in selectable}
    assert "Local Picket Counter" in names
    assert "YOLO-World" in names
    assert all(not m.demo_only for m in selectable)


def test_stale_model_selection_cleared():
    available = ["YOLO-World"]
    cleaned = sanitize_selected_model_names(
        ["YOLO-World Fence Panel", "YOLOv10", "gone-model"],
        available,
    )
    assert cleaned == ["YOLO-World"]
    assert normalize_model_name("YOLO-World Fence Panel") == "YOLO-World"


def test_deleted_yolov10_does_not_break_registry():
    # No YOLOv10 entry exists; loading registry must still succeed.
    models = load_models_from_file()
    assert models
    selectable = get_selectable_analysis_models(models, "Fence Panel", allow_demo=False)
    assert selectable
    assert all(m.name != "YOLOv10" for m in selectable)


def test_models_json_saved_without_api_key():
    raw = Path("models.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    blob = json.dumps(data).lower()
    assert "api_key" not in blob
    assert "roboflow_api_key" not in blob


def test_review_tabs_still_present():
    src = inspect.getsource(app_module.stage_review)
    for tab in ("Adjustments", "Duplicates", "Warnings", "Details"):
        assert tab in src
    assert "Technical Details" in src
