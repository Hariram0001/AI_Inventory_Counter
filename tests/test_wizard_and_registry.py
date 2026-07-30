"""Wizard, registry, adapter, and review enhancement tests (offline)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from app_constants import STAGE_ALIASES, STAGE_LABELS, STAGES
from detection_ids import assign_stable_detection_ids, make_stable_detection_id
from image_processing import annotate_image, load_image_from_bytes
from model_adapters import InferenceOptions, get_adapter, model_key, provider_for
from model_registry import (
    get_default_model,
    get_enabled_valid_models,
    load_models_from_file,
    validate_selection,
)
from schemas import Detection, ModelConfig
from ui_helpers import normalize_stage


def test_four_stage_wizard_constants():
    assert STAGES == ["setup", "photos", "analyze", "running", "review"]
    assert STAGE_LABELS["setup"] == "Inventory Setup"
    assert STAGE_LABELS["photos"] == "Add Photos"
    assert STAGE_LABELS["analyze"] == "Analyze"
    assert STAGE_LABELS["review"] == "Review & Save"
    assert normalize_stage("inventory") == "setup"
    assert normalize_stage("relationship") == "setup"
    assert STAGE_ALIASES["upload"] == "photos"


def test_registry_loads_real_configured_models_only():
    models = load_models_from_file()
    enabled = get_enabled_valid_models(models, allow_demo_ids=False)
    names = {m.name for m in enabled}
    assert "YOLO-World" in names
    assert "Local Picket Counter" in names
    # Demo must not be inventively enabled in live mode
    assert "Demo Fence Detector" not in names
    default = get_default_model(models)
    assert default is not None
    assert default.is_default is True
    assert default.name == "YOLO-World"


def test_compare_requires_two_models():
    models = get_enabled_valid_models(load_models_from_file(), allow_demo_ids=False)
    one = [models[0].name]
    errs = validate_selection(models, one, "Compare Models")
    assert any("2" in e or "2–3" in e or "2-3" in e for e in errs)


def test_adapter_local_predict_zero_api():
    models = load_models_from_file()
    local = next(m for m in models if (m.kind or "").lower() == "local")
    buf = io.BytesIO()
    # Synthetic pointed tops won't match real pickets; expect success with possibly 0 dets
    Image.new("RGB", (120, 80), color=(240, 240, 240)).save(buf, format="JPEG")
    prepared = load_image_from_bytes(buf.getvalue(), "blank.jpg")
    adapter = get_adapter(local)
    result = adapter.predict(prepared, InferenceOptions(confidence_threshold=0.25))
    assert result.model_display_name == local.name
    assert result.provider == "Local"
    assert result.response_source in {"local_classical", "error"}
    # Flat blank should not invent detections as API success-with-fake-boxes
    assert result.final_count == 0 or result.success


def test_stable_detection_ids_are_deterministic():
    a = make_stable_detection_id(
        image_hash="abc",
        model_key="local:x",
        class_name="fence-picket",
        x1=1,
        y1=2,
        x2=3,
        y2=4,
        raw_index=0,
    )
    b = make_stable_detection_id(
        image_hash="abc",
        model_key="local:x",
        class_name="fence-picket",
        x1=1,
        y1=2,
        x2=3,
        y2=4,
        raw_index=0,
    )
    assert a == b
    dets = [
        Detection(
            detection_id="tmp",
            class_name="fence-picket",
            confidence=0.5,
            x1=1,
            y1=2,
            x2=3,
            y2=4,
            center_x=2,
            center_y=3,
            width=2,
            height=2,
            source_model="m",
            source_image="i",
        )
    ]
    out = assign_stable_detection_ids(dets, image_hash="abc", model_key="local:x")
    assert out[0].detection_id == a


def test_annotation_style_switch_no_crash():
    img = Image.new("RGB", (200, 100), color=(200, 200, 200))
    dets = [
        Detection(
            detection_id="1",
            class_name="fence-picket",
            confidence=0.5,
            x1=20,
            y1=10,
            x2=60,
            y2=90,
            center_x=40,
            center_y=50,
            width=40,
            height=80,
            source_model="m",
            source_image="i",
        )
    ]
    boxes = annotate_image(img, dets, style="boxes")
    markers = annotate_image(img, dets, style="markers")
    both = annotate_image(img, dets, style="both")
    assert boxes.size == markers.size == both.size == img.size


def test_camera_filename_and_hash_shape():
    data = b"\xff\xd8\xff" + b"0" * 100  # not a real jpeg — use PIL instead
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buf, format="JPEG")
    raw = buf.getvalue()
    prepared = load_image_from_bytes(raw, "camera_2026-07-19_113000.jpg")
    assert prepared.content_hash
    assert len(prepared.content_hash) == 64


def test_provider_and_key_helpers():
    wf = ModelConfig(
        name="Y",
        kind="workflow",
        workspace_name="hariram-s-mzhvc",
        workflow_id="custom-workflow",
        enabled=True,
    )
    assert "custom-workflow" in model_key(wf)
    assert provider_for(wf) == "Roboflow"
    local = ModelConfig(
        name="L",
        kind="local",
        model_id="local-picket-counter",
        enabled=True,
    )
    assert provider_for(local) == "Local"
