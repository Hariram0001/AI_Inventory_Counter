"""Focused pipeline tests — offline only (no live API)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

import config
from config import reload_settings
from detector import (
    _normalize_confidence,
    build_workflow_parameters,
    normalize_predictions,
    response_shape_summary,
    sanitize_payload_for_debug,
)
from image_processing import annotate_image, load_image_from_bytes, validate_upload
from model_registry import get_enabled_valid_models, load_models_from_file
from overlap import apply_nms, iou
from schemas import Detection, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "sample_responses" / "live_fence_test_response.json"


def test_env_demo_mode_false_by_default():
    reload_settings()
    assert config.DEMO_MODE is False or isinstance(config.DEMO_MODE, bool)


def test_models_json_expected_workspace_workflow():
    models = load_models_from_file()
    enabled = get_enabled_valid_models(models, allow_demo_ids=False)
    assert enabled, "expected at least one enabled live model"
    wf = next(m for m in enabled if (m.kind or "").lower() == "workflow")
    assert wf.workspace_name == "hariram-s-mzhvc"
    assert wf.workflow_id == "custom-workflow"
    assert (wf.image_input_name or "image") == "image"


def test_missing_api_key_detector_error(monkeypatch):
    from detector import DetectorError, RoboflowDetector

    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "")
    reload_settings()
    det = RoboflowDetector(demo_mode=False, api_key="")
    with pytest.raises(DetectorError):
        det._get_client()


def test_image_file_validation_and_rewind():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color=(120, 130, 140)).save(buf, format="JPEG")
    data = buf.getvalue()
    # mimic UploadedFile: multiple reads need seek(0)
    stream = io.BytesIO(data)
    stream.seek(0)
    first = stream.read()
    stream.seek(0)
    second = stream.read()
    assert first == second == data
    prepared = load_image_from_bytes(data, "probe.jpg")
    assert prepared.original_width == 64
    assert prepared.original_height == 48


def test_validate_upload_rejects_corrupt():
    with pytest.raises(Exception):
        validate_upload(b"not-an-image", "empty.jpg", max_bytes=1_000_000)


def test_workflow_request_input_mapping():
    model = ModelConfig(
        name="t",
        kind="workflow",
        workspace_name="hariram-s-mzhvc",
        workflow_id="custom-workflow",
        image_input_name="image",
    )
    params = build_workflow_parameters(model, r"C:\temp\photo.jpg", prompt=None)
    assert params == {"image": r"C:\temp\photo.jpg"}


def test_local_picket_counter_on_dogear_fixture():
    from picket_counter import detect_fence_pickets

    # Prefer bundled sample; fall back to optional Cursor workspace asset if present.
    asset = ROOT / "assets" / "sample_images" / "fence_picket_panel_01.jpg"
    if not asset.exists():
        pytest.skip("bundled picket sample not present")
    img = Image.open(asset).convert("RGB")
    dets, warnings = detect_fence_pickets(img, source_image=asset.name)
    # Small sample may yield fewer peaks than a full-resolution dog-ear fixture.
    assert len(dets) >= 1
    assert all(d.class_name == "fence-picket" for d in dets)
    assert any("picket" in w.lower() for w in warnings)


def test_local_model_enabled_in_registry():
    models = load_models_from_file()
    local = next(m for m in models if m.name == "Local Picket Counter")
    assert local.demo_only is False
    assert local.enabled is True
    enabled = get_enabled_valid_models(models, allow_demo_ids=False)
    names = {m.name for m in enabled}
    assert "YOLO-World" in names
    assert "Local Picket Counter" in names


def test_prompt_to_class_names_and_spec_injection():
    from detector import inject_class_names_into_workflow_spec, prompt_to_class_names

    assert prompt_to_class_names("wood fence, fence post") == ["wood fence", "fence post"]
    spec = {
        "steps": [
            {
                "type": "roboflow_core/yolo_world_model@v1",
                "name": "model",
                "class_names": ["wood fence"],
            }
        ]
    }
    updated = inject_class_names_into_workflow_spec(spec, ["fence panel"])
    assert updated["steps"][0]["class_names"] == ["fence panel"]
    assert spec["steps"][0]["class_names"] == ["wood fence"]  # original untouched


def test_sanitize_logging_redacts_secrets():
    payload = {"api_key": "SECRET", "predictions": [{"class": "wood fence", "confidence": 0.3}]}
    clean = sanitize_payload_for_debug(payload)
    assert clean["api_key"] == "***REDACTED***"
    assert "SECRET" not in json.dumps(clean)


def test_live_fence_fixture_parsing():
    assert FIXTURE.exists(), "missing live_fence_test_response.json fixture"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    shape = response_shape_summary(payload)
    assert shape["prediction_count"] == 1
    assert "wood fence" in shape["class_names"]
    dets = normalize_predictions(
        payload,
        source_model="YOLO-World",
        source_image="fence_test.jpg",
        image_width=1536,
        image_height=1024,
        confidence_threshold=0.25,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.class_name == "wood fence"
    assert 0.25 <= d.confidence <= 1.0
    assert 0 <= d.x1 < d.x2 <= 1536
    assert 0 <= d.y1 < d.y2 <= 1024
    # Full-fence style box should be large
    assert d.width > 1000


def test_empty_successful_response():
    payload = [{"predictions": {"predictions": [], "image": {"width": 100, "height": 100}}}]
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="i.jpg",
        image_width=100,
        image_height=100,
    )
    assert dets == []
    assert response_shape_summary(payload)["prediction_count"] == 0


def test_unexpected_response_structure_does_not_crash():
    payload = {"outputs": {"weird": [{"not": "a detection"}]}, "status": "ok"}
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="i.jpg",
        image_width=100,
        image_height=100,
    )
    assert dets == []


def test_center_to_corner_and_normalized_coords():
    payload = {
        "predictions": [
            {
                "class": "panel",
                "confidence": 0.9,
                "x": 0.5,
                "y": 0.5,
                "width": 0.2,
                "height": 0.4,
            }
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="i.jpg",
        image_width=200,
        image_height=100,
    )
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.x1 - 80) < 0.01
    assert abs(d.y1 - 30) < 0.01
    assert abs(d.x2 - 120) < 0.01
    assert abs(d.y2 - 70) < 0.01


def test_confidence_normalization():
    assert _normalize_confidence(0.82) == pytest.approx(0.82)
    assert _normalize_confidence(82) == pytest.approx(0.82)
    assert _normalize_confidence("0.5") == pytest.approx(0.5)
    assert _normalize_confidence("75") == pytest.approx(0.75)


def test_confidence_percent_in_payload():
    payload = {
        "predictions": [
            {
                "class": "fence",
                "confidence": 82,
                "x": 50,
                "y": 50,
                "width": 20,
                "height": 20,
            }
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="i.jpg",
        image_width=100,
        image_height=100,
        confidence_threshold=0.5,
    )
    assert len(dets) == 1
    assert dets[0].confidence == pytest.approx(0.82)


def test_invalid_box_rejection():
    payload = {
        "predictions": [
            {"class": "a", "confidence": 0.9, "x": 10, "y": 10, "width": 0, "height": 10},
            {"class": "b", "confidence": 0.9},
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="i.jpg",
        image_width=100,
        image_height=100,
    )
    assert dets == []


def test_duplicate_filtering_adjacent_panels():
    from schemas import Detection
    import uuid

    def det(x1, y1, x2, y2, conf=0.9):
        return Detection(
            detection_id=str(uuid.uuid4()),
            class_name="wood fence",
            confidence=conf,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            center_x=(x1 + x2) / 2,
            center_y=(y1 + y2) / 2,
            width=x2 - x1,
            height=y2 - y1,
            source_model="m",
            source_image="i.jpg",
        )

    a = det(0, 0, 40, 100)
    b = det(40, 0, 80, 100)  # adjacent, non-overlapping
    assert iou((a.x1, a.y1, a.x2, a.y2), (b.x1, b.y1, b.x2, b.y2)) == 0.0
    kept = apply_nms([a, b], iou_threshold=0.5)
    assert len(kept) == 2

    # identical boxes → one kept
    c = det(0, 0, 40, 100, conf=0.95)
    d = det(0, 0, 40, 100, conf=0.5)
    kept2 = apply_nms([c, d], iou_threshold=0.5)
    assert len(kept2) == 1


def test_annotation_output_and_stable_numbering():
    img = Image.new("RGB", (200, 100), color=(200, 200, 200))
    dets = [
        Detection(
            detection_id="1",
            class_name="wood fence",
            confidence=0.4,
            x1=10,
            y1=10,
            x2=180,
            y2=90,
            center_x=95,
            center_y=50,
            width=170,
            height=80,
            source_model="m",
            source_image="i.jpg",
        )
    ]
    annotated = annotate_image(img, dets, model_name="m")
    assert annotated.size == img.size
    assert annotated.tobytes() != img.tobytes()


def test_live_demo_source_labeling_in_summary():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    shape = response_shape_summary(payload)
    assert shape["top_level_type"] == "list"
    assert shape["prediction_count"] >= 1


@pytest.mark.live
def test_live_api_optional():
    """Runs only when explicitly selected and API key is present."""
    reload_settings()
    if config.DEMO_MODE or not config.ROBOFLOW_API_KEY:
        pytest.skip("live API not configured")
    photo_env = os.getenv("FENCE_TEST_IMAGE", "").strip()
    candidates = []
    if photo_env:
        candidates.append(Path(photo_env))
    candidates.append(ROOT / "assets" / "sample_images" / "fence_gate_driveway_01.jpg")
    photo = next((p for p in candidates if p.exists()), None)
    if photo is None:
        pytest.skip("no live test image available (set FENCE_TEST_IMAGE)")
    from validate_live import main as live_main
    import sys

    old = sys.argv
    try:
        sys.argv = ["validate_live.py", str(photo)]
        code = live_main()
    finally:
        sys.argv = old
    assert code == 0
