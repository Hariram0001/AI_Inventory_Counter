"""Offline tests for the Roboflow hosted response shape (no live API calls)."""

from __future__ import annotations

import json
from pathlib import Path

from detector import normalize_predictions, response_shape_summary

FIXTURE = Path(__file__).resolve().parents[1] / "sample_responses" / "live_roboflow_shape.json"


def test_live_shape_fixture_loads():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "predictions" in payload


def test_live_shape_summary_is_sanitized():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    summary = response_shape_summary(payload)
    assert summary["top_level_type"] == "dict"
    assert "predictions" in summary["top_level_keys"]
    assert summary["prediction_count"] == 3
    assert "class" in summary["prediction_fields"]
    assert "confidence" in summary["prediction_fields"]
    assert "fence-panel" in summary["class_names"]
    assert "pole" in summary["class_names"]
    # Ensure we did not accidentally embed full prediction blobs
    assert "x" in summary["prediction_fields"]


def test_normalize_live_shape_fields_and_clamping():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    dets = normalize_predictions(
        payload,
        source_model="Live Fixture Model",
        source_image="yard.jpg",
        image_width=800,
        image_height=600,
        confidence_threshold=0.4,
        allowed_classes=["fence-panel", "pole"],
    )
    assert len(dets) == 3
    for d in dets:
        assert d.class_name
        assert 0.0 <= d.confidence <= 1.0
        assert d.x1 <= d.x2
        assert d.y1 <= d.y2
        assert d.width == d.x2 - d.x1
        assert d.height == d.y2 - d.y1
        assert abs(d.center_x - (d.x1 + d.x2) / 2) < 1e-6
        assert abs(d.center_y - (d.y1 + d.y2) / 2) < 1e-6
        assert d.source_model == "Live Fixture Model"
        assert 0 <= d.x1 <= 800
        assert 0 <= d.x2 <= 800
        assert 0 <= d.y1 <= 600
        assert 0 <= d.y2 <= 600


def test_normalize_zero_predictions_is_valid():
    dets = normalize_predictions(
        {"predictions": [], "image": {"width": 100, "height": 100}},
        source_model="m",
        source_image="img.jpg",
        image_width=100,
        image_height=100,
    )
    assert dets == []


def test_demo_model_rejected_when_live():
    from schemas import ModelConfig

    demo = ModelConfig(
        name="Demo Fence Detector",
        kind="model",
        enabled=True,
        model_id="demo-fence-panels/1",
    )
    assert demo.is_valid(allow_demo_ids=True)
    assert not demo.is_valid(allow_demo_ids=False)
