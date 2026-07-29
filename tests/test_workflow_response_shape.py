"""Offline tests for Roboflow Workflow list-shaped responses (no live API)."""

from __future__ import annotations

import json
from pathlib import Path

from detector import normalize_predictions, response_shape_summary

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "sample_responses"
    / "live_workflow_list_shape.json"
)


def test_workflow_list_shape_summary():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    summary = response_shape_summary(payload)
    assert summary["top_level_type"] == "list"
    assert summary["prediction_count"] == 2
    assert "temporary fence panel" in summary["class_names"]
    assert "predictions" in summary["nested_output_keys"]


def test_workflow_list_normalization():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    dets = normalize_predictions(
        payload,
        source_model="YOLO-World",
        source_image="yard.jpg",
        image_width=800,
        image_height=600,
        confidence_threshold=0.25,
        allowed_classes=[],
    )
    assert len(dets) == 2
    for d in dets:
        assert d.class_name == "temporary fence panel"
        assert d.source_model == "YOLO-World"
        assert d.width > 0 and d.height > 0
        assert 0 <= d.x1 <= 800
        assert 0 <= d.x2 <= 800


def test_empty_predictions_list_is_valid_zero():
    payload = [{"predictions": {"image": {"width": 100, "height": 80}, "predictions": []}}]
    summary = response_shape_summary(payload)
    assert summary["prediction_count"] == 0
    dets = normalize_predictions(
        payload,
        source_model="YOLO-World",
        source_image="yard.jpg",
        image_width=100,
        image_height=80,
    )
    assert dets == []


def test_empty_workflow_output_is_valid_zero():
    payload = [{}]
    summary = response_shape_summary(payload)
    assert summary["prediction_count"] == 0
    dets = normalize_predictions(
        payload,
        source_model="YOLO-World",
        source_image="yard.jpg",
        image_width=800,
        image_height=600,
    )
    assert dets == []
