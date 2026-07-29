"""Tests for Roboflow response normalization."""

from __future__ import annotations

from detector import normalize_predictions


def test_converts_center_based_boxes():
    payload = {
        "predictions": [
            {
                "class": "fence-panel",
                "confidence": 0.9,
                "x": 50,
                "y": 40,
                "width": 20,
                "height": 30,
            }
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="img.jpg",
        image_width=100,
        image_height=100,
    )
    assert len(dets) == 1
    d = dets[0]
    assert d.x1 == 40
    assert d.y1 == 25
    assert d.x2 == 60
    assert d.y2 == 55
    assert d.width == 20
    assert d.height == 30


def test_clamps_coordinates():
    payload = {
        "predictions": [
            {
                "class": "fence-panel",
                "confidence": 0.9,
                "x1": -10,
                "y1": -5,
                "x2": 120,
                "y2": 150,
            }
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="img.jpg",
        image_width=100,
        image_height=100,
    )
    assert len(dets) == 1
    assert dets[0].x1 >= 0
    assert dets[0].y1 >= 0
    assert dets[0].x2 <= 100
    assert dets[0].y2 <= 100


def test_filters_malformed_and_low_confidence():
    payload = {
        "predictions": [
            {"class": "fence-panel", "confidence": 0.1, "x": 10, "y": 10, "width": 5, "height": 5},
            {"class": "fence-panel", "confidence": 0.9},  # missing box
            {"confidence": 0.9, "x": 10, "y": 10, "width": 0, "height": 0},
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="img.jpg",
        image_width=100,
        image_height=100,
        confidence_threshold=0.4,
    )
    assert dets == []


def test_filters_allowed_classes():
    payload = {
        "predictions": [
            {
                "class": "person",
                "confidence": 0.99,
                "x": 10,
                "y": 10,
                "width": 8,
                "height": 8,
            },
            {
                "class": "fence-panel",
                "confidence": 0.9,
                "x": 20,
                "y": 20,
                "width": 8,
                "height": 8,
            },
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="m",
        source_image="img.jpg",
        image_width=100,
        image_height=100,
        allowed_classes=["fence-panel"],
    )
    assert len(dets) == 1
    assert dets[0].class_name == "fence-panel"


def test_nested_workflow_output():
    payload = {
        "outputs": [
            {
                "predictions": {
                    "predictions": [
                        {
                            "class_name": "pole",
                            "confidence": 0.8,
                            "x": 15,
                            "y": 20,
                            "width": 10,
                            "height": 40,
                        }
                    ]
                }
            }
        ]
    }
    dets = normalize_predictions(
        payload,
        source_model="workflow",
        source_image="img.jpg",
        image_width=100,
        image_height=100,
    )
    assert len(dets) == 1
    assert dets[0].class_name == "pole"
