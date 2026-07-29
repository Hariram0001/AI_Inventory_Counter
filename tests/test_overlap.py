"""Tests for overlap metrics and deduplication strategies."""

from __future__ import annotations

import uuid

from schemas import Detection
from overlap import (
    apply_conservative_dedup,
    apply_nmm,
    apply_nms,
    aspect_ratio_similarity,
    ios,
    iou,
    normalized_center_distance,
)


def _det(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    conf: float = 0.9,
    cls: str = "fence-panel",
    tile_id: str | None = None,
    scale_id: str | None = None,
    model: str = "m1",
) -> Detection:
    w = x2 - x1
    h = y2 - y1
    return Detection(
        detection_id=str(uuid.uuid4()),
        class_name=cls,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=(x1 + x2) / 2,
        center_y=(y1 + y2) / 2,
        width=w,
        height=h,
        source_model=model,
        source_image="test.jpg",
        tile_id=tile_id,
        scale_id=scale_id,
    )


def test_iou_no_intersection():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert iou(a, b) == 0.0


def test_iou_partial_intersection():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    val = iou(a, b)
    assert 0.14 < val < 0.15


def test_iou_identical_boxes():
    a = (0, 0, 10, 10)
    assert iou(a, a) == 1.0


def test_ios_contained_boxes():
    outer = (0, 0, 100, 100)
    inner = (25, 25, 75, 75)
    assert ios(outer, inner) == 1.0  # intersection == smaller


def test_iou_zero_area_boxes():
    a = (0, 0, 0, 10)
    b = (0, 0, 10, 10)
    assert iou(a, b) == 0.0


def test_nms_removes_lower_confidence_duplicate():
    high = _det(0, 0, 10, 10, conf=0.95, tile_id="t1")
    low = _det(1, 1, 11, 11, conf=0.5, tile_id="t2")
    kept = apply_nms([high, low], iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0].confidence == 0.95


def test_nms_preserves_non_overlapping():
    a = _det(0, 0, 10, 10, conf=0.9)
    b = _det(50, 50, 60, 60, conf=0.8)
    kept = apply_nms([a, b], iou_threshold=0.5)
    assert len(kept) == 2


def test_nms_preserves_different_classes():
    a = _det(0, 0, 10, 10, conf=0.9, cls="fence-panel")
    b = _det(1, 1, 11, 11, conf=0.8, cls="pole")
    kept = apply_nms([a, b], iou_threshold=0.5)
    assert len(kept) == 2


def test_conservative_preserves_distinct_centers():
    a = _det(0, 0, 40, 100, conf=0.9, tile_id="t1")
    # Shifted enough to be a separate panel
    b = _det(30, 0, 70, 100, conf=0.88, tile_id="t2")
    kept = apply_conservative_dedup([a, b], iou_threshold=0.5)
    assert len(kept) == 2


def test_nmm_merges_true_duplicates():
    a = _det(0, 0, 10, 10, conf=0.9, tile_id="t1")
    b = _det(0.5, 0.5, 10.5, 10.5, conf=0.8, tile_id="t2")
    merged = apply_nmm([a, b], iou_threshold=0.5)
    assert len(merged) == 1
    m = merged[0]
    assert m.x2 > m.x1
    assert m.y2 > m.y1
    assert m.merged_from
    assert m.confidence >= 0.8


def test_nmm_retains_source_metadata():
    a = _det(0, 0, 10, 10, conf=0.9, model="A", tile_id="t1")
    b = _det(1, 1, 11, 11, conf=0.85, model="B", tile_id="t2")
    merged = apply_nmm([a, b], iou_threshold=0.5)
    assert len(merged) == 1
    assert "A" in merged[0].source_model or "A" in merged[0].contributing_models


def test_center_distance_and_aspect():
    a = _det(0, 0, 10, 10)
    b = _det(0, 0, 10, 10)
    assert normalized_center_distance(a, b) == 0.0
    assert aspect_ratio_similarity(a, b) == 1.0
