"""Tests for photo preparation geometry, filtering, and persistence helpers."""

from __future__ import annotations

import inspect
import io
from copy import deepcopy

import pytest
from PIL import Image

import app as app_module
from image_quality import assess_image_bytes
from photo_preparation import (
    MASK_MODE_COUNT_FILTER,
    MASK_MODE_HIDE_FROM_AI,
    OVERLAP_RULE_MIN_IOU,
    CropRect,
    PhotoPreparation,
    apply_preparation_to_detections,
    aspects_compatible,
    build_prepared_image,
    create_polygon_region,
    create_rectangle_region,
    default_preparation,
    eligible_area_ratio,
    evaluate_detection_against_preparation,
    mask_excluded_for_inference,
    open_source_image,
    persistable_preparation,
    point_eligible,
    preparation_status,
    render_count_area_preview,
)
from schemas import Detection


def _png_bytes(w: int = 200, h: int = 100, color=(120, 140, 100)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _det(cx: float, cy: float, *, det_id: str = "d1", size: float = 10) -> Detection:
    return Detection(
        detection_id=det_id,
        class_name="fence-panel",
        confidence=0.9,
        x1=cx - size,
        y1=cy - size,
        x2=cx + size,
        y2=cy + size,
        center_x=cx,
        center_y=cy,
        width=size * 2,
        height=size * 2,
        source_model="test",
        source_image="t.png",
    )


# --- Photos stage (upload → continue; no prep workspace) ----------------------


def test_stage_photos_is_upload_and_continue():
    src = inspect.getsource(app_module.stage_photos)
    assert "Continue to Analyze" in src
    assert "Preview & Prepare" not in src
    assert "render_photo_cards" not in src
    assert "render_preparation_workspace" not in src


def test_no_preparation_required_to_continue():
    src = inspect.getsource(app_module.stage_photos)
    assert "can_next = len(st.session_state.uploaded_images) >= 1" in src


# --- Regions / geometry -------------------------------------------------------


def test_rectangle_include_and_exclude():
    prep = default_preparation("img1", width=100, height=100)
    prep.include_regions = [create_rectangle_region("include", 0.0, 0.0, 0.5, 1.0)]
    prep.exclude_regions = [create_rectangle_region("exclude", 0.1, 0.1, 0.3, 0.3)]
    assert point_eligible(0.2, 0.5, prep)[0] is True
    assert point_eligible(0.2, 0.2, prep)[0] is False  # exclude wins
    assert point_eligible(0.8, 0.5, prep)[0] is False  # outside include


def test_multiple_includes_union():
    prep = default_preparation("img1")
    prep.include_regions = [
        create_rectangle_region("include", 0.0, 0.0, 0.2, 0.2),
        create_rectangle_region("include", 0.8, 0.8, 1.0, 1.0),
    ]
    assert point_eligible(0.1, 0.1, prep)[0]
    assert point_eligible(0.9, 0.9, prep)[0]
    assert not point_eligible(0.5, 0.5, prep)[0]


def test_full_image_default_without_includes():
    prep = default_preparation("img1")
    prep.exclude_regions = [create_rectangle_region("exclude", 0.0, 0.0, 0.2, 0.2)]
    assert point_eligible(0.5, 0.5, prep)[0]
    assert not point_eligible(0.1, 0.1, prep)[0]


def test_polygon_and_brush_serialization():
    poly = create_polygon_region(
        "include",
        [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)],
        shape_type="polygon",
    )
    brush = create_polygon_region(
        "exclude",
        [(0.2, 0.2), (0.3, 0.25), (0.4, 0.2), (0.3, 0.15)],
        shape_type="brush",
    )
    d1, d2 = poly.to_dict(), brush.to_dict()
    assert d1["shape_type"] == "polygon"
    assert d2["shape_type"] == "brush"
    assert len(d1["points_normalized"]) == 3
    restored = PhotoPreparation.from_dict(
        {
            "image_id": "x",
            "include_regions": [d1],
            "exclude_regions": [d2],
        }
    )
    assert restored.include_regions[0].shape_type == "polygon"
    assert restored.exclude_regions[0].shape_type == "brush"


def test_undo_delete_clear():
    prep = default_preparation("img1")
    prep.push_undo()
    prep.include_regions = [create_rectangle_region("include", 0, 0, 1, 1)]
    prep.push_undo()
    prep.include_regions = []
    assert prep.undo()
    assert len(prep.include_regions) == 1
    assert prep.redo()
    assert prep.include_regions == []
    prep.exclude_regions = [create_rectangle_region("exclude", 0, 0, 0.1, 0.1)]
    prep.push_undo()
    prep.exclude_regions = []
    assert prep.undo()
    assert len(prep.exclude_regions) == 1


def test_normalized_coords_persist_across_resize_display():
    prep = default_preparation("img1")
    prep.include_regions = [create_rectangle_region("include", 0.25, 0.25, 0.75, 0.75)]
    blob = prep.to_dict()
    # Simulate different display size — eligibility uses normalized space
    p2 = PhotoPreparation.from_dict(blob)
    assert point_eligible(0.5, 0.5, p2)[0]
    assert not point_eligible(0.1, 0.1, p2)[0]


def test_detection_center_rules():
    prep = default_preparation("img", width=100, height=100)
    prep.include_regions = [create_rectangle_region("include", 0.0, 0.0, 0.5, 1.0)]
    inside = evaluate_detection_against_preparation(_det(25, 50), prep, 100, 100)
    outside = evaluate_detection_against_preparation(_det(75, 50), prep, 100, 100)
    assert inside["excluded_by_region"] is False
    assert outside["excluded_by_region"] is True
    prep.exclude_regions = [create_rectangle_region("exclude", 0.0, 0.0, 0.4, 1.0)]
    excluded = evaluate_detection_against_preparation(_det(25, 50), prep, 100, 100)
    assert excluded["excluded_by_region"] is True
    assert excluded["inside_exclude_area"] is True


def test_overlap_rule_minimum():
    prep = default_preparation("img", width=100, height=100)
    prep.overlap_rule = OVERLAP_RULE_MIN_IOU
    prep.minimum_detection_overlap = 0.5
    prep.include_regions = [create_rectangle_region("include", 0.0, 0.0, 0.5, 1.0)]
    # Box mostly outside include
    det = _det(55, 50, size=20)
    meta = evaluate_detection_against_preparation(det, prep, 100, 100)
    assert "region_overlap_ratio" in meta


def test_crop_and_rotation_transform():
    data = _png_bytes(200, 100)
    prep = default_preparation("img", width=200, height=100)
    prep.rotation = 90
    rotated = build_prepared_image(data, prep)
    assert rotated.size == (100, 200)
    prep.rotation = 0
    prep.crop = CropRect(0.25, 0.0, 0.75, 1.0)
    cropped = build_prepared_image(data, prep)
    assert cropped.size[0] == 100
    assert cropped.size[1] == 100


def test_original_image_unchanged_by_masking():
    img = Image.new("RGB", (80, 80), (20, 180, 40))
    # Non-uniform so neutral fill differs from source pixels
    for x in range(40):
        for y in range(40):
            img.putpixel((x, y), (240, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    prep = default_preparation("img")
    prep.exclude_regions = [create_rectangle_region("exclude", 0.0, 0.0, 0.5, 0.5)]
    prep.mask_mode = MASK_MODE_HIDE_FROM_AI
    src = open_source_image(data)
    before = src.tobytes()
    masked = mask_excluded_for_inference(src, prep)
    assert src.tobytes() == before
    assert masked.tobytes() != before


def test_count_area_preview_not_black_canvas():
    data = _png_bytes(120, 80, color=(200, 210, 190))
    prep = default_preparation("img")
    prep.include_regions = [create_rectangle_region("include", 0.1, 0.1, 0.9, 0.9)]
    img = build_prepared_image(data, prep)
    preview = render_count_area_preview(img, prep)
    # Mean luminance should not be near-black
    import numpy as np

    mean = float(np.asarray(preview).mean())
    assert mean > 40


def test_apply_preparation_keeps_excluded_in_list():
    prep = default_preparation("img", width=100, height=100)
    prep.include_regions = [create_rectangle_region("include", 0.0, 0.0, 0.5, 1.0)]
    dets = [_det(25, 50, det_id="in"), _det(75, 50, det_id="out")]
    all_dets, counted, excl_n = apply_preparation_to_detections(dets, prep, 100, 100)
    assert len(all_dets) == 2
    assert len(counted) == 1
    assert excl_n == 1
    out = next(d for d in all_dets if d.detection_id == "out")
    assert out.excluded_by_region is True
    assert out.included_in_count is False


def test_default_mask_mode_is_count_filter():
    prep = default_preparation("img")
    assert prep.mask_mode == MASK_MODE_COUNT_FILTER


def test_persistable_and_old_record_compat():
    prep = default_preparation("img1", width=10, height=10)
    prep.include_regions = [create_rectangle_region("include", 0, 0, 1, 1)]
    saved = persistable_preparation(
        prep, source="upload", original_hash="abc", prepared_hash="def"
    )
    assert "include_regions" in saved
    assert saved["source"] == "upload"
    # Old record: missing prep → full image eligible
    empty = PhotoPreparation.from_dict(None, image_id="old")
    assert preparation_status(empty) == "Not reviewed"
    assert point_eligible(0.5, 0.5, empty)[0]


def test_status_labels():
    prep = default_preparation("x")
    assert preparation_status(prep) == "Not reviewed"
    prep.exclude_regions = [create_rectangle_region("exclude", 0, 0, 0.2, 0.2)]
    assert preparation_status(prep) == "Ready"
    prep2 = default_preparation("y")
    prep2.crop = CropRect(0.1, 0.1, 0.9, 0.9)
    assert preparation_status(prep2) == "Cropped"


def test_quality_checks_local():
    tiny = _png_bytes(20, 20)
    result = assess_image_bytes(tiny, "tiny.png")
    assert result["blocking"] is True
    ok = assess_image_bytes(_png_bytes(400, 300), "ok.png")
    assert ok["ok"] is True


def test_aspects_compatible():
    assert aspects_compatible(100, 50, 200, 100)
    assert not aspects_compatible(100, 50, 100, 100)


def test_eligible_area_ratio():
    prep = default_preparation("x")
    assert eligible_area_ratio(prep) == pytest.approx(1.0, abs=0.02)
    prep.include_regions = [create_rectangle_region("include", 0, 0, 0.5, 1)]
    assert 0.4 < eligible_area_ratio(prep) < 0.6


def test_app_has_no_photo_preparation_wiring():
    src_app = open(app_module.__file__, encoding="utf-8").read()
    assert "photo_preparation_by_image_id" not in src_app
    assert "_apply_prep_to_result" not in src_app
    assert "review_show_region_excluded" not in src_app
    assert "review_show_count_areas" not in src_app
    assert "Preview & Prepare" not in src_app


def test_multi_image_independent_state_keys():
    # Session-style dict keyed by image id
    a = default_preparation("aaa")
    b = default_preparation("bbb")
    a.include_regions = [create_rectangle_region("include", 0, 0, 0.5, 0.5)]
    store = {"aaa": a.to_dict(), "bbb": b.to_dict()}
    assert store["aaa"]["include_regions"]
    assert not store["bbb"]["include_regions"]


def test_detection_dataclass_region_fields_roundtrip():
    d = _det(10, 10)
    d.excluded_by_region = True
    d.region_exclusion_reason = "test"
    d.inside_include_area = False
    clone = Detection(**d.to_dict())
    assert clone.excluded_by_region is True
    assert clone.region_exclusion_reason == "test"
