"""Offline tests for local OpenCV shape detection (no network)."""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
import pytest
from PIL import Image

from shape_detection import (
    MSG_TOO_MANY,
    circularity,
    detect_circles,
    detect_contour_candidates,
    detect_hough_candidates,
    generate_synthetic_circle_sample,
    merge_candidates,
    run_shape_detection,
    sanitize_upload_filename,
    validate_shape_image_bytes,
    ShapeDetectionError,
    _RawCandidate,
    _scale_for_processing,
)
from shape_detection_models import ShapeDetectionSettings, hash_image_bytes
from shape_registry import (
    UNSUPPORTED_SHAPE_MESSAGE,
    ShapeResolutionError,
    coming_soon_shapes,
    resolve_shape,
)


def _blank(w=200, h=200):
    return np.full((h, w, 3), 240, dtype=np.uint8)


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


def test_registry_circle_aliases():
    for term in [
        "circle",
        "Circles",
        "CIRCULAR OBJECT",
        "circular objects",
        "round object",
        "round objects",
    ]:
        assert resolve_shape(term).key == "circle"
        assert resolve_shape(term).enabled


def test_registry_unsupported_and_coming_soon():
    with pytest.raises(ShapeResolutionError, match="testing phase"):
        resolve_shape("triangle")
    with pytest.raises(ShapeResolutionError, match="testing phase"):
        resolve_shape("hexagon")
    soon = {s.key for s in coming_soon_shapes()}
    assert "triangle" in soon
    assert "rectangle" in soon
    assert UNSUPPORTED_SHAPE_MESSAGE


def test_blank_image_zero_circles():
    result = run_shape_detection(_encode(_blank()), requested_shape="circles")
    assert result.detected_count == 0
    assert result.final_count == 0


def test_one_filled_circle():
    img = _blank(320, 320)
    cv2.circle(img, (160, 160), 50, (20, 20, 20), -1)
    result = run_shape_detection(_encode(img), requested_shape="circle")
    assert result.included_count >= 1


def test_one_outlined_circle():
    img = _blank(320, 320)
    cv2.circle(img, (160, 160), 55, (10, 10, 10), 3)
    settings = ShapeDetectionSettings(target_type="drawn_outlined", mode="sensitive")
    result = run_shape_detection(
        _encode(img), requested_shape="circles", settings=settings
    )
    assert result.included_count >= 1


def test_several_separated_and_radii():
    data, expected = generate_synthetic_circle_sample()
    result = run_shape_detection(
        data,
        requested_shape="circular objects",
        settings=ShapeDetectionSettings(mode="balanced", target_type="both"),
    )
    # Synthetic sample should recover the known circles with little FP noise.
    assert result.included_count >= max(1, expected - 2)
    assert result.included_count <= expected + 2


def test_noisy_background_still_runs():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(240, 240, 3), dtype=np.uint8)
    cv2.circle(img, (120, 120), 40, (255, 255, 255), -1)
    result = run_shape_detection(_encode(img), requested_shape="circle")
    assert result.processing_time_seconds < 10
    assert isinstance(result.detections, list)


def test_partial_edge_circle_flagged():
    img = _blank(300, 300)
    cv2.circle(img, (10, 150), 40, (0, 0, 0), -1)
    settings = ShapeDetectionSettings(include_partial=True, mode="sensitive")
    result = run_shape_detection(
        _encode(img), requested_shape="circles", settings=settings
    )
    if result.detections:
        assert any(d.partial for d in result.detections)


def test_concentric_dedup_default_and_separate():
    img = _blank(400, 400)
    cv2.circle(img, (200, 200), 80, (0, 0, 0), 3)
    cv2.circle(img, (200, 200), 40, (0, 0, 0), 3)
    off = ShapeDetectionSettings(
        count_concentric_separately=False,
        target_type="drawn_outlined",
        mode="sensitive",
    )
    on = ShapeDetectionSettings(
        count_concentric_separately=True,
        target_type="drawn_outlined",
        mode="sensitive",
    )
    r_off = run_shape_detection(_encode(img), requested_shape="circles", settings=off)
    r_on = run_shape_detection(_encode(img), requested_shape="circles", settings=on)
    assert r_on.included_count >= r_off.included_count


def test_rectangle_and_ellipse_rejected():
    img = _blank(300, 300)
    cv2.rectangle(img, (40, 80), (260, 160), (0, 0, 0), -1)
    result = run_shape_detection(_encode(img), requested_shape="circle")
    # A wide rectangle should not become a confident circle count of many.
    assert result.included_count <= 1

    img2 = _blank(300, 300)
    cv2.ellipse(img2, (150, 150), (90, 30), 0, 0, 360, (0, 0, 0), -1)
    result2 = run_shape_detection(_encode(img2), requested_shape="circle")
    assert result2.included_count <= 1


def test_circularity_formula():
    # Perfect circle: area = πr², peri = 2πr → circularity ≈ 1
    r = 10.0
    area = math.pi * r * r
    peri = 2 * math.pi * r
    assert circularity(area, peri) == pytest.approx(1.0, abs=1e-6)
    assert circularity(0, 10) == 0.0


def test_hough_and_contour_and_merge():
    img = _blank(280, 280)
    cv2.circle(img, (140, 140), 45, (0, 0, 0), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    hough = detect_hough_candidates(
        gray, min_radius=20, max_radius=80, min_dist=20, param1=100, param2=20
    )
    contour = detect_contour_candidates(
        gray,
        min_area=200,
        max_area=50000,
        min_radius=20,
        max_radius=80,
        circularity_threshold=0.6,
        outlined=False,
    )
    assert isinstance(hough, list)
    assert isinstance(contour, list)
    merged = merge_candidates(
        hough + contour, count_concentric_separately=False, include_partial=True
    )
    assert len(merged) <= len(hough) + len(contour)


def test_coordinate_mapping_after_downscale():
    img = np.full((2400, 2400, 3), 245, dtype=np.uint8)
    cv2.circle(img, (1200, 1200), 200, (0, 0, 0), -1)
    processed, scale = _scale_for_processing(img, max_dim=800)
    assert scale < 1.0
    dets, meta = detect_circles(
        img, ShapeDetectionSettings(mode="balanced", target_type="circular_objects")
    )
    assert meta["processed_width"] <= 1600
    if dets:
        # Centers should be near original coordinates, not processed ones
        assert abs(dets[0].center_x - 1200) < 250


def test_unsupported_shape_does_not_run_detection():
    with pytest.raises(ShapeDetectionError, match="testing phase"):
        run_shape_detection(_encode(_blank()), requested_shape="triangle")


def test_corrupt_and_filename_sanitization():
    with pytest.raises(ShapeDetectionError):
        validate_shape_image_bytes(b"not-an-image")
    assert sanitize_upload_filename("My Photo.JPG") == "my_photo"
    with pytest.raises(ShapeDetectionError):
        sanitize_upload_filename("../secret.png")


def test_final_count_and_review_helpers():
    from shape_detection_models import CircleDetection, BoundingBox

    dets = [
        CircleDetection(
            id="shape-1",
            included=True,
            sequence_number=1,
            bounding_box=BoundingBox(0, 0, 1, 1),
        ),
        CircleDetection(
            id="shape-2",
            included=False,
            review_status="false_positive",
            sequence_number=2,
            bounding_box=BoundingBox(0, 0, 1, 1),
        ),
    ]
    # compute_final_count lives on shape_detection module
    from shape_detection import compute_final_count as cfc

    counts = cfc(dets, manually_added=2)
    assert counts["included"] == 1
    assert counts["excluded"] == 1
    assert counts["final"] == 3


def test_webp_and_png_validation():
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    meta = validate_shape_image_bytes(buf.getvalue())
    assert meta["format"] == "PNG"
    assert hash_image_bytes(buf.getvalue())


def test_synthetic_sample_deterministic_count():
    data, expected = generate_synthetic_circle_sample(expected_count=8)
    assert expected == 8
    meta = validate_shape_image_bytes(data)
    assert meta["width"] == 640
