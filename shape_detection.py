"""Local OpenCV circle detection — no network, no paid APIs.

Pipeline inspired by Count Things–style counting apps:
HoughCircles for edge-sensitive rings/objects + contour circularity for
filled/outlined blobs, then circle-aware duplicate suppression.
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import config
from shape_detection_models import (
    MAX_CANDIDATES_BEFORE_DEDUP,
    MAX_FINAL_DETECTIONS,
    MAX_PROCESS_DIMENSION,
    MAX_SOURCE_DIMENSION,
    BoundingBox,
    CircleDetection,
    ShapeDetectionResult,
    ShapeDetectionSettings,
    apply_mode_presets,
    balanced_defaults,
    hash_image_bytes,
)
from shape_registry import ShapeResolutionError, resolve_shape

PROGRESS_STEPS = (
    "Preparing image",
    "Finding circular boundaries",
    "Evaluating circular shapes",
    "Removing duplicate detections",
    "Preparing results",
)

EXPERIMENTAL_NOTICE = (
    "Shape detection is experimental. Reflections, shadows, repeated textures "
    "and curved object parts may produce false detections."
)

MSG_NO_IMAGE = "Upload an image, use the camera, or choose a test sample."
MSG_NO_CIRCLES = (
    "Detection completed successfully, but no likely shapes were found. "
    "Try Sensitive mode or adjust the size range."
)
MSG_NO_SHAPES = MSG_NO_CIRCLES
MSG_TOO_MANY = (
    "The image produced too many possible circles. "
    "Try Strict mode or increase the minimum circle size."
)
MSG_CORRUPT = "This image could not be decoded."
MSG_INTERNAL = (
    "Circle detection could not process this image. "
    "Open Technical Details for the sanitized error."
)


class ShapeDetectionError(Exception):
    """User-facing detection failure with a safe primary message."""

    def __init__(self, message: str, *, technical: str = "") -> None:
        super().__init__(message)
        self.technical = technical


@dataclass
class _RawCandidate:
    cx: float
    cy: float
    radius: float
    method: str
    quality: float
    partial: bool = False


def circularity(area: float, perimeter: float) -> float:
    """4π·area / perimeter² — geometric quality, not model confidence."""
    if perimeter <= 1e-6 or area <= 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def validate_shape_image_bytes(data: bytes) -> dict[str, Any]:
    """Decode and validate an upload for shape detection."""
    if not data:
        raise ShapeDetectionError(MSG_NO_IMAGE)
    max_bytes = int(getattr(config, "MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
    if len(data) > max_bytes:
        raise ShapeDetectionError(
            f"Images must be {max_bytes // (1024 * 1024)} MB or smaller."
        )

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fmt = (image.format or "").upper()
            width, height = image.size
            mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ShapeDetectionError(MSG_CORRUPT) from exc

    allowed = {"JPEG", "JPG", "PNG", "WEBP"}
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in allowed and fmt != "JPEG":
        # Pillow reports JPEG not JPG
        if fmt not in {"JPEG", "PNG", "WEBP"}:
            raise ShapeDetectionError(
                "Supported formats are JPG, JPEG, PNG, and WEBP."
            )
    if width < 16 or height < 16:
        raise ShapeDetectionError("Image is too small to analyze.")
    if width > MAX_SOURCE_DIMENSION or height > MAX_SOURCE_DIMENSION:
        raise ShapeDetectionError(
            f"Images must be at most {MAX_SOURCE_DIMENSION} pixels on either side."
        )
    mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(fmt, "application/octet-stream")
    return {
        "format": fmt,
        "width": width,
        "height": height,
        "size_bytes": len(data),
        "mime_type": mime,
        "mode": mode,
        "hash": hash_image_bytes(data),
    }


def decode_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ShapeDetectionError(MSG_CORRUPT)
    return image


def generate_synthetic_circle_sample(
    *,
    width: int = 640,
    height: int = 480,
    expected_count: int = 8,
) -> tuple[bytes, int]:
    """Local synthetic sample — filled + outlined circles, known count."""
    # Clean background (deterministic). Mild grain is added after drawing so
    # edges stay sharp enough for Hough/contour without inventing circles.
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    specs = [
        ((90, 90), 40, True),
        ((220, 100), 28, True),
        ((360, 110), 50, False),
        ((500, 95), 35, True),
        ((120, 280), 45, False),
        ((280, 300), 32, True),
        ((430, 290), 55, False),
        ((560, 320), 25, True),
    ][:expected_count]

    for (cx, cy), r, filled in specs:
        color = (40, 90, 200) if filled else (30, 30, 30)
        thickness = -1 if filled else 3
        cv2.circle(img, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
        if not filled:
            cv2.circle(
                img, (cx, cy), max(1, r - 8), (245, 245, 245), -1, lineType=cv2.LINE_AA
            )

    grain = np.random.default_rng(42).integers(0, 8, size=img.shape, dtype=np.uint8)
    img = cv2.subtract(img, grain)

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ShapeDetectionError("Could not generate the built-in test sample.")
    return bytes(buf), len(specs)


def _scale_for_processing(
    image: np.ndarray, max_dim: int = MAX_PROCESS_DIMENSION
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _resolve_diameter_bounds(
    width: int, height: int, settings: ShapeDetectionSettings
) -> tuple[float, float]:
    short = float(min(width, height))
    if settings.size_mode == "custom":
        if settings.min_diameter_px > 0 and settings.max_diameter_px > 0:
            min_d = float(settings.min_diameter_px)
            max_d = float(settings.max_diameter_px)
        else:
            min_d = short * (float(settings.min_diameter_pct) / 100.0)
            max_d = short * (float(settings.max_diameter_pct) / 100.0)
    else:
        # Auto — Count Things–style relative ranges
        min_d = short * 0.02
        max_d = short * 0.48

    min_d = max(4.0, min_d)
    max_d = max(min_d + 2.0, min(max_d, short * 0.95))
    if max_d <= min_d:
        raise ShapeDetectionError(
            "Maximum circle diameter must be greater than the minimum."
        )
    return min_d, max_d


def _is_partial(cx: float, cy: float, radius: float, w: int, h: int) -> bool:
    return (
        cx - radius < 0
        or cy - radius < 0
        or cx + radius > w - 1
        or cy + radius > h - 1
    )


def _prepare_gray(image: np.ndarray, *, for_objects: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if for_objects:
        gray = cv2.medianBlur(gray, 5)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return gray


def detect_hough_candidates(
    gray: np.ndarray,
    *,
    min_radius: int,
    max_radius: int,
    min_dist: float,
    param1: float,
    param2: float,
) -> list[_RawCandidate]:
    h, w = gray.shape[:2]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1.0, min_dist),
        param1=float(param1),
        param2=float(param2),
        minRadius=max(1, int(min_radius)),
        maxRadius=max(int(min_radius) + 1, int(max_radius)),
    )
    out: list[_RawCandidate] = []
    if circles is None:
        return out
    for x, y, r in np.round(circles[0]).astype(float):
        if r < 1:
            continue
        # Quality from how well the radius fits image scale + edge strength proxy
        quality = float(
            np.clip(1.0 - abs((2 * r) / max(min(w, h), 1) - 0.15) * 0.5, 0.15, 0.95)
        )
        out.append(
            _RawCandidate(
                cx=float(x),
                cy=float(y),
                radius=float(r),
                method="hough",
                quality=quality,
                partial=_is_partial(x, y, r, w, h),
            )
        )
    return out


def detect_contour_candidates(
    gray: np.ndarray,
    *,
    min_area: float,
    max_area: float,
    min_radius: float,
    max_radius: float,
    circularity_threshold: float,
    outlined: bool,
) -> list[_RawCandidate]:
    h, w = gray.shape[:2]
    candidates: list[_RawCandidate] = []

    if outlined:
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        # Adaptive threshold catches filled blobs (coins, caps, plates)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = float(cv2.arcLength(cnt, True))
        circ = circularity(area, peri)
        if circ < circularity_threshold:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 2 or bh < 2:
            continue
        aspect = bw / float(bh)
        if aspect < 0.75 or aspect > 1.35:
            # Reject elongated ellipses / rectangles
            continue
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < 0.80:
            continue
        (_cx, _cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_radius or radius > max_radius:
            continue
        # Prefer circularity as the quality measure
        quality = float(np.clip(circ * 0.85 + solidity * 0.15, 0.0, 1.0))
        candidates.append(
            _RawCandidate(
                cx=float(_cx),
                cy=float(_cy),
                radius=float(radius),
                method="contour",
                quality=quality,
                partial=_is_partial(_cx, _cy, radius, w, h),
            )
        )
    return candidates


def _centers_close(a: _RawCandidate, b: _RawCandidate) -> bool:
    dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
    # Relative to the larger radius so Hough+contour pairs merge reliably.
    return dist <= 0.55 * max(a.radius, b.radius) + 3.0


def _radii_similar(a: _RawCandidate, b: _RawCandidate) -> bool:
    larger = max(a.radius, b.radius)
    if larger <= 1e-6:
        return True
    return abs(a.radius - b.radius) / larger <= 0.40


def _is_concentric(a: _RawCandidate, b: _RawCandidate) -> bool:
    dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
    return dist <= 0.25 * min(a.radius, b.radius) + 2.0 and not _radii_similar(a, b)


def merge_candidates(
    candidates: list[_RawCandidate],
    *,
    count_concentric_separately: bool,
    include_partial: bool,
) -> list[_RawCandidate]:
    """Circle-aware duplicate suppression; prefer dual-method support."""
    if not candidates:
        return []

    # Sort: dual-method will be built by merging — start with higher quality
    ordered = sorted(candidates, key=lambda c: (-c.quality, -c.radius))
    groups: list[list[_RawCandidate]] = []

    for cand in ordered:
        placed = False
        for group in groups:
            rep = group[0]
            concentric = _is_concentric(cand, rep)
            if concentric and not count_concentric_separately:
                # Keep the outer (larger) ring as the object
                group.append(cand)
                group.sort(key=lambda c: -c.radius)
                placed = True
                break
            if concentric and count_concentric_separately:
                continue
            if _centers_close(cand, rep) and _radii_similar(cand, rep):
                group.append(cand)
                placed = True
                break
        if not placed:
            groups.append([cand])

    merged: list[_RawCandidate] = []
    for group in groups:
        methods = sorted({c.method for c in group})
        # Prefer mean of higher-quality members; boost when both methods agree
        weights = [max(0.05, c.quality) for c in group]
        tw = sum(weights)
        cx = sum(c.cx * w for c, w in zip(group, weights)) / tw
        cy = sum(c.cy * w for c, w in zip(group, weights)) / tw
        if not count_concentric_separately and len({round(c.radius) for c in group}) > 1:
            radius = max(c.radius for c in group)
        else:
            radius = sum(c.radius * w for c, w in zip(group, weights)) / tw
        quality = max(c.quality for c in group)
        if len(methods) > 1:
            quality = min(1.0, quality + 0.08)
        partial = any(c.partial for c in group)
        if partial and not include_partial:
            continue
        # Represent merged methods on a synthetic candidate via quality only;
        # methods carried forward when building CircleDetection.
        primary = _RawCandidate(
            cx=cx,
            cy=cy,
            radius=radius,
            method="+".join(methods),
            quality=quality,
            partial=partial,
        )
        merged.append(primary)
    return merged


def _to_detections(
    merged: list[_RawCandidate],
    *,
    scale: float,
    orig_w: int,
    orig_h: int,
) -> list[CircleDetection]:
    inv = 1.0 / scale if scale > 0 else 1.0
    dets: list[CircleDetection] = []
    # Stable order: top-to-bottom, left-to-right
    sorted_m = sorted(merged, key=lambda c: (c.cy * inv, c.cx * inv))
    for idx, cand in enumerate(sorted_m[:MAX_FINAL_DETECTIONS], start=1):
        cx = cand.cx * inv
        cy = cand.cy * inv
        radius = cand.radius * inv
        methods = cand.method.split("+") if cand.method else []
        dets.append(
            CircleDetection(
                id=f"shape-{idx}",
                shape="circle",
                center_x=cx,
                center_y=cy,
                radius=radius,
                diameter=radius * 2.0,
                bounding_box=BoundingBox(
                    x1=max(0.0, cx - radius),
                    y1=max(0.0, cy - radius),
                    x2=min(float(orig_w), cx + radius),
                    y2=min(float(orig_h), cy + radius),
                ),
                detection_methods=methods,
                quality_score=float(round(cand.quality, 4)),
                partial=_is_partial(cx, cy, radius, orig_w, orig_h) or cand.partial,
                included=True,
                sequence_number=idx,
            )
        )
    return dets


def detect_circles(
    image_bgr: np.ndarray,
    settings: ShapeDetectionSettings | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[CircleDetection], dict[str, Any]]:
    """Run Hough + contour pipeline; coordinates are in original image space."""
    settings = apply_mode_presets(settings or balanced_defaults())
    if progress:
        progress(PROGRESS_STEPS[0])

    orig_h, orig_w = image_bgr.shape[:2]
    processed, scale = _scale_for_processing(image_bgr)
    proc_h, proc_w = processed.shape[:2]
    min_d, max_d = _resolve_diameter_bounds(proc_w, proc_h, settings)
    min_r = min_d / 2.0
    max_r = max_d / 2.0
    min_dist = max(
        4.0, min(proc_w, proc_h) * (float(settings.min_center_distance_pct) / 100.0)
    )
    min_area = math.pi * (min_r**2) * 0.55
    max_area = math.pi * (max_r**2) * 1.25

    if progress:
        progress(PROGRESS_STEPS[1])

    raw: list[_RawCandidate] = []
    target = settings.target_type

    run_hough = settings.use_hough and target in {
        "circular_objects",
        "drawn_outlined",
        "both",
    }
    run_contour_objects = settings.use_contour and target in {
        "circular_objects",
        "both",
    }
    run_contour_outlined = settings.use_contour and target in {
        "drawn_outlined",
        "both",
    }

    if run_hough:
        # Objects: slightly stronger blur; outlined: sharper edges — run both when Both
        if target == "circular_objects":
            hough_passes = [True]
        elif target == "drawn_outlined":
            hough_passes = [False]
        else:
            hough_passes = [True, False]
        for for_objects in hough_passes:
            gray = _prepare_gray(processed, for_objects=for_objects)
            raw.extend(
                detect_hough_candidates(
                    gray,
                    min_radius=int(min_r),
                    max_radius=int(max_r),
                    min_dist=min_dist,
                    param1=float(settings.edge_sensitivity),
                    param2=float(settings.hough_accumulator),
                )
            )

    if progress:
        progress(PROGRESS_STEPS[2])

    if run_contour_objects:
        gray_obj = _prepare_gray(processed, for_objects=True)
        raw.extend(
            detect_contour_candidates(
                gray_obj,
                min_area=min_area,
                max_area=max_area,
                min_radius=min_r,
                max_radius=max_r,
                circularity_threshold=float(settings.contour_circularity),
                outlined=False,
            )
        )
    if run_contour_outlined:
        gray_line = _prepare_gray(processed, for_objects=False)
        raw.extend(
            detect_contour_candidates(
                gray_line,
                min_area=min_area * 0.35,
                max_area=max_area,
                min_radius=min_r,
                max_radius=max_r,
                circularity_threshold=max(0.55, float(settings.contour_circularity) - 0.08),
                outlined=True,
            )
        )

    if len(raw) > MAX_CANDIDATES_BEFORE_DEDUP:
        raise ShapeDetectionError(MSG_TOO_MANY)

    if progress:
        progress(PROGRESS_STEPS[3])

    merged = merge_candidates(
        raw,
        count_concentric_separately=bool(settings.count_concentric_separately),
        include_partial=bool(settings.include_partial),
    )
    if progress:
        progress(PROGRESS_STEPS[4])

    detections = _to_detections(
        merged, scale=scale, orig_w=orig_w, orig_h=orig_h
    )
    meta = {
        "original_width": orig_w,
        "original_height": orig_h,
        "processed_width": proc_w,
        "processed_height": proc_h,
        "scale": scale,
        "min_diameter": min_d / scale,
        "max_diameter": max_d / scale,
        "raw_candidates": len(raw),
        "merged_candidates": len(merged),
    }
    return detections, meta


_DETECTOR_KIND = {
    "opencv_circle_detector": "circle",
    "opencv_rectangle_detector": "rectangle",
    "opencv_square_detector": "square",
    "opencv_triangle_detector": "triangle",
    "opencv_polygon_detector": "polygon",
    "opencv_line_detector": "line",
    "opencv_ellipse_detector": "ellipse",
}


def run_shape_detection(
    image_bytes: bytes,
    *,
    requested_shape: str,
    settings: ShapeDetectionSettings | None = None,
    progress: Callable[[str], None] | None = None,
) -> ShapeDetectionResult:
    """Full entry point: validate shape + image, detect, return result model."""
    try:
        shape = resolve_shape(requested_shape)
    except ShapeResolutionError as exc:
        raise ShapeDetectionError(str(exc)) from exc

    kind = _DETECTOR_KIND.get(shape.detector)
    if not kind:
        raise ShapeDetectionError(
            "That shape detector is not available in this build."
        )

    meta_img = validate_shape_image_bytes(image_bytes)
    settings = settings or balanced_defaults()
    started = time.perf_counter()
    try:
        image = decode_bgr(image_bytes)
        if progress:
            progress(PROGRESS_STEPS[0])
        if kind == "circle":
            detections, meta = detect_circles(image, settings, progress=progress)
        else:
            from shape_geometry import detect_geometry_shapes

            if progress:
                progress(PROGRESS_STEPS[1])
            detections, meta = detect_geometry_shapes(
                image, kind=kind, settings=settings  # type: ignore[arg-type]
            )
            if progress:
                progress(PROGRESS_STEPS[4])
    except ShapeDetectionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ShapeDetectionError(MSG_INTERNAL, technical=type(exc).__name__) from exc

    elapsed = time.perf_counter() - started
    warning = ""
    if not detections:
        warning = MSG_NO_SHAPES

    return ShapeDetectionResult(
        requested_shape=str(requested_shape),
        normalized_shape=shape.key,
        detections=detections,
        processing_time_seconds=float(elapsed),
        original_width=int(meta["original_width"]),
        original_height=int(meta["original_height"]),
        processed_width=int(meta["processed_width"]),
        processed_height=int(meta["processed_height"]),
        settings=settings.summary_dict(),
        image_hash=str(meta_img["hash"]),
        warning=warning,
    )


def _draw_shape_outline(
    canvas: np.ndarray,
    det: CircleDetection,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    shape = (det.shape or "circle").lower()
    cx, cy = int(round(det.center_x)), int(round(det.center_y))
    if shape == "circle":
        r = max(1, int(round(det.radius)))
        cv2.circle(canvas, (cx, cy), r, color, thickness, lineType=cv2.LINE_AA)
        return
    if shape == "ellipse" and det.width > 0 and det.height > 0:
        axes = (max(1, int(round(det.width / 2))), max(1, int(round(det.height / 2))))
        cv2.ellipse(
            canvas,
            (cx, cy),
            axes,
            float(det.angle),
            0,
            360,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return
    if shape == "line" and len(det.points) >= 2:
        p1 = (int(round(det.points[0][0])), int(round(det.points[0][1])))
        p2 = (int(round(det.points[1][0])), int(round(det.points[1][1])))
        cv2.line(canvas, p1, p2, color, max(2, thickness), lineType=cv2.LINE_AA)
        return
    if det.points and len(det.points) >= 3:
        pts = np.array(
            [[int(round(x)), int(round(y))] for x, y in det.points], dtype=np.int32
        )
        cv2.polylines(canvas, [pts], True, color, thickness, lineType=cv2.LINE_AA)
        return
    bb = det.bounding_box
    cv2.rectangle(
        canvas,
        (int(bb.x1), int(bb.y1)),
        (int(bb.x2), int(bb.y2)),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def annotate_circles(
    image_bgr: np.ndarray,
    detections: list[CircleDetection],
    *,
    style: str = "numbered",
    selected_id: str | None = None,
    solo: bool = False,
) -> np.ndarray:
    """Draw detections onto a copy of the image.

    ``solo=True`` draws only the selected detection as a clean outline
    (no numbers, no other markers) so one item can be inspected alone.
    """
    canvas = image_bgr.copy()
    if solo:
        target = next(
            (d for d in detections if d.id == selected_id and d.included),
            None,
        )
        if target is None and selected_id:
            target = next((d for d in detections if d.id == selected_id), None)
        if target is None:
            return canvas
        _draw_shape_outline(canvas, target, (0, 200, 255), 3)
        return canvas

    show_outline = style in {"numbered", "outlines", "all", "solo"}
    show_center = style in {"centers", "all"}
    show_box = style in {"boxes", "all"}
    show_number = style in {"numbered", "all"}

    for det in detections:
        if not det.included:
            continue
        cx, cy = int(round(det.center_x)), int(round(det.center_y))
        is_sel = bool(selected_id and det.id == selected_id)
        color = (
            (0, 200, 255)
            if is_sel
            else ((40, 180, 80) if not det.partial else (0, 165, 255))
        )
        thickness = 3 if is_sel else 2
        if show_outline:
            _draw_shape_outline(canvas, det, color, thickness)
        if show_center:
            cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        if show_box:
            bb = det.bounding_box
            cv2.rectangle(
                canvas,
                (int(bb.x1), int(bb.y1)),
                (int(bb.x2), int(bb.y2)),
                color,
                1,
                lineType=cv2.LINE_AA,
            )
        if show_number:
            label = str(det.sequence_number or det.id)
            ty = max(16, int(det.bounding_box.y1) - 6)
            tx = int(det.bounding_box.x1)
            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (20, 20, 20),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                1,
                cv2.LINE_AA,
            )
    return canvas


# Public alias used by the UI
annotate_shapes = annotate_circles


def encode_image(image_bgr: np.ndarray, *, fmt: str = "png") -> bytes:
    ext = ".jpg" if fmt.lower() in {"jpg", "jpeg"} else ".png"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        raise ShapeDetectionError("Could not encode the annotated image.")
    return bytes(buf)


def sanitize_upload_filename(name: str | None) -> str:
    from admin_samples import slugify

    original = str(name or "upload")
    if ".." in original or original.startswith(("/", "\\")) or "\x00" in original:
        raise ShapeDetectionError("That filename is not allowed.")
    raw = original.replace("\\", "/").split("/")[-1]
    if not raw or raw in {".", ".."}:
        raise ShapeDetectionError("That filename is not allowed.")
    stem = raw.rsplit(".", 1)[0] if "." in raw else raw
    safe = slugify(stem, fallback="upload")
    return safe[:80]


def compute_final_count(
    detections: list[CircleDetection], *, manually_added: int = 0
) -> dict[str, int]:
    included = sum(1 for d in detections if d.included)
    excluded = sum(1 for d in detections if not d.included)
    manual = max(0, int(manually_added))
    return {
        "included": included,
        "excluded": excluded,
        "manually_added": manual,
        "final": included + manual,
    }
