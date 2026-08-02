"""OpenCV contour / Hough detectors for non-circle shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from shape_detection_models import (
    MAX_CANDIDATES_BEFORE_DEDUP,
    MAX_FINAL_DETECTIONS,
    MAX_PROCESS_DIMENSION,
    BoundingBox,
    CircleDetection,
    ShapeDetectionSettings,
    apply_mode_presets,
)

ShapeKind = Literal[
    "rectangle", "square", "triangle", "polygon", "line", "ellipse"
]


@dataclass
class GeometryCandidate:
    shape: str
    cx: float
    cy: float
    points: list[tuple[float, float]] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0
    radius: float = 0.0
    quality: float = 0.0
    partial: bool = False
    method: str = "contour"


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
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def _prepare_edges(gray: np.ndarray, *, sensitive: bool) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    lo, hi = (40, 120) if sensitive else (60, 160)
    edges = cv2.Canny(blur, lo, hi)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def _is_partial_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> bool:
    pad = 2.0
    return x1 <= pad or y1 <= pad or x2 >= w - pad or y2 >= h - pad


def _bbox_from_points(pts: list[tuple[float, float]]) -> BoundingBox:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def _angle_cos(p0, p1, p2) -> float:
    v1 = np.array(p0, dtype=float) - np.array(p1, dtype=float)
    v2 = np.array(p2, dtype=float) - np.array(p1, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def _approx_contours(
    image_bgr: np.ndarray, settings: ShapeDetectionSettings
) -> list[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    sensitive = settings.mode == "sensitive"
    edges = _prepare_edges(gray, sensitive=sensitive)
    # Also try binary for filled shapes
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )
    contours: list[np.ndarray] = []
    for src in (edges, binary):
        found, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    return contours


def _size_ok(area: float, short: float, settings: ShapeDetectionSettings) -> bool:
    min_area = (short * 0.02) ** 2
    max_area = (short * 0.85) ** 2
    if settings.mode == "strict":
        min_area = (short * 0.03) ** 2
    elif settings.mode == "sensitive":
        min_area = (short * 0.015) ** 2
    return min_area <= area <= max_area


def detect_polygonal(
    image_bgr: np.ndarray,
    *,
    kind: ShapeKind,
    settings: ShapeDetectionSettings,
) -> list[GeometryCandidate]:
    h, w = image_bgr.shape[:2]
    short = float(min(w, h))
    out: list[GeometryCandidate] = []
    for cnt in _approx_contours(image_bgr, settings):
        area = float(cv2.contourArea(cnt))
        if not _size_ok(area, short, settings):
            continue
        peri = float(cv2.arcLength(cnt, True))
        if peri < 12:
            continue
        eps = 0.04 * peri if kind != "polygon" else 0.03 * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        verts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        n = len(verts)
        if kind == "triangle" and n != 3:
            continue
        if kind in {"rectangle", "square"} and n != 4:
            continue
        if kind == "polygon" and n < 5:
            continue
        if kind == "polygon" and n > 12:
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        if bw < 6 or bh < 6:
            continue
        aspect = bw / float(bh)
        if kind == "square":
            if aspect < 0.82 or aspect > 1.22:
                continue
        if kind == "rectangle":
            # Prefer non-square rectangles; squares have their own detector
            if 0.88 <= aspect <= 1.12:
                continue

        # Corner angle check for quads
        if n == 4:
            cosines = [
                abs(_angle_cos(verts[i - 1], verts[i], verts[(i + 1) % 4]))
                for i in range(4)
            ]
            # Right angles → cos near 0
            if max(cosines) > 0.45:
                continue

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) or 1.0
        solidity = area / hull_area
        if solidity < 0.75 and kind != "polygon":
            continue

        M = cv2.moments(approx)
        if abs(M["m00"]) < 1e-6:
            cx, cy = x + bw / 2.0, y + bh / 2.0
        else:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

        quality = float(np.clip(solidity * 0.7 + min(1.0, area / (short * short)) * 0.3, 0, 1))
        out.append(
            GeometryCandidate(
                shape=kind,
                cx=cx,
                cy=cy,
                points=verts,
                width=float(bw),
                height=float(bh),
                radius=math.hypot(bw, bh) / 2.0,
                quality=quality,
                partial=_is_partial_box(x, y, x + bw, y + bh, w, h),
                method="contour",
            )
        )
        if len(out) > MAX_CANDIDATES_BEFORE_DEDUP:
            break
    return out


def detect_ellipses(
    image_bgr: np.ndarray, *, settings: ShapeDetectionSettings
) -> list[GeometryCandidate]:
    h, w = image_bgr.shape[:2]
    short = float(min(w, h))
    out: list[GeometryCandidate] = []
    for cnt in _approx_contours(image_bgr, settings):
        if len(cnt) < 5:
            continue
        area = float(cv2.contourArea(cnt))
        if not _size_ok(area, short, settings):
            continue
        try:
            (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        major, minor = max(ma, mi), min(ma, mi)
        if minor < 8 or major < 12:
            continue
        ratio = major / max(minor, 1e-6)
        # Must be elongated enough to not be a circle
        if ratio < 1.25:
            continue
        if ratio > 4.5:
            continue
        # Circularity of contour should still be reasonably smooth
        peri = float(cv2.arcLength(cnt, True))
        circ = (4 * math.pi * area) / (peri * peri) if peri > 1 else 0
        if circ < 0.45:
            continue
        x1, y1 = cx - major / 2.0, cy - minor / 2.0
        x2, y2 = cx + major / 2.0, cy + minor / 2.0
        out.append(
            GeometryCandidate(
                shape="ellipse",
                cx=float(cx),
                cy=float(cy),
                width=float(major),
                height=float(minor),
                angle=float(angle),
                radius=float(major / 2.0),
                quality=float(np.clip(circ, 0, 1)),
                partial=_is_partial_box(x1, y1, x2, y2, w, h),
                method="ellipse_fit",
            )
        )
    return out


def detect_lines(
    image_bgr: np.ndarray, *, settings: ShapeDetectionSettings
) -> list[GeometryCandidate]:
    h, w = image_bgr.shape[:2]
    short = float(min(w, h))
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = _prepare_edges(gray, sensitive=settings.mode == "sensitive")
    min_len = short * (0.12 if settings.mode == "strict" else 0.08 if settings.mode == "balanced" else 0.05)
    threshold = 60 if settings.mode == "strict" else 40 if settings.mode == "balanced" else 25
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=int(min_len),
        maxLineGap=int(short * 0.03),
    )
    out: list[GeometryCandidate] = []
    if lines is None:
        return out
    for x1, y1, x2, y2 in lines[:, 0]:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length < min_len:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        out.append(
            GeometryCandidate(
                shape="line",
                cx=cx,
                cy=cy,
                points=[(float(x1), float(y1)), (float(x2), float(y2))],
                width=length,
                height=2.0,
                angle=angle,
                radius=length / 2.0,
                quality=float(np.clip(length / short, 0.2, 1.0)),
                partial=_is_partial_box(
                    min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), w, h
                ),
                method="hough",
            )
        )
    return out


def _iou_box(a: GeometryCandidate, b: GeometryCandidate) -> float:
    if a.points and b.points and a.shape != "line":
        ba = _bbox_from_points(a.points)
        bb = _bbox_from_points(b.points)
    else:
        ba = BoundingBox(a.cx - a.width / 2, a.cy - a.height / 2, a.cx + a.width / 2, a.cy + a.height / 2)
        bb = BoundingBox(b.cx - b.width / 2, b.cy - b.height / 2, b.cx + b.width / 2, b.cy + b.height / 2)
    ix1, iy1 = max(ba.x1, bb.x1), max(ba.y1, bb.y1)
    ix2, iy2 = min(ba.x2, bb.x2), min(ba.y2, bb.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        # Lines: center proximity
        if a.shape == "line" and b.shape == "line":
            dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
            return 1.0 if dist < 0.15 * max(a.width, b.width) and abs(a.angle - b.angle) < 12 else 0.0
        return 0.0
    area_a = max(1.0, (ba.x2 - ba.x1) * (ba.y2 - ba.y1))
    area_b = max(1.0, (bb.x2 - bb.x1) * (bb.y2 - bb.y1))
    return inter / (area_a + area_b - inter)


def merge_geometry(
    candidates: list[GeometryCandidate],
    *,
    include_partial: bool,
) -> list[GeometryCandidate]:
    ordered = sorted(candidates, key=lambda c: -c.quality)
    kept: list[GeometryCandidate] = []
    for cand in ordered:
        if cand.partial and not include_partial:
            continue
        if any(_iou_box(cand, k) >= 0.45 for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= MAX_FINAL_DETECTIONS:
            break
    return kept


def candidates_to_detections(
    merged: list[GeometryCandidate],
    *,
    scale: float,
    orig_w: int,
    orig_h: int,
) -> list[CircleDetection]:
    inv = 1.0 / scale if scale > 0 else 1.0
    dets: list[CircleDetection] = []
    sorted_m = sorted(merged, key=lambda c: (c.cy * inv, c.cx * inv))
    for idx, cand in enumerate(sorted_m, start=1):
        cx, cy = cand.cx * inv, cand.cy * inv
        pts = [(x * inv, y * inv) for x, y in cand.points]
        if pts:
            bb = _bbox_from_points(pts)
        else:
            hw, hh = (cand.width * inv) / 2.0, (cand.height * inv) / 2.0
            bb = BoundingBox(cx - hw, cy - hh, cx + hw, cy + hh)
        radius = cand.radius * inv
        dets.append(
            CircleDetection(
                id=f"shape-{idx}",
                shape=cand.shape,
                center_x=cx,
                center_y=cy,
                radius=radius,
                diameter=radius * 2.0,
                bounding_box=bb,
                detection_methods=[cand.method],
                quality_score=float(round(cand.quality, 4)),
                partial=cand.partial,
                included=True,
                sequence_number=idx,
                points=pts,
                width=float(cand.width * inv),
                height=float(cand.height * inv),
                angle=float(cand.angle),
            )
        )
    return dets


def detect_geometry_shapes(
    image_bgr: np.ndarray,
    *,
    kind: ShapeKind,
    settings: ShapeDetectionSettings | None = None,
) -> tuple[list[CircleDetection], dict]:
    settings = apply_mode_presets(settings or ShapeDetectionSettings())
    orig_h, orig_w = image_bgr.shape[:2]
    processed, scale = _scale_for_processing(image_bgr)
    if kind == "ellipse":
        raw = detect_ellipses(processed, settings=settings)
    elif kind == "line":
        raw = detect_lines(processed, settings=settings)
    else:
        raw = detect_polygonal(processed, kind=kind, settings=settings)
    if len(raw) > MAX_CANDIDATES_BEFORE_DEDUP:
        from shape_detection import ShapeDetectionError, MSG_TOO_MANY

        raise ShapeDetectionError(MSG_TOO_MANY)
    merged = merge_geometry(raw, include_partial=bool(settings.include_partial))
    dets = candidates_to_detections(
        merged, scale=scale, orig_w=orig_w, orig_h=orig_h
    )
    meta = {
        "original_width": orig_w,
        "original_height": orig_h,
        "processed_width": processed.shape[1],
        "processed_height": processed.shape[0],
        "scale": scale,
        "raw_candidates": len(raw),
        "merged_candidates": len(merged),
    }
    return dets, meta
