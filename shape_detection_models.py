"""Dataclasses and settings for local shape detection (no Streamlit)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DetectionMode = Literal["strict", "balanced", "sensitive"]
TargetType = Literal["circular_objects", "drawn_outlined", "both"]
SizeMode = Literal["auto", "custom"]
AnnotationStyle = Literal[
    "numbered", "outlines", "centers", "boxes", "all"
]
ReviewStatus = Literal[
    "unreviewed", "correct", "false_positive", "duplicate", "ignore"
]

DETECTOR_VERSION = "opencv_circle_v1"

MODE_LABELS: dict[str, str] = {
    "strict": "Strict",
    "balanced": "Balanced",
    "sensitive": "Sensitive",
}

TARGET_LABELS: dict[str, str] = {
    "circular_objects": "Circular objects",
    "drawn_outlined": "Drawn or outlined circles",
    "both": "Both",
}

ANNOTATION_LABELS: dict[str, str] = {
    "numbered": "Numbered Circles",
    "outlines": "Circle Outlines",
    "centers": "Centers",
    "boxes": "Bounding Boxes",
    "all": "All",
}

REVIEW_STATUS_LABELS: dict[str, str] = {
    "unreviewed": "Unreviewed",
    "correct": "Correct",
    "false_positive": "False positive",
    "duplicate": "Duplicate",
    "ignore": "Ignore",
}

# Soft caps for safety
MAX_PROCESS_DIMENSION = 1600
MAX_SOURCE_DIMENSION = 6000
MAX_CANDIDATES_BEFORE_DEDUP = 2500
MAX_FINAL_DETECTIONS = 500


@dataclass
class ShapeDetectionSettings:
    mode: DetectionMode = "balanced"
    target_type: TargetType = "both"
    size_mode: SizeMode = "auto"
    # Percentage of shortest image side (used when size_mode == custom)
    min_diameter_pct: float = 2.0
    max_diameter_pct: float = 45.0
    # Absolute pixels override when > 0 and size_mode == custom
    min_diameter_px: float = 0.0
    max_diameter_px: float = 0.0
    min_center_distance_pct: float = 4.0
    edge_sensitivity: float = 100.0
    hough_accumulator: float = 28.0
    contour_circularity: float = 0.72
    include_partial: bool = True
    count_concentric_separately: bool = False
    use_hough: bool = True
    use_contour: bool = True

    def cache_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_type": self.target_type,
            "size_mode": self.size_mode,
            "min_diameter_pct": self.min_diameter_pct,
            "max_diameter_pct": self.max_diameter_pct,
            "min_diameter_px": self.min_diameter_px,
            "max_diameter_px": self.max_diameter_px,
            "min_center_distance_pct": self.min_center_distance_pct,
            "edge_sensitivity": self.edge_sensitivity,
            "hough_accumulator": self.hough_accumulator,
            "contour_circularity": self.contour_circularity,
            "include_partial": self.include_partial,
            "count_concentric_separately": self.count_concentric_separately,
            "use_hough": self.use_hough,
            "use_contour": self.use_contour,
            "detector_version": DETECTOR_VERSION,
        }


def balanced_defaults() -> ShapeDetectionSettings:
    return ShapeDetectionSettings()


def apply_mode_presets(settings: ShapeDetectionSettings) -> ShapeDetectionSettings:
    """Tune advanced thresholds from the primary Strict/Balanced/Sensitive mode."""
    s = ShapeDetectionSettings(**asdict(settings))
    if s.mode == "strict":
        s.edge_sensitivity = 120.0
        s.hough_accumulator = 36.0
        s.contour_circularity = 0.80
        s.min_center_distance_pct = 5.0
    elif s.mode == "sensitive":
        s.edge_sensitivity = 80.0
        s.hough_accumulator = 18.0
        s.contour_circularity = 0.62
        s.min_center_distance_pct = 3.0
    else:
        s.edge_sensitivity = 100.0
        s.hough_accumulator = 28.0
        s.contour_circularity = 0.72
        s.min_center_distance_pct = 4.0
    return s


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
        }


@dataclass
class CircleDetection:
    id: str
    shape: str = "circle"
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0
    diameter: float = 0.0
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )
    detection_methods: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    partial: bool = False
    included: bool = True
    review_status: ReviewStatus = "unreviewed"
    sequence_number: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shape": self.shape,
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "radius": float(self.radius),
            "diameter": float(self.diameter),
            "bounding_box": self.bounding_box.as_dict(),
            "detection_methods": list(self.detection_methods),
            "quality_score": float(self.quality_score),
            "partial": bool(self.partial),
            "included": bool(self.included),
            "review_status": self.review_status,
            "sequence_number": int(self.sequence_number),
        }


@dataclass
class ShapeDetectionResult:
    requested_shape: str
    normalized_shape: str
    detections: list[CircleDetection]
    processing_time_seconds: float
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    settings: dict[str, Any]
    image_hash: str
    detector_version: str = DETECTOR_VERSION
    warning: str = ""
    error: str = ""
    manually_added_count: int = 0
    manual_notes: str = ""

    @property
    def detected_count(self) -> int:
        return len(self.detections)

    @property
    def included_count(self) -> int:
        return sum(1 for d in self.detections if d.included)

    @property
    def excluded_count(self) -> int:
        return sum(1 for d in self.detections if not d.included)

    @property
    def partial_count(self) -> int:
        return sum(1 for d in self.detections if d.partial and d.included)

    @property
    def hough_count(self) -> int:
        return sum(
            1 for d in self.detections if d.included and "hough" in d.detection_methods
        )

    @property
    def contour_count(self) -> int:
        return sum(
            1
            for d in self.detections
            if d.included and "contour" in d.detection_methods
        )

    @property
    def final_count(self) -> int:
        return max(0, self.included_count + int(self.manually_added_count or 0))

    def cache_key(self) -> str:
        return build_cache_key(
            self.image_hash, self.normalized_shape, self.settings
        )

    def public_export_dict(self) -> dict[str, Any]:
        """JSON-safe export without secrets or private paths."""
        return {
            "requested_shape": self.requested_shape,
            "normalized_shape": self.normalized_shape,
            "image": {
                "hash": self.image_hash,
                "original_width": self.original_width,
                "original_height": self.original_height,
                "processed_width": self.processed_width,
                "processed_height": self.processed_height,
            },
            "settings": dict(self.settings),
            "detections": [d.as_dict() for d in self.detections],
            "counts": {
                "detected": self.detected_count,
                "included": self.included_count,
                "excluded": self.excluded_count,
                "partial": self.partial_count,
                "manually_added": int(self.manually_added_count or 0),
                "final": self.final_count,
            },
            "processing_time_seconds": float(self.processing_time_seconds),
            "detector_version": self.detector_version,
            "manual_notes": self.manual_notes,
            "warning": self.warning,
        }


def build_cache_key(
    image_hash: str,
    normalized_shape: str,
    settings: dict[str, Any] | ShapeDetectionSettings,
) -> str:
    payload = {
        "image_hash": image_hash,
        "shape": normalized_shape,
        "settings": settings.cache_dict()
        if isinstance(settings, ShapeDetectionSettings)
        else dict(settings),
        "detector_version": DETECTOR_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def hash_image_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
