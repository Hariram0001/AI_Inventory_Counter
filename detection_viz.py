"""Stable detection colors and annotation helpers (no Streamlit)."""

from __future__ import annotations

import hashlib
from typing import Sequence

from schemas import Detection

# Controlled rainbow palette for numbered detections (RGB).
DETECTION_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (229, 83, 75),  # red
    (255, 140, 0),  # orange
    (154, 205, 50),  # yellow-green
    (46, 160, 67),  # green
    (0, 150, 136),  # teal
    (33, 150, 243),  # blue
    (63, 81, 181),  # indigo
    (156, 39, 176),  # purple
]


def color_for_detection_id(detection_id: str | None, fallback_index: int = 0) -> tuple[int, int, int]:
    """Stable color for a detection ID (same ID → same color across UI surfaces)."""
    key = (detection_id or f"idx:{fallback_index}").encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    idx = int(digest[:8], 16) % len(DETECTION_COLOR_PALETTE)
    return DETECTION_COLOR_PALETTE[idx]


def color_for_detection(det: Detection, fallback_index: int = 0) -> tuple[int, int, int]:
    return color_for_detection_id(getattr(det, "detection_id", None), fallback_index)


def contrasting_text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if luminance > 160 else (255, 255, 255)


def css_rgb(rgb: tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def assign_marker_numbers(
    detections: Sequence[Detection],
    *,
    number_region_excluded: bool = False,
) -> list[Detection]:
    """Sort by position and assign consecutive visible marker numbers.

    By default, detections with ``excluded_by_region`` are left unnumbered so
    toggling them in Review does not renumber included markers.
    """
    ordered = sorted(
        list(detections), key=lambda d: (d.center_y, d.center_x, d.detection_id)
    )
    n = 1
    for d in ordered:
        if getattr(d, "excluded_by_region", False) and not number_region_excluded:
            d.marker_number = None
            continue
        d.marker_number = n
        n += 1
    return ordered
