"""Scalable detection navigation helpers for Review (no Streamlit)."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from confidence_ui import format_confidence_percent
from schemas import Detection

DetectionFilter = Literal["all", "included", "excluded", "warnings", "manual"]

PAGE_SIZE = 15


def format_detection_option(det: Detection, *, excluded: bool = False) -> str:
    """Human-facing selector label — never includes internal detection IDs."""
    num = det.marker_number or "?"
    conf = format_confidence_percent(det.confidence)
    status = "Excluded" if excluded else ("Manual" if det.is_manual else conf)
    if excluded:
        return f"#{num} — {det.class_name} — Excluded"
    if det.is_manual:
        return f"#{num} — {det.class_name} — Manual · {conf}"
    return f"#{num} — {det.class_name} — {conf}"


def filter_detections(
    detections: Sequence[Detection],
    filter_key: str,
    *,
    excluded_detections: Sequence[Detection] | None = None,
) -> list[Detection]:
    """
    Filter navigator candidates.

    `detections` should be the currently included/visible set.
    Pass `excluded_detections` when filter_key == 'excluded'.
    """
    key = (filter_key or "all").strip().lower()
    if key == "excluded":
        return list(excluded_detections or [])
    out: list[Detection] = []
    for d in detections:
        if key in {"all", "included"}:
            out.append(d)
        elif key == "warnings" and (
            d.suspected_overlap or d.suspected_occlusion or float(d.confidence) < 0.5
        ):
            out.append(d)
        elif key == "manual" and d.is_manual:
            out.append(d)
    return out


def index_of_detection(detections: Sequence[Detection], detection_id: str | None) -> int:
    if not detection_id:
        return 0
    for i, d in enumerate(detections):
        if d.detection_id == detection_id:
            return i
    return 0


def step_detection_id(
    detections: Sequence[Detection],
    detection_id: str | None,
    *,
    delta: int,
) -> str | None:
    if not detections:
        return None
    idx = index_of_detection(detections, detection_id)
    nxt = max(0, min(len(detections) - 1, idx + delta))
    return detections[nxt].detection_id


def paginate(
    items: Sequence[Any],
    page: int,
    page_size: int = PAGE_SIZE,
) -> tuple[list[Any], int, int]:
    """Return (page_items, page_index_0based, total_pages)."""
    if page_size < 1:
        page_size = PAGE_SIZE
    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(total_pages - 1, int(page)))
    start = page * page_size
    return list(items[start : start + page_size]), page, total_pages


def build_synthetic_detections(count: int) -> list[Detection]:
    """Offline fixture for scalability tests (no API)."""
    dets: list[Detection] = []
    cols = 20
    for i in range(count):
        row, col = divmod(i, cols)
        x1 = 10.0 + col * 40
        y1 = 10.0 + row * 30
        dets.append(
            Detection(
                detection_id=f"synth-{i:04d}",
                class_name="fence panel" if i % 5 else "fence post",
                confidence=0.35 + (i % 60) / 100.0,
                x1=x1,
                y1=y1,
                x2=x1 + 28,
                y2=y1 + 22,
                center_x=x1 + 14,
                center_y=y1 + 11,
                width=28,
                height=22,
                source_model="YOLO-World",
                source_image="synthetic.jpg",
                suspected_overlap=(i % 17 == 0),
                suspected_occlusion=(i % 23 == 0),
                marker_number=i + 1,
            )
        )
    return dets
