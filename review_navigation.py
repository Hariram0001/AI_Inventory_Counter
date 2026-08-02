"""Scalable detection navigation helpers for Review (no Streamlit)."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from confidence_ui import format_confidence_percent
from schemas import Detection

DetectionFilter = Literal["all", "included", "excluded", "warnings", "manual"]

PAGE_SIZE = 15
ITEM_TYPE_ALL = "All types"


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


def resolve_item_type(
    class_name: str | None,
    *,
    alias_map: dict[str, str] | None = None,
) -> str:
    """Map a detection class onto its primary item type label."""
    raw = str(class_name or "").strip()
    if not raw:
        return "object"
    if alias_map:
        hit = alias_map.get(raw.casefold()) or alias_map.get(
            raw.replace("_", " ").casefold()
        )
        if hit:
            return hit
    return raw.replace("_", " ")


def available_item_types(
    detections: Sequence[Detection],
    *,
    primary_types: Sequence[str] | None = None,
    alias_map: dict[str, str] | None = None,
    requested_only: bool = True,
) -> list[str]:
    """Item-type options for the Review type selector.

    When ``requested_only`` is True (default), only the inventory types the
    user asked to detect are listed — not every class the model happened to
    return.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for name in primary_types or []:
        key = str(name).strip()
        fold = key.casefold()
        if key and fold not in seen:
            seen.add(fold)
            ordered.append(key)
    if requested_only:
        return ordered
    for det in detections:
        label = resolve_item_type(getattr(det, "class_name", None), alias_map=alias_map)
        fold = label.casefold()
        if fold not in seen:
            seen.add(fold)
            ordered.append(label)
    return ordered


def next_detection_id_after_toggle(
    pool: Sequence[Detection],
    detection_id: str | None,
) -> str | None:
    """Pick the next navigator id after include/exclude removes the current one."""
    if not pool:
        return None
    idx = index_of_detection(pool, detection_id)
    if idx + 1 < len(pool):
        return pool[idx + 1].detection_id
    if idx - 1 >= 0:
        return pool[idx - 1].detection_id
    return None


def filter_detections(
    detections: Sequence[Detection],
    filter_key: str,
    *,
    excluded_detections: Sequence[Detection] | None = None,
    item_type: str | None = None,
    alias_map: dict[str, str] | None = None,
) -> list[Detection]:
    """
    Filter navigator candidates.

    `detections` should be the currently included/visible set.
    Pass `excluded_detections` when filter_key == 'excluded'.

    ``item_type`` filters to one primary class/type for viewing. Marker numbers
    are left unchanged so numbering stays shared across all types.
    """
    key = (filter_key or "all").strip().lower()
    type_filter = str(item_type or "").strip()
    if type_filter.casefold() in {"", "all", "all types", ITEM_TYPE_ALL.casefold()}:
        type_filter = ""
    type_fold = type_filter.casefold()

    if key == "excluded":
        base = list(excluded_detections or [])
    else:
        base = []
        for d in detections:
            if key in {"all", "included"}:
                base.append(d)
            elif key == "warnings" and (
                d.suspected_overlap
                or d.suspected_occlusion
                or float(d.confidence) < 0.5
            ):
                base.append(d)
            elif key == "manual" and d.is_manual:
                base.append(d)

    if not type_fold:
        return base
    out: list[Detection] = []
    for d in base:
        label = resolve_item_type(getattr(d, "class_name", None), alias_map=alias_map)
        if label.casefold() == type_fold:
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
