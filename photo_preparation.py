"""Photo preparation: transforms, count areas, eligibility, and masking.

Coordinate convention
---------------------
All region coordinates are stored in **normalized prepared-image space**
(after EXIF transpose, user rotation, and user crop), with x/y in [0, 1].

Detection filtering default rule
--------------------------------
Keep a detection when its center point lies inside the effective count mask:
  effective = (union of includes, or full image if none) MINUS (union of excludes)

An internal ``minimum_detection_overlap`` rule is available for future use but
is not exposed in the normal UI (default center-point rule).
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from schemas import Detection

MASK_MODE_COUNT_FILTER = "count_filter"
MASK_MODE_HIDE_FROM_AI = "hide_from_ai"

OVERLAP_RULE_CENTER = "center_point"
OVERLAP_RULE_MIN_IOU = "minimum_detection_overlap"

RegionType = Literal["include", "exclude"]
ShapeType = Literal["rectangle", "polygon", "brush"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass
class RegionPoint:
    x: float
    y: float

    def normalized(self) -> dict[str, float]:
        return {"x": _clamp01(self.x), "y": _clamp01(self.y)}


@dataclass
class CountRegion:
    region_id: str
    region_type: RegionType
    shape_type: ShapeType
    points_normalized: list[RegionPoint]
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        pts = [p.normalized() for p in self.points_normalized]
        d: dict[str, Any] = {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "shape_type": self.shape_type,
            "points_normalized": pts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "label": self.label,
        }
        if self.shape_type == "rectangle" and len(pts) >= 2:
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            d["x1"] = min(xs)
            d["y1"] = min(ys)
            d["x2"] = max(xs)
            d["y2"] = max(ys)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CountRegion":
        pts_raw = data.get("points_normalized") or []
        if not pts_raw and all(k in data for k in ("x1", "y1", "x2", "y2")):
            pts_raw = [
                {"x": data["x1"], "y": data["y1"]},
                {"x": data["x2"], "y": data["y2"]},
            ]
        points = [
            RegionPoint(x=float(p["x"]), y=float(p["y"]))
            for p in pts_raw
            if isinstance(p, dict) and "x" in p and "y" in p
        ]
        return cls(
            region_id=str(data.get("region_id") or uuid.uuid4().hex[:12]),
            region_type=("exclude" if data.get("region_type") == "exclude" else "include"),
            shape_type=_normalize_shape(data.get("shape_type")),
            points_normalized=points,
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            label=str(data.get("label") or ""),
        )


def _normalize_shape(value: Any) -> ShapeType:
    v = str(value or "rectangle").lower()
    if v in {"polygon", "poly"}:
        return "polygon"
    if v in {"brush", "freehand", "freedraw"}:
        return "brush"
    return "rectangle"


@dataclass
class CropRect:
    """Normalized crop on the post-rotation image (0–1)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, float]:
        x1, x2 = sorted((_clamp01(self.x1), _clamp01(self.x2)))
        y1, y2 = sorted((_clamp01(self.y1), _clamp01(self.y2)))
        if x2 - x1 < 0.01:
            x2 = min(1.0, x1 + 0.01)
        if y2 - y1 < 0.01:
            y2 = min(1.0, y1 + 0.01)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CropRect | None":
        if not data:
            return None
        try:
            return cls(
                x1=float(data["x1"]),
                y1=float(data["y1"]),
                x2=float(data["x2"]),
                y2=float(data["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class PhotoPreparation:
    image_id: str
    rotation: int = 0  # degrees clockwise: 0, 90, 180, 270
    crop: CropRect | None = None
    include_regions: list[CountRegion] = field(default_factory=list)
    exclude_regions: list[CountRegion] = field(default_factory=list)
    mask_mode: str = MASK_MODE_COUNT_FILTER
    overlap_rule: str = OVERLAP_RULE_CENTER
    minimum_detection_overlap: float = 0.5
    quality_warnings: list[str] = field(default_factory=list)
    is_reviewed: bool = False
    original_width: int = 0
    original_height: int = 0
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "image_id": self.image_id,
            "rotation": int(self.rotation) % 360,
            "crop": self.crop.to_dict() if self.crop else None,
            "include_regions": [r.to_dict() for r in self.include_regions],
            "exclude_regions": [r.to_dict() for r in self.exclude_regions],
            "mask_mode": self.mask_mode,
            "overlap_rule": self.overlap_rule,
            "minimum_detection_overlap": float(self.minimum_detection_overlap),
            "quality_warnings": list(self.quality_warnings),
            "is_reviewed": bool(self.is_reviewed),
            "original_width": int(self.original_width),
            "original_height": int(self.original_height),
        }
        if include_history:
            d["undo_stack"] = copy.deepcopy(self.undo_stack)
            d["redo_stack"] = copy.deepcopy(self.redo_stack)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, image_id: str = "") -> "PhotoPreparation":
        data = data or {}
        includes = [
            CountRegion.from_dict(r)
            for r in (data.get("include_regions") or [])
            if isinstance(r, dict)
        ]
        excludes = [
            CountRegion.from_dict(r)
            for r in (data.get("exclude_regions") or [])
            if isinstance(r, dict)
        ]
        return cls(
            image_id=str(data.get("image_id") or image_id or ""),
            rotation=int(data.get("rotation") or 0) % 360,
            crop=CropRect.from_dict(data.get("crop")),
            include_regions=includes,
            exclude_regions=excludes,
            mask_mode=str(data.get("mask_mode") or MASK_MODE_COUNT_FILTER),
            overlap_rule=str(data.get("overlap_rule") or OVERLAP_RULE_CENTER),
            minimum_detection_overlap=float(data.get("minimum_detection_overlap") or 0.5),
            quality_warnings=list(data.get("quality_warnings") or []),
            is_reviewed=bool(data.get("is_reviewed")),
            original_width=int(data.get("original_width") or 0),
            original_height=int(data.get("original_height") or 0),
            undo_stack=list(data.get("undo_stack") or []),
            redo_stack=list(data.get("redo_stack") or []),
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "rotation": self.rotation,
            "crop": self.crop.to_dict() if self.crop else None,
            "include_regions": [r.to_dict() for r in self.include_regions],
            "exclude_regions": [r.to_dict() for r in self.exclude_regions],
            "mask_mode": self.mask_mode,
        }

    def push_undo(self) -> None:
        self.undo_stack.append(self.snapshot())
        if len(self.undo_stack) > 40:
            self.undo_stack = self.undo_stack[-40:]
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.snapshot())
        snap = self.undo_stack.pop()
        self._apply_snapshot(snap)
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.snapshot())
        snap = self.redo_stack.pop()
        self._apply_snapshot(snap)
        return True

    def _apply_snapshot(self, snap: dict[str, Any]) -> None:
        self.rotation = int(snap.get("rotation") or 0) % 360
        self.crop = CropRect.from_dict(snap.get("crop"))
        self.include_regions = [
            CountRegion.from_dict(r) for r in (snap.get("include_regions") or [])
        ]
        self.exclude_regions = [
            CountRegion.from_dict(r) for r in (snap.get("exclude_regions") or [])
        ]
        self.mask_mode = str(snap.get("mask_mode") or MASK_MODE_COUNT_FILTER)

    def has_edits(self) -> bool:
        return bool(
            self.rotation
            or self.crop
            or self.include_regions
            or self.exclude_regions
            or self.mask_mode == MASK_MODE_HIDE_FROM_AI
        )


def default_preparation(image_id: str, *, width: int = 0, height: int = 0) -> PhotoPreparation:
    return PhotoPreparation(
        image_id=image_id,
        original_width=width,
        original_height=height,
    )


def preparation_status(prep: PhotoPreparation | dict[str, Any] | None) -> str:
    """Human-readable status for Add Photos cards."""
    if prep is None:
        return "Not reviewed"
    p = prep if isinstance(prep, PhotoPreparation) else PhotoPreparation.from_dict(prep)
    if p.original_width and p.original_height:
        if p.original_width < 32 or p.original_height < 32:
            return "Invalid image"
    if p.quality_warnings and any(
        "corrupt" in w.lower() or "unusable" in w.lower() for w in p.quality_warnings
    ):
        return "Invalid image"
    if p.include_regions or p.exclude_regions:
        return "Ready"
    if p.crop and not p.is_reviewed:
        return "Cropped"
    if p.is_reviewed or p.rotation:
        return "Ready"
    if p.quality_warnings:
        return "Needs attention"
    return "Not reviewed"


def preparation_status_detail(prep: PhotoPreparation) -> str:
    n_ex = len(prep.exclude_regions)
    n_in = len(prep.include_regions)
    status = preparation_status(prep)
    if status in {"Ready", "Has include areas", "Has exclude areas", "Cropped"} and (n_in or n_ex):
        parts = []
        if n_in:
            parts.append(f"{n_in} include area{'s' if n_in != 1 else ''}")
        if n_ex:
            parts.append(f"{n_ex} excluded area{'s' if n_ex != 1 else ''}")
        return " · ".join(parts)
    return ""


def open_source_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def apply_rotation(image: Image.Image, degrees: int) -> Image.Image:
    deg = int(degrees) % 360
    if deg == 0:
        return image
    # PIL rotate is counter-clockwise; user "rotate right" is clockwise.
    return image.rotate(-deg, expand=True)


def apply_crop(image: Image.Image, crop: CropRect | None) -> Image.Image:
    if crop is None:
        return image
    c = crop.to_dict()
    w, h = image.size
    x1 = int(round(c["x1"] * w))
    y1 = int(round(c["y1"] * h))
    x2 = int(round(c["x2"] * w))
    y2 = int(round(c["y2"] * h))
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return image
    return image.crop((x1, y1, x2, y2))


def build_prepared_image(data: bytes, prep: PhotoPreparation) -> Image.Image:
    """EXIF → rotation → crop. Does not bake area overlays or masks."""
    img = open_source_image(data)
    img = apply_rotation(img, prep.rotation)
    img = apply_crop(img, prep.crop)
    return img


def prepared_image_bytes(data: bytes, prep: PhotoPreparation, *, format: str = "JPEG") -> bytes:
    img = build_prepared_image(data, prep)
    buf = io.BytesIO()
    img.save(buf, format=format, quality=92)
    return buf.getvalue()


def prepared_content_hash(data: bytes, prep: PhotoPreparation) -> str:
    raw = prepared_image_bytes(data, prep)
    return hashlib.sha256(raw).hexdigest()


def preparation_cache_fingerprint(prep: PhotoPreparation) -> str:
    payload = prep.to_dict(include_history=False)
    blob = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


# --- Geometry -----------------------------------------------------------------


def _region_polygon_pixels(
    region: CountRegion, width: int, height: int
) -> list[tuple[int, int]]:
    pts = region.points_normalized
    if region.shape_type == "rectangle" or len(pts) == 2:
        xs = [_clamp01(p.x) for p in pts]
        ys = [_clamp01(p.y) for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return [
            (int(round(x1 * width)), int(round(y1 * height))),
            (int(round(x2 * width)), int(round(y1 * height))),
            (int(round(x2 * width)), int(round(y2 * height))),
            (int(round(x1 * width)), int(round(y2 * height))),
        ]
    return [
        (int(round(_clamp01(p.x) * width)), int(round(_clamp01(p.y) * height)))
        for p in pts
    ]


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray casting; polygon vertices in same coordinate space as x/y."""
    if len(polygon) < 3:
        if len(polygon) == 2:
            (x1, y1), (x2, y2) = polygon
            return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
        return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_region_normalized(nx: float, ny: float, region: CountRegion) -> bool:
    pts = region.points_normalized
    if not pts:
        return False
    if region.shape_type == "rectangle" or len(pts) == 2:
        xs = [_clamp01(p.x) for p in pts]
        ys = [_clamp01(p.y) for p in pts]
        return min(xs) <= nx <= max(xs) and min(ys) <= ny <= max(ys)
    poly = [(_clamp01(p.x), _clamp01(p.y)) for p in pts]
    return point_in_polygon(nx, ny, poly)


def point_eligible(
    nx: float,
    ny: float,
    prep: PhotoPreparation,
) -> tuple[bool, bool, bool]:
    """Return (eligible, inside_include, inside_exclude)."""
    includes = prep.include_regions
    excludes = prep.exclude_regions
    inside_exclude = any(point_in_region_normalized(nx, ny, r) for r in excludes)
    if includes:
        inside_include = any(point_in_region_normalized(nx, ny, r) for r in includes)
    else:
        inside_include = True
    eligible = inside_include and not inside_exclude
    return eligible, inside_include, inside_exclude


def detection_box_overlap_ratio(
    det: Detection,
    prep: PhotoPreparation,
    image_width: int,
    image_height: int,
    *,
    sample: int = 24,
) -> float:
    """Approximate fraction of detection box area inside the effective mask."""
    if image_width <= 0 or image_height <= 0:
        return 0.0
    x1 = max(0.0, min(float(det.x1), float(image_width)))
    x2 = max(0.0, min(float(det.x2), float(image_width)))
    y1 = max(0.0, min(float(det.y1), float(image_height)))
    y2 = max(0.0, min(float(det.y2), float(image_height)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hits = 0
    total = 0
    for i in range(sample):
        for j in range(sample):
            px = x1 + (i + 0.5) / sample * (x2 - x1)
            py = y1 + (j + 0.5) / sample * (y2 - y1)
            nx, ny = px / image_width, py / image_height
            eligible, _, _ = point_eligible(nx, ny, prep)
            total += 1
            if eligible:
                hits += 1
    return hits / float(total) if total else 0.0


def evaluate_detection_against_preparation(
    det: Detection,
    prep: PhotoPreparation,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if image_width <= 0 or image_height <= 0:
        return {
            "inside_include_area": True,
            "inside_exclude_area": False,
            "region_overlap_ratio": 1.0,
            "excluded_by_region": False,
            "region_exclusion_reason": None,
        }
    nx = float(det.center_x) / float(image_width)
    ny = float(det.center_y) / float(image_height)
    eligible, inside_include, inside_exclude = point_eligible(nx, ny, prep)
    overlap = detection_box_overlap_ratio(det, prep, image_width, image_height)

    excluded = False
    reason = None
    rule = prep.overlap_rule or OVERLAP_RULE_CENTER
    if rule == OVERLAP_RULE_MIN_IOU:
        if overlap < float(prep.minimum_detection_overlap):
            excluded = True
            reason = (
                f"Detection overlap {overlap:.2f} below minimum "
                f"{prep.minimum_detection_overlap:.2f}"
            )
    else:
        if not eligible:
            excluded = True
            if inside_exclude:
                reason = "Detection center is inside an exclude area"
            elif prep.include_regions and not inside_include:
                reason = "Detection center is outside all include areas"
            else:
                reason = "Detection outside effective count area"

    return {
        "inside_include_area": inside_include,
        "inside_exclude_area": inside_exclude,
        "region_overlap_ratio": round(overlap, 4),
        "excluded_by_region": excluded,
        "region_exclusion_reason": reason,
    }


def apply_preparation_to_detections(
    detections: Iterable[Detection],
    prep: PhotoPreparation | None,
    image_width: int,
    image_height: int,
) -> tuple[list[Detection], list[Detection], int]:
    """Annotate detections with region fields; return (all, counted, excluded_n).

    Region-excluded detections remain in ``all`` with ``excluded_by_region=True``
    and ``included_in_count=False``. They are omitted from the counted list used
    for ``final_count``.
    """
    all_dets: list[Detection] = []
    counted: list[Detection] = []
    excluded_n = 0
    for det in detections:
        data = det.to_dict()
        if prep is None or (
            not prep.include_regions and not prep.exclude_regions
        ):
            data.update(
                {
                    "inside_include_area": True,
                    "inside_exclude_area": False,
                    "region_overlap_ratio": 1.0,
                    "excluded_by_region": False,
                    "region_exclusion_reason": None,
                    "included_in_count": True,
                }
            )
            clone = Detection(**data)
            all_dets.append(clone)
            counted.append(clone)
            continue
        meta = evaluate_detection_against_preparation(
            det, prep, image_width, image_height
        )
        data.update(meta)
        if meta["excluded_by_region"]:
            data["included_in_count"] = False
            excluded_n += 1
            clone = Detection(**data)
            all_dets.append(clone)
        else:
            data["included_in_count"] = True
            clone = Detection(**data)
            all_dets.append(clone)
            counted.append(clone)
    return all_dets, counted, excluded_n


def filter_inference_result_by_preparation(
    result: Any,
    prep: PhotoPreparation | None,
) -> Any:
    """Mutate/rebuild InferenceResult counts after region filtering.

    Keeps region-excluded detections in ``result.detections`` for Review.
    ``final_count`` reflects only non-excluded detections.
    """
    if result is None or prep is None:
        return result
    if not prep.include_regions and not prep.exclude_regions:
        # Still stamp fields for consistency
        w = getattr(result, "prepared_width", None) or prep.original_width
        h = getattr(result, "prepared_height", None) or prep.original_height
        # Prefer dimensions from first detection space: use annotated image size if known
        all_dets, counted, _ = apply_preparation_to_detections(
            result.detections, None, max(1, int(w or 1)), max(1, int(h or 1))
        )
        result.detections = all_dets
        result.final_count = len(counted)
        return result

    # Image space for detections = prepared image dimensions
    # Infer from detection extents or prep metadata
    width = int(getattr(result, "image_width", 0) or 0)
    height = int(getattr(result, "image_height", 0) or 0)
    if not width or not height:
        if result.detections:
            width = int(
                max(max(d.x2, d.center_x) for d in result.detections) + 1
            )
            height = int(
                max(max(d.y2, d.center_y) for d in result.detections) + 1
            )
        else:
            width = max(1, prep.original_width or 1)
            height = max(1, prep.original_height or 1)

    # Prefer explicit prepared dims when attached
    pw = int(getattr(result, "prepared_width", 0) or 0)
    ph = int(getattr(result, "prepared_height", 0) or 0)
    if pw and ph:
        width, height = pw, ph

    all_dets, counted, _excl = apply_preparation_to_detections(
        result.detections, prep, width, height
    )
    result.detections = all_dets
    # Keep raw_count as pre-region count; final_count is post-region
    result.final_count = len(counted)
    note = (
        f"{_excl} detection(s) excluded by photo preparation."
        if _excl
        else None
    )
    if note:
        warnings = list(result.warnings or [])
        if note not in warnings:
            warnings.append(note)
        result.warnings = warnings
    return result


def eligible_area_ratio(prep: PhotoPreparation, *, grid: int = 64) -> float:
    """Approximate fraction of prepared image that is eligible for counting."""
    hits = 0
    total = grid * grid
    for i in range(grid):
        for j in range(grid):
            nx = (i + 0.5) / grid
            ny = (j + 0.5) / grid
            eligible, _, _ = point_eligible(nx, ny, prep)
            if eligible:
                hits += 1
    return hits / float(total)


def create_rectangle_region(
    region_type: RegionType,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    label: str = "",
) -> CountRegion:
    xa, xb = sorted((_clamp01(x1), _clamp01(x2)))
    ya, yb = sorted((_clamp01(y1), _clamp01(y2)))
    if xb - xa < 0.005:
        xb = min(1.0, xa + 0.005)
    if yb - ya < 0.005:
        yb = min(1.0, ya + 0.005)
    return CountRegion(
        region_id=uuid.uuid4().hex[:12],
        region_type=region_type,
        shape_type="rectangle",
        points_normalized=[
            RegionPoint(xa, ya),
            RegionPoint(xb, yb),
        ],
        label=label,
    )


def create_polygon_region(
    region_type: RegionType,
    points: list[tuple[float, float]],
    *,
    shape_type: ShapeType = "polygon",
    label: str = "",
) -> CountRegion:
    pts = [RegionPoint(_clamp01(x), _clamp01(y)) for x, y in points]
    return CountRegion(
        region_id=uuid.uuid4().hex[:12],
        region_type=region_type,
        shape_type=shape_type,
        points_normalized=pts,
        label=label,
    )


def relabel_regions(prep: PhotoPreparation) -> None:
    for i, r in enumerate(prep.include_regions, start=1):
        r.label = f"Include {i}"
        r.updated_at = _utc_now()
    for i, r in enumerate(prep.exclude_regions, start=1):
        r.label = f"Exclude {i}"
        r.updated_at = _utc_now()


# --- Masking / overlays -------------------------------------------------------


def build_effective_mask(
    width: int,
    height: int,
    prep: PhotoPreparation,
) -> Image.Image:
    """L mode mask: 255 = eligible, 0 = ignored."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if prep.include_regions:
        for r in prep.include_regions:
            poly = _region_polygon_pixels(r, width, height)
            if len(poly) >= 2:
                draw.polygon(poly, fill=255)
    else:
        draw.rectangle([0, 0, width, height], fill=255)
    for r in prep.exclude_regions:
        poly = _region_polygon_pixels(r, width, height)
        if len(poly) >= 2:
            draw.polygon(poly, fill=0)
    return mask


def mask_excluded_for_inference(
    image: Image.Image,
    prep: PhotoPreparation,
    *,
    fill_mode: str = "neutral_gray",
) -> Image.Image:
    """Return a copy with excluded (and outside-include) pixels filled.

    Does not modify ``image``.
    """
    base = image.copy().convert("RGB")
    w, h = base.size
    mask = build_effective_mask(w, h, prep)
    # Invert: pixels to hide are where mask == 0
    hide = Image.eval(mask, lambda p: 255 if p == 0 else 0)
    if fill_mode == "blur":
        filler = base.filter(ImageFilter.GaussianBlur(radius=max(8, min(w, h) // 40)))
    else:
        # Sample mean color for a less harsh edge than pure black
        try:
            import numpy as np

            arr = np.asarray(base)
            mean = tuple(int(x) for x in arr.mean(axis=(0, 1)))
        except Exception:  # noqa: BLE001
            mean = (160, 160, 160)
        filler = Image.new("RGB", (w, h), mean)
    return Image.composite(filler, base, hide)


def render_count_area_preview(
    image: Image.Image,
    prep: PhotoPreparation,
    *,
    dim_ignored: bool = True,
) -> Image.Image:
    """Overlay green includes / red excludes; dim ignored regions."""
    base = image.copy().convert("RGBA")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if dim_ignored:
        mask = build_effective_mask(w, h, prep)
        dim = Image.new("RGBA", (w, h), (40, 44, 52, 110))
        # Keep eligible fully visible: composite dim only where ineligible
        inv = Image.eval(mask, lambda p: 255 if p == 0 else 0)
        base = Image.composite(dim, base, inv)

    for i, r in enumerate(prep.include_regions, start=1):
        poly = _region_polygon_pixels(r, w, h)
        if len(poly) >= 2:
            draw.polygon(poly, fill=(46, 160, 67, 55), outline=(46, 160, 67, 220))
            label = r.label or f"Include {i}"
            x, y = poly[0]
            draw.text((x + 4, y + 4), label, fill=(20, 90, 40, 255))

    for i, r in enumerate(prep.exclude_regions, start=1):
        poly = _region_polygon_pixels(r, w, h)
        if len(poly) >= 2:
            draw.polygon(poly, fill=(229, 83, 75, 55), outline=(229, 83, 75, 220))
            label = r.label or f"Exclude {i}"
            x, y = poly[0]
            draw.text((x + 4, y + 4), label, fill=(120, 30, 30, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def render_regions_on_image(
    image: Image.Image,
    prep: PhotoPreparation,
) -> Image.Image:
    return render_count_area_preview(image, prep, dim_ignored=False)


def summary_dict(prep: PhotoPreparation) -> dict[str, Any]:
    ratio = eligible_area_ratio(prep)
    return {
        "include_areas": len(prep.include_regions),
        "exclude_areas": len(prep.exclude_regions),
        "eligible_image_area_pct": round(ratio * 100.0, 1),
        "rotation": int(prep.rotation) % 360,
        "crop_applied": prep.crop is not None,
        "status": preparation_status(prep),
        "mask_mode": prep.mask_mode,
        "is_reviewed": prep.is_reviewed,
        "explanation": (
            "Green areas will be analyzed. Red areas will be ignored."
            if (prep.include_regions or prep.exclude_regions)
            else "Entire image will be analyzed."
        ),
    }


def persistable_preparation(
    prep: PhotoPreparation,
    *,
    source: str,
    original_hash: str,
    prepared_hash: str,
) -> dict[str, Any]:
    d = prep.to_dict(include_history=False)
    d.update(
        {
            "source": source,
            "original_hash": original_hash,
            "prepared_image_hash": prepared_hash,
            "effective_area_percentage": summary_dict(prep)["eligible_image_area_pct"],
            "area_application_mode": prep.mask_mode,
        }
    )
    return d


def aspects_compatible(
    w1: int, h1: int, w2: int, h2: int, *, tol: float = 0.08
) -> bool:
    if min(w1, h1, w2, h2) <= 0:
        return False
    a1 = w1 / float(h1)
    a2 = w2 / float(h2)
    return abs(a1 - a2) / max(a1, a2) <= tol


def copy_preparation_geometry(
    source: PhotoPreparation, target_image_id: str
) -> PhotoPreparation:
    """Copy regions/transforms to another image id (caller checks compatibility)."""
    cloned = PhotoPreparation.from_dict(source.to_dict(include_history=False), image_id=target_image_id)
    cloned.image_id = target_image_id
    cloned.is_reviewed = False
    cloned.undo_stack = []
    cloned.redo_stack = []
    # Regenerate region ids
    for r in cloned.include_regions + cloned.exclude_regions:
        r.region_id = uuid.uuid4().hex[:12]
    relabel_regions(cloned)
    return cloned
