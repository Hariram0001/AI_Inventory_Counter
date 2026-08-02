"""Image loading, EXIF correction, tiling, annotation, and coordinate mapping."""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

from schemas import Detection


@dataclass
class PreparedImage:
    """Image prepared for display and optional inference resize."""

    original: Image.Image
    inference: Image.Image
    image_name: str
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int
    scale_x: float
    scale_y: float
    used_resized_copy: bool
    content_hash: str
    temp_path: Path | None = None


@dataclass
class TileSpec:
    tile_id: str
    x_offset: int
    y_offset: int
    width: int
    height: int
    image: Image.Image


class ImageProcessingError(Exception):
    """User-facing image processing error."""


def safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned:
        cleaned = f"image_{uuid.uuid4().hex[:8]}.jpg"
    return cleaned[:180]


def compute_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_image_from_bytes(data: bytes, filename: str = "upload.jpg") -> PreparedImage:
    if not data:
        raise ImageProcessingError("Empty image upload.")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ImageProcessingError(
            "Could not open image. The file may be corrupted or an unsupported format."
        ) from exc

    original = img
    ow, oh = original.size
    inference = original
    used_resized = False

    from config import MAX_INFERENCE_DIMENSION

    max_dim = max(ow, oh)
    if max_dim > MAX_INFERENCE_DIMENSION:
        scale = MAX_INFERENCE_DIMENSION / float(max_dim)
        new_w = max(1, int(round(ow * scale)))
        new_h = max(1, int(round(oh * scale)))
        inference = original.resize((new_w, new_h), Image.Resampling.LANCZOS)
        used_resized = True

    iw, ih = inference.size
    scale_x = ow / float(iw) if iw else 1.0
    scale_y = oh / float(ih) if ih else 1.0

    return PreparedImage(
        original=original,
        inference=inference,
        image_name=safe_filename(filename),
        original_width=ow,
        original_height=oh,
        inference_width=iw,
        inference_height=ih,
        scale_x=scale_x,
        scale_y=scale_y,
        used_resized_copy=used_resized,
        content_hash=compute_bytes_hash(data),
    )


def validate_upload(
    data: bytes,
    filename: str,
    max_bytes: int,
) -> None:
    allowed = {".jpg", ".jpeg", ".png"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ImageProcessingError(
            f"Unsupported image type '{ext}'. Allowed: JPG, JPEG, PNG."
        )
    if len(data) > max_bytes:
        raise ImageProcessingError(
            f"Image exceeds maximum upload size of {max_bytes // (1024 * 1024)} MB."
        )
    # Quick corruption check
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001
        raise ImageProcessingError("Corrupted or unreadable image file.") from exc


def save_temp_image(image: Image.Image, directory: Path, prefix: str = "inf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{uuid.uuid4().hex}.jpg"
    image.save(path, format="JPEG", quality=92)
    return path


def map_box_to_original(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale_x: float,
    scale_y: float,
    original_width: int,
    original_height: int,
) -> tuple[float, float, float, float]:
    ox1 = x1 * scale_x
    oy1 = y1 * scale_y
    ox2 = x2 * scale_x
    oy2 = y2 * scale_y
    return clamp_box(ox1, oy1, ox2, oy2, original_width, original_height)


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    x1 = float(np.nan_to_num(x1, nan=0.0))
    y1 = float(np.nan_to_num(y1, nan=0.0))
    x2 = float(np.nan_to_num(x2, nan=0.0))
    y2 = float(np.nan_to_num(y2, nan=0.0))
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def estimate_tile_count(
    width: int,
    height: int,
    tile_size: int,
    overlap: float,
) -> int:
    if tile_size <= 0:
        return 0
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = _axis_starts(width, tile_size, stride)
    ys = _axis_starts(height, tile_size, stride)
    return len(xs) * len(ys)


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    # Deduplicate while preserving order
    seen: set[int] = set()
    out: list[int] = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def create_tiles(
    image: Image.Image,
    tile_size: int = 800,
    overlap: float = 0.25,
    max_tiles: int = 60,
) -> tuple[list[TileSpec], list[str]]:
    """
    Divide image into overlapping tiles.
    If tile count would exceed max_tiles, automatically increase tile size.
    """
    warnings: list[str] = []
    width, height = image.size
    effective_size = tile_size
    count = estimate_tile_count(width, height, effective_size, overlap)

    while count > max_tiles and effective_size < max(width, height):
        effective_size = min(max(width, height), int(effective_size * 1.25) + 64)
        count = estimate_tile_count(width, height, effective_size, overlap)
        warnings.append(
            f"Tile count exceeded {max_tiles}; increased tile size to {effective_size}."
        )

    if count > max_tiles:
        warnings.append(
            f"Still {count} tiles after upsizing. Consider using a smaller image. "
            f"Limiting to first {max_tiles} tiles."
        )

    stride = max(1, int(effective_size * (1.0 - overlap)))
    xs = _axis_starts(width, effective_size, stride)
    ys = _axis_starts(height, effective_size, stride)

    tiles: list[TileSpec] = []
    idx = 0
    for y in ys:
        for x in xs:
            if len(tiles) >= max_tiles:
                break
            w = min(effective_size, width - x)
            h = min(effective_size, height - y)
            crop = image.crop((x, y, x + w, y + h))
            tiles.append(
                TileSpec(
                    tile_id=f"tile_{idx}_{x}_{y}",
                    x_offset=x,
                    y_offset=y,
                    width=w,
                    height=h,
                    image=crop,
                )
            )
            idx += 1
        if len(tiles) >= max_tiles:
            break
    return tiles, warnings


def translate_tile_detections(
    detections: list[Detection],
    tile: TileSpec,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    original_width: int | None = None,
    original_height: int | None = None,
) -> list[Detection]:
    """Translate tile-local coordinates into original-image coordinates."""
    out: list[Detection] = []
    for det in detections:
        x1 = (det.x1 + tile.x_offset) * scale_x
        y1 = (det.y1 + tile.y_offset) * scale_y
        x2 = (det.x2 + tile.x_offset) * scale_x
        y2 = (det.y2 + tile.y_offset) * scale_y
        if original_width is not None and original_height is not None:
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, original_width, original_height)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        out.append(
            Detection(
                detection_id=det.detection_id,
                class_name=det.class_name,
                confidence=det.confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=(x1 + x2) / 2.0,
                center_y=(y1 + y2) / 2.0,
                width=width,
                height=height,
                source_model=det.source_model,
                source_image=det.source_image,
                tile_id=tile.tile_id,
                scale_id=det.scale_id,
                is_edge_detection=det.is_edge_detection,
                suspected_overlap=det.suspected_overlap,
                suspected_occlusion=det.suspected_occlusion,
                included_in_count=det.included_in_count,
                contributing_models=list(det.contributing_models),
                agreement_count=det.agreement_count,
                merged_from=list(det.merged_from),
            )
        )
    return out


def _get_font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:  # noqa: BLE001
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:  # noqa: BLE001
            return ImageFont.load_default()


# Focus color when navigating detections in Review (high-contrast on wood/yard photos).
SELECTED_OUTLINE_RGB = (220, 38, 38)
# Above this many detections, skip per-box class chips (keep thin boxes / numbers).
DENSE_LABEL_THRESHOLD = 12


def _format_confidence(confidence: float) -> str:
    """Stable percent label — avoids tiny/broken glyph chips when fonts fall back."""
    try:
        pct = int(round(float(confidence) * 100.0))
    except (TypeError, ValueError):
        pct = 0
    return f"{max(0, min(100, pct))}%"


def _draw_label_chip(
    draw: ImageDraw.ImageDraw,
    *,
    x1: float,
    y1: float,
    label: str,
    fill: tuple[int, int, int],
    text_color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    image_width: int,
    image_height: int,
) -> None:
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = max(28, int(bbox[2] - bbox[0]) + 14)
        th = max(18, int(bbox[3] - bbox[1]) + 8)
    except Exception:  # noqa: BLE001
        tw = 12 + 8 * max(1, len(label))
        th = 22

    lx1 = int(max(0, min(image_width - tw - 1, x1)))
    ly1 = int(y1) - th - 2
    if ly1 < 0:
        ly1 = int(min(image_height - th - 1, y1 + 2))
    draw.rectangle([lx1, ly1, lx1 + tw, ly1 + th], fill=fill, outline=(20, 20, 20), width=1)
    draw.text((lx1 + 7, ly1 + 3), label, fill=text_color, font=font)


def _draw_roboflow_label(
    draw: ImageDraw.ImageDraw,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    selected: bool,
    image_width: int,
    image_height: int,
    draw_label: bool = True,
) -> None:
    """Roboflow-style box + optional class/confidence label chip."""
    outline = SELECTED_OUTLINE_RGB if selected else color
    box_w = 5 if selected else 2
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=box_w)
    if selected:
        draw.rectangle(
            [x1 - 3, y1 - 3, x2 + 3, y2 + 3],
            outline=(255, 255, 255),
            width=2,
        )

    if not draw_label:
        return

    chip_fill = SELECTED_OUTLINE_RGB if selected else color
    _draw_label_chip(
        draw,
        x1=x1,
        y1=y1,
        label=label,
        fill=chip_fill,
        text_color=(255, 255, 255) if selected else text_color,
        font=font,
        image_width=image_width,
        image_height=image_height,
    )


def annotate_image(
    image: Image.Image,
    detections: list[Detection],
    model_name: str = "",
    *,
    style: str = "both",
    selected_detection_id: str | None = None,
    show_legend: bool = False,
    show_region_excluded: bool = False,
    muted_region_excluded: bool = True,
    solo: bool = False,
) -> Image.Image:
    """Draw detections with stable per-detection rainbow colors.

    style:
      - "boxes": bounding boxes + corner index badge
      - "markers": circular numbered markers at centers
      - "both": boxes and center markers (default)
      - "roboflow": Roboflow-style class-colored boxes + label chips (no center dots)

    When ``solo`` is True and ``selected_detection_id`` is set, only that one
    detection is drawn — no other numbers or boxes on the image.

    Region-excluded detections (``excluded_by_region``) are omitted unless
    ``show_region_excluded`` is True; they then use a muted style and keep any
    existing ``marker_number`` without renumbering included detections.
    """
    from detection_viz import (
        color_for_class,
        color_for_detection,
        contrasting_text_color,
    )

    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font_small = _get_font(14)
    font_num = _get_font(18)
    font_label = _get_font(15)
    font_focus = _get_font(20)
    style_key = (style or "both").strip().lower().replace(" ", "_")
    # Aliases for the UI label "Roboflow Labels"
    roboflow = style_key in {
        "roboflow",
        "roboflow_labels",
        "label_boxes",
        "supervision",
    }
    draw_boxes = roboflow or style_key in {"boxes", "both", "box", "bounding_boxes"}
    draw_markers = (not roboflow) and style_key in {
        "markers",
        "both",
        "marker",
        "numbered_markers",
    }
    width, height = canvas.size

    visible: list[Detection] = []
    for d in detections:
        if getattr(d, "excluded_by_region", False) and not show_region_excluded:
            continue
        if not getattr(d, "included_in_count", True) and not getattr(
            d, "excluded_by_region", False
        ):
            # Manually excluded from count elsewhere — skip unless caller included them
            continue
        visible.append(d)

    if solo and selected_detection_id:
        only = [d for d in visible if d.detection_id == selected_detection_id]
        if only:
            visible = only

    # Dense scenes: never paint dozens of class+confidence chips on top of each other.
    dense = (not solo) and len(visible) >= DENSE_LABEL_THRESHOLD

    count_only_rows = [
        d for d in visible if bool(getattr(d, "count_only", False))
    ]
    if count_only_rows:
        # Count-only OpenRouter results: draw a summary banner, never invent boxes.
        total = sum(max(0, int(getattr(d, "item_count", 1) or 1)) for d in count_only_rows)
        lines = [f"Count-only result · total {total}"]
        for d in count_only_rows[:8]:
            n = max(0, int(getattr(d, "item_count", 1) or 1))
            lines.append(
                f"• {n}× {d.class_name} ({_format_confidence(d.confidence)})"
            )
        if len(count_only_rows) > 8:
            lines.append(f"• … +{len(count_only_rows) - 8} more classes")
        pad = 8
        line_h = 18
        box_h = pad * 2 + line_h * len(lines)
        draw.rectangle([8, 8, min(width - 8, 8 + 420), 8 + box_h], fill=(20, 20, 20))
        for i, line in enumerate(lines):
            draw.text((16, 12 + i * line_h), line, fill=(240, 240, 240), font=font_small)
        # Skip geometry drawing for count-only rows.
        visible = [d for d in visible if not bool(getattr(d, "count_only", False))]

    for order_idx, det in enumerate(visible, start=1):
        region_excl = bool(getattr(det, "excluded_by_region", False))
        idx = int(getattr(det, "marker_number", None) or (0 if region_excl else order_idx))
        if region_excl and muted_region_excluded:
            color = (140, 140, 140)
            text_color = (255, 255, 255)
        elif roboflow:
            color = color_for_class(det.class_name, order_idx)
            text_color = contrasting_text_color(color)
        else:
            color = color_for_detection(det, order_idx)
            text_color = contrasting_text_color(color)
        selected = bool(
            selected_detection_id and det.detection_id == selected_detection_id
        )
        conf_txt = _format_confidence(det.confidence)

        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        # Degenerate boxes (common when VLM returns bad coords) — skip box draw.
        has_box = (x2 - x1) > 1.0 and (y2 - y1) > 1.0
        cx = max(0.0, min(float(width - 1), float(det.center_x)))
        cy = max(0.0, min(float(height - 1), float(det.center_y)))

        if roboflow and has_box and not getattr(det, "is_manual", False):
            # Dense: outline only for others; full chip only on the focused item.
            show_chip = selected or solo or not dense
            if show_chip:
                label = f"{det.class_name}  {conf_txt}"
                if idx:
                    label = f"#{idx}  {label}"
            else:
                label = ""
            _draw_roboflow_label(
                draw,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=label,
                color=color,
                text_color=text_color,
                font=font_focus if selected or solo else font_label,
                selected=selected,
                image_width=width,
                image_height=height,
                draw_label=bool(show_chip and label),
            )
        elif draw_boxes and has_box and not getattr(det, "is_manual", False):
            outline = SELECTED_OUTLINE_RGB if selected else color
            box_w = 5 if selected else (2 if dense else 3)
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=box_w)
            if selected:
                draw.rectangle(
                    [x1 - 3, y1 - 3, x2 + 3, y2 + 3],
                    outline=(255, 255, 255),
                    width=2,
                )
            badge = f"{idx}"
            bx1, by1 = int(x1), int(max(0, y1 - 28))
            badge_fill = SELECTED_OUTLINE_RGB if selected else color
            tw = 10 + 12 * len(badge)
            draw.rectangle([bx1, by1, bx1 + tw, by1 + 26], fill=badge_fill)
            draw.text((bx1 + 4, by1 + 2), badge, fill=(255, 255, 255), font=font_num)
            # Full class + confidence only when focused or the scene is sparse.
            if selected or solo or not dense:
                label = f"{det.class_name}  {conf_txt}"
                flags = []
                if det.suspected_overlap:
                    flags.append("OV")
                if det.suspected_occlusion:
                    flags.append("OC")
                if flags:
                    label = f"{label} [{'+'.join(flags)}]"
                _draw_label_chip(
                    draw,
                    x1=x1 + tw + 2,
                    y1=y1,
                    label=label,
                    fill=badge_fill if selected else (30, 30, 30),
                    text_color=(255, 255, 255),
                    font=font_focus if selected or solo else font_small,
                    image_width=width,
                    image_height=height,
                )

        if draw_markers or (roboflow and not has_box):
            # Markers mode, or Roboflow fallback when a detection has no usable box.
            base_r = max(10, min(18, int(min(width, height) * 0.035)))
            radius = base_r + (5 if selected else 0)
            cx_i = int(max(radius + 3, min(width - radius - 4, cx)))
            cy_i = int(max(radius + 3, min(height - radius - 4, cy)))
            ring = SELECTED_OUTLINE_RGB if selected else (255, 255, 255)
            draw.ellipse(
                [
                    cx_i - radius - 3,
                    cy_i - radius - 3,
                    cx_i + radius + 3,
                    cy_i + radius + 3,
                ],
                outline=(20, 20, 20) if not selected else SELECTED_OUTLINE_RGB,
                width=4 if selected else 2,
            )
            fill = SELECTED_OUTLINE_RGB if selected else color
            draw.ellipse(
                [cx_i - radius, cy_i - radius, cx_i + radius, cy_i + radius],
                fill=fill,
                outline=ring,
                width=2,
            )
            badge = str(idx)
            if getattr(det, "is_manual", False):
                badge = f"{idx}*"
            tx = cx_i - (4 * len(badge.replace("*", ""))) - (2 if "*" in badge else 0)
            ty = cy_i - 9
            draw.text((tx, ty), badge, fill=(255, 255, 255), font=font_num)
            # Confidence beside the focused marker (readable, not stacked on every log).
            if selected or solo:
                _draw_label_chip(
                    draw,
                    x1=cx_i + radius + 6,
                    y1=cy_i - 8,
                    label=f"{det.class_name}  {conf_txt}",
                    fill=SELECTED_OUTLINE_RGB,
                    text_color=(255, 255, 255),
                    font=font_focus,
                    image_width=width,
                    image_height=height,
                )

    if show_legend:
        # Lightweight caption only — never a large black panel
        legend = f"View: {style_key}"
        if model_name:
            legend = f"{legend} · {model_name}"
        box_w = min(width - 12, 12 + 7 * len(legend))
        draw.rectangle([6, 6, 6 + box_w, 28], fill=(245, 245, 245), outline=(46, 160, 67))
        draw.text((12, 10), legend, fill=(40, 40, 40), font=font_small)

    return canvas


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def preview_resize(image: Image.Image, max_side: int = 1200) -> Image.Image:
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image.copy()
    scale = max_side / float(m)
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
