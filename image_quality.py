"""Lightweight local image-quality checks (no paid AI models)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MIN_DIMENSION = 64
LOW_RESOLUTION_MAX = 320
BLUR_VARIANCE_THRESHOLD = 45.0
OVEREXPOSE_MEAN = 235.0
UNDEREXPOSE_MEAN = 35.0
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "MPO"}


def assess_image_bytes(data: bytes, filename: str = "image.jpg") -> dict[str, Any]:
    """Return quality assessment. Noncritical issues are warnings, not blockers."""
    warnings: list[str] = []
    blocking = False
    width = 0
    height = 0
    fmt = None

    if not data:
        return {
            "ok": False,
            "blocking": True,
            "width": 0,
            "height": 0,
            "format": None,
            "warnings": ["Empty or corrupt image — the file has no data."],
            "metrics": {},
        }

    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        fmt = (img.format or "").upper() or None
        img = img.convert("RGB")
        width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {
            "ok": False,
            "blocking": True,
            "width": 0,
            "height": 0,
            "format": None,
            "warnings": [f"Unsupported or corrupt image: {exc}"],
            "metrics": {},
        }

    if fmt and fmt not in SUPPORTED_FORMATS and filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in {"jpg", "jpeg", "png"}:
            warnings.append(f"Unsupported format '{fmt or ext}'. Prefer JPG or PNG.")

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        blocking = True
        warnings.append(
            f"Image is unusable at {width}×{height} (minimum {MIN_DIMENSION}px)."
        )
    elif width < LOW_RESOLUTION_MAX or height < LOW_RESOLUTION_MAX:
        warnings.append(
            f"This photo is low resolution ({width}×{height}), which could reduce detection quality."
        )

    arr = np.asarray(img, dtype=np.float32)
    gray = arr.mean(axis=2)
    mean_lum = float(gray.mean())
    # Laplacian-like variance via simple second difference
    gy, gx = np.gradient(gray)
    blur_score = float(gx.var() + gy.var())

    if blur_score < BLUR_VARIANCE_THRESHOLD:
        warnings.append(
            "This photo may be blurry, which could reduce detection quality."
        )
    if mean_lum >= OVEREXPOSE_MEAN:
        warnings.append(
            "This photo may be overexposed (very bright), which could reduce detection quality."
        )
    elif mean_lum <= UNDEREXPOSE_MEAN:
        warnings.append(
            "This photo may be underexposed (very dark), which could reduce detection quality."
        )

    return {
        "ok": not blocking,
        "blocking": blocking,
        "width": width,
        "height": height,
        "format": fmt,
        "warnings": warnings,
        "metrics": {
            "mean_luminance": round(mean_lum, 2),
            "blur_score": round(blur_score, 2),
        },
    }
