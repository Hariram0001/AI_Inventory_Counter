"""Classical fence-picket counter (local, not Roboflow).

Best for dog-ear / pointed picket silhouettes against a plain background.
Returns one detection box per estimated picket. This is experimental and can
miscount flat privacy panels, angled photos, or busy backgrounds.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from PIL import Image

from schemas import Detection


def _smooth(values: np.ndarray, kernel: int) -> np.ndarray:
    k = max(3, int(kernel) | 1)
    window = np.ones(k, dtype=float) / float(k)
    return np.convolve(values.astype(float), window, mode="same")


def detect_fence_pickets(
    image: Image.Image,
    *,
    source_image: str = "image",
    source_model: str = "Local Picket Counter",
) -> tuple[list[Detection], list[str]]:
    """
    Detect repeating vertical pickets from the top silhouette / upper band.

    Returns (detections, warnings).
    """
    warnings: list[str] = [
        "Local Picket Counter uses classical image analysis, not the Roboflow API.",
        "Works best on pointed (dog-ear) picket fences with a plain background.",
    ]
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    height, width = arr.shape[:2]
    if width < 32 or height < 32:
        warnings.append("Image too small for reliable picket counting.")
        return [], warnings

    gray = arr.mean(axis=2)
    # Upper band emphasizes pointed tops against bright sky/background.
    band_h = max(8, height // 3)
    band = gray[:band_h, :]
    score = 255.0 - band.mean(axis=0)
    smooth = _smooth(score, max(3, width // 70))

    thr = float(smooth.mean() + 0.12 * (smooth.max() - smooth.mean()))
    min_sep = max(6, width / 36.0)
    peaks: list[int] = []
    for i in range(2, width - 2):
        if (
            smooth[i] >= thr
            and smooth[i] >= smooth[i - 1]
            and smooth[i] >= smooth[i + 1]
            and smooth[i] >= smooth[i - 2]
            and smooth[i] >= smooth[i + 2]
        ):
            if not peaks or (i - peaks[-1]) >= min_sep:
                peaks.append(i)

    # Require a zig-zag top silhouette so flat privacy fences are skipped.
    mask = gray < np.percentile(gray, 72)
    tops = np.full(width, float(height), dtype=float)
    for x in range(width):
        ys = np.where(mask[:, x])[0]
        if len(ys):
            tops[x] = float(ys.min())
    wood = np.where(tops < height * 0.92)[0]
    if len(wood) < 16:
        warnings.append("Could not isolate fence boards from the background.")
        return [], warnings
    x0, x1 = int(wood[0]), int(wood[-1])
    top_seg = _smooth(tops, max(3, (x1 - x0) // 50))[x0 : x1 + 1]
    if float(top_seg.std()) < max(1.2, height * 0.008):
        warnings.append(
            "Fence top looks flat — picket tip counting is not reliable for this photo. "
            "Use YOLO-World for whole-fence detection, or photograph pointed pickets clearly."
        )
        return [], warnings

    if len(peaks) < 2:
        warnings.append("Not enough repeating picket tips were found.")
        return [], warnings

    # Trim weak edge peaks
    margin = max(2, int((x1 - x0) * 0.01))
    peaks = [p for p in peaks if x0 + margin <= p <= x1 - margin]
    if len(peaks) < 2:
        warnings.append("Picket tips were only found near image edges.")
        return [], warnings

    # Build vertical strip boxes between midpoints of adjacent tips
    bounds = [max(0, x0)]
    for i in range(len(peaks) - 1):
        bounds.append(int((peaks[i] + peaks[i + 1]) / 2))
    bounds.append(min(width - 1, x1))

    y1 = max(0, int(np.percentile(tops[x0 : x1 + 1], 5)) - 2)
    y2 = min(height - 1, int(height * 0.98))
    detections: list[Detection] = []
    for i in range(len(bounds) - 1):
        left = float(bounds[i])
        right = float(bounds[i + 1])
        if right - left < 3:
            continue
        conf = 0.55
        detections.append(
            Detection(
                detection_id=str(uuid.uuid4()),
                class_name="fence-picket",
                confidence=conf,
                x1=left,
                y1=float(y1),
                x2=right,
                y2=float(y2),
                center_x=(left + right) / 2.0,
                center_y=(y1 + y2) / 2.0,
                width=right - left,
                height=float(y2 - y1),
                source_model=source_model,
                source_image=source_image,
                scale_id="local_picket",
            )
        )

    warnings.append(
        f"Estimated {len(detections)} individual picket(s). "
        "Review numbered boxes carefully — classical counting can be off by 1–2."
    )
    return detections, warnings


def local_picket_response_payload(detections: list[Detection]) -> list[dict[str, Any]]:
    """Shape compatible with normalize_predictions / diagnostics."""
    preds = []
    for d in detections:
        preds.append(
            {
                "class": d.class_name,
                "confidence": d.confidence,
                "x": d.center_x,
                "y": d.center_y,
                "width": d.width,
                "height": d.height,
                "detection_id": d.detection_id,
            }
        )
    return [{"predictions": {"predictions": preds, "image": {}}}]
