"""Stable detection identity helpers for markers, review rows, and saves."""

from __future__ import annotations

import hashlib
from typing import Iterable

from schemas import Detection


def make_stable_detection_id(
    *,
    image_hash: str,
    model_key: str,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    raw_index: int,
) -> str:
    payload = (
        f"{image_hash}|{model_key}|{class_name}|"
        f"{round(x1, 1)}|{round(y1, 1)}|{round(x2, 1)}|{round(y2, 1)}|{raw_index}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assign_stable_detection_ids(
    detections: Iterable[Detection],
    *,
    image_hash: str,
    model_key: str,
) -> list[Detection]:
    out: list[Detection] = []
    for idx, det in enumerate(detections):
        new_id = make_stable_detection_id(
            image_hash=image_hash or det.source_image,
            model_key=model_key or det.source_model,
            class_name=det.class_name,
            x1=det.x1,
            y1=det.y1,
            x2=det.x2,
            y2=det.y2,
            raw_index=idx,
        )
        det.detection_id = new_id
        out.append(det)
    return out
