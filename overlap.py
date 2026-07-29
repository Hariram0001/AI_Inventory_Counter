"""Overlap metrics and duplicate-removal strategies."""

from __future__ import annotations

import math
import uuid

from schemas import Detection


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def box_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return box_area(ix1, iy1, ix2, iy2)


def iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection over Union."""
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = box_area(*a)
    area_b = box_area(*b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def ios(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection over Smaller area."""
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = box_area(*a)
    area_b = box_area(*b)
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def detection_box(det: Detection) -> tuple[float, float, float, float]:
    return (det.x1, det.y1, det.x2, det.y2)


def normalized_center_distance(a: Detection, b: Detection) -> float:
    """Distance between centers relative to average width/height."""
    dx = abs(a.center_x - b.center_x)
    dy = abs(a.center_y - b.center_y)
    avg_w = (max(a.width, 1e-6) + max(b.width, 1e-6)) / 2.0
    avg_h = (max(a.height, 1e-6) + max(b.height, 1e-6)) / 2.0
    return math.sqrt((dx / avg_w) ** 2 + (dy / avg_h) ** 2)


def aspect_ratio_similarity(a: Detection, b: Detection) -> float:
    """1.0 = identical aspect ratios; 0.0 = very different."""
    ra = a.aspect_ratio
    rb = b.aspect_ratio
    if ra <= 0 or rb <= 0:
        return 0.0
    ratio = min(ra, rb) / max(ra, rb)
    return max(0.0, min(1.0, ratio))


def classes_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True
    return a.strip().lower() == b.strip().lower()


def _has_cross_source_signal(a: Detection, b: Detection) -> bool:
    """True when detections likely come from different inference passes."""
    if a.tile_id and b.tile_id and a.tile_id != b.tile_id:
        return True
    if a.scale_id and b.scale_id and a.scale_id != b.scale_id:
        return True
    if a.source_model and b.source_model and a.source_model != b.source_model:
        return True
    if bool(a.tile_id) != bool(b.tile_id):
        return True
    if bool(a.scale_id) != bool(b.scale_id):
        return True
    return False


def apply_nms(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Non-Maximum Suppression by class."""
    kept: list[Detection] = []
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        key = (det.class_name or "").lower()
        by_class.setdefault(key, []).append(det)

    for group in by_class.values():
        ordered = sorted(group, key=lambda d: _safe_float(d.confidence), reverse=True)
        suppressed = [False] * len(ordered)
        for i, det_i in enumerate(ordered):
            if suppressed[i]:
                continue
            kept.append(det_i)
            for j in range(i + 1, len(ordered)):
                if suppressed[j]:
                    continue
                if iou(detection_box(det_i), detection_box(ordered[j])) >= iou_threshold:
                    suppressed[j] = True
    return kept


def _merge_group(group: list[Detection]) -> Detection:
    """Confidence-weighted merge of overlapping detections."""
    weights = [max(_safe_float(d.confidence), 1e-6) for d in group]
    total_w = sum(weights)
    x1 = sum(d.x1 * w for d, w in zip(group, weights)) / total_w
    y1 = sum(d.y1 * w for d, w in zip(group, weights)) / total_w
    x2 = sum(d.x2 * w for d, w in zip(group, weights)) / total_w
    y2 = sum(d.y2 * w for d, w in zip(group, weights)) / total_w
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    best = max(group, key=lambda d: _safe_float(d.confidence))
    conf = max(_safe_float(d.confidence) for d in group)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    models = sorted({d.source_model for d in group if d.source_model})
    merged_ids = [d.detection_id for d in group]
    return Detection(
        detection_id=str(uuid.uuid4()),
        class_name=best.class_name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x=(x1 + x2) / 2.0,
        center_y=(y1 + y2) / 2.0,
        width=width,
        height=height,
        source_model="+".join(models) if models else best.source_model,
        source_image=best.source_image,
        tile_id=best.tile_id,
        scale_id=best.scale_id,
        is_edge_detection=any(d.is_edge_detection for d in group),
        suspected_overlap=False,
        suspected_occlusion=any(d.suspected_occlusion for d in group),
        contributing_models=models,
        agreement_count=max(1, len(models)),
        merged_from=merged_ids,
    )


def apply_nmm(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Non-Maximum Merge: cluster overlapping same-class boxes and merge."""
    remaining = sorted(detections, key=lambda d: _safe_float(d.confidence), reverse=True)
    merged: list[Detection] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        still: list[Detection] = []
        for det in remaining:
            if not classes_compatible(seed.class_name, det.class_name):
                still.append(det)
                continue
            overlaps = any(
                iou(detection_box(c), detection_box(det)) >= iou_threshold for c in cluster
            )
            if overlaps:
                cluster.append(det)
            else:
                still.append(det)
        remaining = still
        if len(cluster) == 1:
            merged.append(cluster[0])
        else:
            merged.append(_merge_group(cluster))
    return merged


def apply_conservative_dedup(
    detections: list[Detection],
    iou_threshold: float = 0.5,
    center_distance_threshold: float = 0.35,
    aspect_similarity_threshold: float = 0.7,
) -> list[Detection]:
    """
    Merge only when multiple signals suggest the same physical object.
    Recommended for densely stacked inventory.
    """
    remaining = sorted(detections, key=lambda d: _safe_float(d.confidence), reverse=True)
    kept: list[Detection] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        still: list[Detection] = []
        for det in remaining:
            if not classes_compatible(seed.class_name, det.class_name):
                still.append(det)
                continue

            box_iou = iou(detection_box(seed), detection_box(det))
            box_ios = ios(detection_box(seed), detection_box(det))
            center_dist = normalized_center_distance(seed, det)
            aspect_sim = aspect_ratio_similarity(seed, det)
            cross_source = _has_cross_source_signal(seed, det)

            is_narrow = False
            for d in (seed, det):
                ar = d.aspect_ratio
                if ar > 0 and (ar > 4.0 or ar < 0.25):
                    is_narrow = True
                    break
            center_limit = center_distance_threshold * (0.7 if is_narrow else 1.0)

            both_strong = (
                _safe_float(seed.confidence) >= 0.7
                and _safe_float(det.confidence) >= 0.7
                and center_dist > center_limit
            )
            if both_strong:
                still.append(det)
                continue

            likely_duplicate = (
                box_iou >= iou_threshold
                and center_dist <= center_limit
                and (
                    aspect_sim >= aspect_similarity_threshold
                    or box_ios >= 0.85
                )
                and (cross_source or box_iou >= max(iou_threshold, 0.65) or box_ios >= 0.9)
            )
            if likely_duplicate:
                cluster.append(det)
            else:
                still.append(det)

        remaining = still
        if len(cluster) == 1:
            kept.append(cluster[0])
        else:
            kept.append(_merge_group(cluster))
    return kept


def deduplicate(
    detections: list[Detection],
    strategy: str,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    strategy_key = (strategy or "conservative").strip().lower()
    if strategy_key in {"none", "none/debug", "debug"}:
        return list(detections)
    if strategy_key == "nms":
        return apply_nms(detections, iou_threshold=iou_threshold)
    if strategy_key == "nmm":
        return apply_nmm(detections, iou_threshold=iou_threshold)
    return apply_conservative_dedup(detections, iou_threshold=iou_threshold)


def strategy_comparison_counts(
    detections: list[Detection],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    return {
        "Raw": len(detections),
        "NMS": len(apply_nms(detections, iou_threshold)),
        "NMM": len(apply_nmm(detections, iou_threshold)),
        "Conservative": len(apply_conservative_dedup(detections, iou_threshold)),
    }


def mark_overlap_and_occlusion(
    detections: list[Detection],
    image_width: float,
    image_height: float,
    overlap_ios_threshold: float = 0.15,
) -> list[Detection]:
    """Flag suspected physical overlaps and occlusions after deduplication."""
    if not detections:
        return detections

    confidences = [_safe_float(d.confidence) for d in detections]
    avg_conf = sum(confidences) / max(len(confidences), 1)

    aspect_by_class: dict[str, list[float]] = {}
    for d in detections:
        if d.aspect_ratio > 0:
            aspect_by_class.setdefault(d.class_name.lower(), []).append(d.aspect_ratio)

    for i, a in enumerate(detections):
        a.suspected_overlap = False
        a.suspected_occlusion = False

        edge_margin = 3.0
        touches_edge = (
            a.x1 <= edge_margin
            or a.y1 <= edge_margin
            or a.x2 >= image_width - edge_margin
            or a.y2 >= image_height - edge_margin
        )
        a.is_edge_detection = touches_edge

        low_conf = _safe_float(a.confidence) < max(0.25, avg_conf * 0.55)

        peers = aspect_by_class.get(a.class_name.lower(), [])
        unusual_aspect = False
        if peers and a.aspect_ratio > 0:
            median = sorted(peers)[len(peers) // 2]
            if median > 0:
                ratio = min(a.aspect_ratio, median) / max(a.aspect_ratio, median)
                unusual_aspect = ratio < 0.45

        tile_boundary = bool(a.tile_id) and touches_edge

        for j, b in enumerate(detections):
            if i == j:
                continue
            inter = intersection_area(detection_box(a), detection_box(b))
            if inter <= 0:
                continue
            box_ios = ios(detection_box(a), detection_box(b))
            center_dist = normalized_center_distance(a, b)

            if box_ios >= overlap_ios_threshold and center_dist > 0.25:
                a.suspected_overlap = True

            if box_ios >= 0.45 or (
                box_ios >= 0.25 and _safe_float(a.confidence) < _safe_float(b.confidence)
            ):
                a.suspected_occlusion = True

        if touches_edge or low_conf or unusual_aspect or tile_boundary:
            if touches_edge or unusual_aspect or (low_conf and a.suspected_overlap):
                a.suspected_occlusion = True
            if tile_boundary and a.suspected_overlap:
                a.suspected_occlusion = True

    return detections


def cluster_for_consensus(
    detections: list[Detection],
    iou_threshold: float = 0.45,
    center_distance_threshold: float = 0.4,
) -> list[list[Detection]]:
    """Cluster detections that likely represent the same physical object."""
    remaining = list(detections)
    clusters: list[list[Detection]] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            still: list[Detection] = []
            for det in remaining:
                matches = False
                for member in cluster:
                    if not classes_compatible(member.class_name, det.class_name):
                        continue
                    if (
                        iou(detection_box(member), detection_box(det)) >= iou_threshold
                        or ios(detection_box(member), detection_box(det)) >= 0.7
                    ) and normalized_center_distance(member, det) <= center_distance_threshold:
                        matches = True
                        break
                if matches:
                    cluster.append(det)
                    changed = True
                else:
                    still.append(det)
            remaining = still
        clusters.append(cluster)
    return clusters


def build_consensus_detections(
    model_detections: dict[str, list[Detection]],
    min_agreement: int = 2,
    iou_threshold: float = 0.45,
) -> tuple[list[Detection], int, int]:
    """
    Build consensus detections across models.
    Returns (consensus_list, multi_model_count, single_model_count).
    """
    flat: list[Detection] = []
    for model_name, dets in model_detections.items():
        for d in dets:
            if not d.source_model:
                d.source_model = model_name
            flat.append(d)

    clusters = cluster_for_consensus(flat, iou_threshold=iou_threshold)
    consensus: list[Detection] = []
    multi = 0
    single = 0

    for cluster in clusters:
        models = sorted({d.source_model for d in cluster if d.source_model})
        agreement = len(models)
        if agreement < min_agreement:
            if min_agreement <= 1:
                best = max(cluster, key=lambda d: _safe_float(d.confidence))
                best.agreement_count = agreement
                best.contributing_models = models
                consensus.append(best)
                single += 1
            continue

        merged = _merge_group(cluster)
        merged.agreement_count = agreement
        merged.contributing_models = models
        per_model_best: dict[str, float] = {}
        for d in cluster:
            per_model_best[d.source_model] = max(
                per_model_best.get(d.source_model, 0.0), _safe_float(d.confidence)
            )
        if per_model_best:
            merged.confidence = sum(per_model_best.values()) / len(per_model_best)
        consensus.append(merged)
        if agreement >= 2:
            multi += 1
        else:
            single += 1

    return consensus, multi, single
