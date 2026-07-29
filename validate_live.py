#!/usr/bin/env python
"""Validate the live Roboflow path without printing secrets.

Supports direct models and Workflows (e.g. YOLO-World).

Usage:
  .\\.venv\\Scripts\\python.exe validate_live.py
  .\\.venv\\Scripts\\python.exe validate_live.py path\\to\\photo.jpg

Never prints ROBOFLOW_API_KEY. Saves an annotated preview and sanitized
raw response under data/.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

import config
from config import reload_settings
from detector import (
    DetectorError,
    RoboflowDetector,
    response_shape_summary,
    run_inference_on_prepared_image,
    sanitize_model_id,
    sanitize_payload_for_debug,
    save_last_live_response,
)
from image_processing import load_image_from_bytes, save_temp_image
from model_registry import get_enabled_valid_models, load_models_from_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("validate_live")

DEFAULT_LIVE_CONFIDENCE = 0.25


def _sanitize_id(value: str | None) -> str:
    if not value:
        return "(missing)"
    if "/" in value:
        return sanitize_model_id(value)
    if len(value) <= 4:
        return "***"
    return value[:3] + "***" + value[-2:]


def _pick_live_config():
    """Prefer an enabled Workflow, else an enabled direct model (non-demo)."""
    models = get_enabled_valid_models(load_models_from_file(), allow_demo_ids=False)
    workflows = [m for m in models if (m.kind or "").lower() == "workflow"]
    directs = [m for m in models if (m.kind or "").lower() == "model"]
    if workflows:
        return workflows[0]
    if directs:
        return directs[0]
    return None


def _load_or_make_image(path: Path | None) -> tuple[str, bytes, int, int]:
    if path and path.exists():
        data = path.read_bytes()
        name = path.name
    else:
        img = Image.new("RGB", (1024, 768), color=(190, 195, 200))
        draw = ImageDraw.Draw(img)
        for i, x in enumerate(range(80, 900, 110)):
            draw.rectangle([x, 120, x + 90, 520], outline=(70, 90, 70), width=3)
            draw.rectangle(
                [x + 8, 130, x + 82, 510],
                fill=(160 + i * 5, 165, 170),
            )
        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        name = "synthetic_yard.jpg"
        logger.info("No photo argument provided — using synthetic JPEG for smoke test.")

    prepared_probe = load_image_from_bytes(data, name)
    return name, data, prepared_probe.original_width, prepared_probe.original_height


def main() -> int:
    reload_settings()
    started = time.perf_counter()
    print("=== Live Roboflow validation ===")
    print(f"DEMO_MODE={config.DEMO_MODE}")
    print(f"API_KEY_CONFIGURED={bool(config.ROBOFLOW_API_KEY)}")
    print(f"API_URL={config.ROBOFLOW_API_URL}")
    print(f"CONFIDENCE_THRESHOLD={DEFAULT_LIVE_CONFIDENCE}")

    if config.DEMO_MODE:
        print("FAIL: DEMO_MODE is true. Set DEMO_MODE=false in .env for live validation.")
        return 2
    if not config.ROBOFLOW_API_KEY:
        print("FAIL: ROBOFLOW_API_KEY is missing or empty in .env")
        return 2

    model = _pick_live_config()
    if model is None:
        print(
            "FAIL: No enabled non-demo model/workflow in models.json. "
            "Enable a Workflow (YOLO-World) or a direct model_id."
        )
        return 2

    kind = (model.kind or "model").lower()
    print(f"CONFIG_NAME={model.name}")
    print(f"CONFIG_KIND={kind}")
    if kind == "workflow":
        print(f"WORKSPACE={model.workspace_name}")
        print(f"WORKFLOW_ID={model.workflow_id}")
        print(f"WORKSPACE_SANITIZED={_sanitize_id(model.workspace_name)}")
        print(f"WORKFLOW_ID_SANITIZED={_sanitize_id(model.workflow_id)}")
        print(f"IMAGE_INPUT_NAME={model.image_input_name or 'image'}")
        print(f"SUPPORTS_PROMPT={model.supports_prompt}")
    else:
        print(f"MODEL_ID_SANITIZED={sanitize_model_id(model.model_id)}")

    detector = RoboflowDetector(demo_mode=False)
    ok, msg = detector.test_connectivity()
    print(f"AUTH_OK={ok}")
    print(f"AUTH_MESSAGE={msg}")
    if not ok:
        return 3

    photo_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if photo_arg and not photo_arg.exists():
        print(f"FAIL: Photograph not found: {photo_arg}")
        return 2

    name, data, w, h = _load_or_make_image(photo_arg)
    print(f"PHOTO_NAME={name}")
    print(f"PHOTO_PATH={photo_arg.resolve() if photo_arg else '(synthetic)'}")
    print(f"PHOTO_BYTES={len(data)}")
    print(f"PHOTO_DIMENSIONS={w}x{h}")
    print(f"PHOTO_EXISTS={bool(photo_arg and photo_arg.exists())}")

    prepared = load_image_from_bytes(data, name)
    print(
        f"INFERENCE_DIMENSIONS={prepared.inference_width}x{prepared.inference_height}"
    )
    print(f"USED_RESIZED_COPY={prepared.used_resized_copy}")

    tmp = Path(tempfile.mkdtemp(prefix="aic_live_"))
    tmp_path = save_temp_image(prepared.inference, tmp)
    print(f"TEMP_IMAGE_PATH={tmp_path}")
    print(f"TEMP_IMAGE_EXISTS={tmp_path.exists()}")
    print(f"TEMP_IMAGE_BYTES={tmp_path.stat().st_size if tmp_path.exists() else 0}")
    print(f"REQUEST_INPUT_MAPPING={{{model.image_input_name or 'image'}: <temp_image_path>}}")
    print("REQUEST_SOURCE=live_roboflow")
    print(f"REQUEST_TIMESTAMP_UTC={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    # Dynamic inventory prompt (model name is generic; inventory supplies classes)
    from config import inventory_detection_prompt

    live_prompt = inventory_detection_prompt("Fence Panel")
    print(f"INVENTORY_KEY=Fence Panel")
    print(f"EFFECTIVE_DETECTION_QUERIES={live_prompt}")

    try:
        raw = detector.infer_image_path(
            model, str(tmp_path), prompt=live_prompt, confidence=DEFAULT_LIVE_CONFIDENCE
        )
    except DetectorError as exc:
        print(f"INFERENCE_ERROR={exc}")
        return 4
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
            tmp.rmdir()
        except Exception:
            pass

    if detector.last_source != "live_roboflow":
        print(f"FAIL: Expected live_roboflow source, got {detector.last_source}")
        return 5

    shape = response_shape_summary(raw)
    debug_path = save_last_live_response(raw)
    print("RESPONSE_SHAPE=" + json.dumps(shape))
    print(f"SOURCE_PROVEN={detector.last_source}")
    print(f"INVOCATION_MODE={detector.last_invocation_mode}")
    print(f"EMPTY_DRAFT_FALLBACK={detector.last_empty_draft_fallback}")
    print(f"RAW_RESPONSE_SAVED={debug_path}")

    result = run_inference_on_prepared_image(
        detector,
        prepared,
        model,
        confidence_threshold=DEFAULT_LIVE_CONFIDENCE,
        iou_threshold=0.5,
        inference_mode="Whole-image inference",
        deduplication_strategy="Conservative",
    )

    print(f"RAW_PREDICTIONS={shape['prediction_count']}")
    print(f"CLASS_NAMES={shape['class_names']}")
    print(f"NORMALIZED_COUNT={result.normalized_prediction_count}")
    print(f"FINAL_COUNT={result.final_count}")
    print(f"DUPLICATES_REMOVED={result.duplicates_removed}")
    print(f"OVERLAP_WARNINGS={result.suspected_overlap_count}")
    print(f"OCCLUSION_WARNINGS={result.suspected_occlusion_count}")
    print(f"PROCESSING_TIME_S={result.processing_time_seconds:.3f}")
    print(f"PIPELINE_SUCCESS={result.success}")
    print(f"PIPELINE_SOURCE={result.source}")
    print(f"ERROR_TYPE={result.error_type}")
    print(f"PREDICTIONS_FOUND={result.predictions_found}")
    if result.warnings:
        print("WARNINGS=" + " | ".join(result.warnings))

    if result.detections:
        d0 = result.detections[0]
        required_ok = all(
            v is not None
            for v in (
                d0.class_name,
                d0.confidence,
                d0.x1,
                d0.y1,
                d0.x2,
                d0.y2,
                d0.center_x,
                d0.center_y,
                d0.width,
                d0.height,
                d0.source_model,
            )
        )
        print(f"NORMALIZED_FIELDS_OK={required_ok}")
        print(
            f"TOP_DETECTION=class={d0.class_name} conf={d0.confidence:.4f} "
            f"box=({d0.x1:.1f},{d0.y1:.1f},{d0.x2:.1f},{d0.y2:.1f})"
        )
    else:
        print("NORMALIZED_FIELDS_OK=n/a (zero detections — inspect RAW_PREDICTIONS)")

    out_dir = config.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.annotated_image_bytes:
        out_path = out_dir / "live_validation_annotated.png"
        out_path.write_bytes(result.annotated_image_bytes)
        print(f"ANNOTATED_SAVED={out_path}")

    shape_path = out_dir / "last_live_response_shape.json"
    shape_path.write_text(json.dumps(shape, indent=2), encoding="utf-8")
    print(f"SHAPE_SAVED={shape_path}")

    # Also store a compact sanitized dump next to the full response
    compact_path = out_dir / "debug" / "last_live_response_compact.json"
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path.write_text(
        json.dumps(
            {
                "shape": shape,
                "source": detector.last_source,
                "invocation_mode": detector.last_invocation_mode,
                "empty_draft_fallback": detector.last_empty_draft_fallback,
                "sanitized_sample": sanitize_payload_for_debug(raw, max_depth=4),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"COMPACT_SAVED={compact_path}")
    print(f"TOTAL_WALL_TIME_S={time.perf_counter() - started:.3f}")
    print("PASS: Live Roboflow path validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
