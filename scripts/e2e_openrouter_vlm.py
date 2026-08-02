"""Real end-to-end OpenRouter VLM inference using the Streamlit adapter path.

Never prints API keys. Writes a sanitized report under data/debug/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

config.reload_settings()

from image_processing import load_image_from_bytes
from model_adapters import InferenceOptions, get_adapter
from openrouter_runtime import get_openrouter_inference_key
from openrouter_vlm import call_openrouter_vlm, configured_openrouter_model_id
from schemas import ModelConfig


def _openrouter_model() -> ModelConfig:
    from model_registry import load_models_from_file

    for model in load_models_from_file():
        if "openrouter" in (model.provider or "").lower() or model.requires_user_api_key:
            if "OpenRouter" in model.name or "Luna" in model.name:
                return model
    return ModelConfig(
        name="OpenRouter VLM Detector",
        kind="workflow",
        enabled=True,
        workspace_name="hariram-s-mzhvc",
        workflow_id="playground-gpt-5-6-luna-od",
        key="workflow:hariram-s-mzhvc/playground-gpt-5-6-luna-od",
        provider="openrouter",
        requires_user_api_key=True,
        supports_prompt=True,
        dynamic_classes=True,
        allowed_classes=[],
    )


def main() -> int:
    image_path = ROOT / "assets" / "sample_images" / "fence_picket_panel_01.jpg"
    if not image_path.exists():
        print(f"MISSING_IMAGE: {image_path}")
        return 2
    image_bytes = image_path.read_bytes()
    key = get_openrouter_inference_key()
    if not key:
        print("MISSING_KEY: no verified OpenRouter deployment key")
        return 3

    model_id = configured_openrouter_model_id()
    classes = ["fence panel", "fence picket", "wood panel"]

    print("=== Direct call_openrouter_vlm (raw shape) ===")
    raw = call_openrouter_vlm(
        api_key=key,
        image_bytes=image_bytes,
        class_names=classes,
        model_id=model_id,
        image_name=image_path.name,
    )
    tech = raw.technical.to_public_dict()
    print("http_success:", bool(raw.ok))
    print("http_status:", tech.get("http_status"))
    print("response_type:", tech.get("response_type"))
    print("top_level_keys:", tech.get("top_level_keys"))
    print("usage:", tech.get("usage"))
    print("parser_stage:", tech.get("parser_stage"))
    print("retryable:", tech.get("retryable"))
    print("selected_model:", tech.get("selected_model"))
    preview = str(tech.get("response_preview") or "")[:400]
    print("response_preview:", preview)
    if raw.parsed:
        print("parsed_total:", raw.parsed.total_count)
        print("parsed_detections:", raw.parsed.detections)
        print("parsed_warnings:", raw.parsed.warnings)
        print("parsed_summary:", raw.parsed.summary)
    else:
        print("error_message:", raw.error_message)

    print("\n=== Adapter path (same as Streamlit Analyze) ===")
    prepared = load_image_from_bytes(image_bytes, filename=image_path.name)
    adapter = get_adapter(_openrouter_model(), model_api_key=key)
    options = InferenceOptions(
        prompt=", ".join(classes),
        confidence_threshold=0.25,
        inference_mode="Whole-image inference",
    )
    mir = adapter.predict(prepared, options)
    print("success:", mir.success)
    print("error_type:", mir.error_type)
    print("error_message:", mir.error_message)
    print("final_count:", mir.final_count)
    print("raw_count:", mir.raw_count)
    print("classes:", mir.classes)
    print("warnings:", mir.warnings)
    dets = []
    for d in mir.detections:
        dets.append(
            {
                "class_name": d.class_name,
                "confidence": d.confidence,
                "count_only": getattr(d, "count_only", False),
                "item_count": getattr(d, "item_count", 1),
                "box": [d.x1, d.y1, d.x2, d.y2],
            }
        )
    print("detections:", json.dumps(dets, indent=2))
    if mir.inference_result is not None:
        ir = mir.inference_result
        print("inference_mode:", ir.inference_mode)
        print("source:", ir.source)
        print("invocation_mode:", ir.invocation_mode)
        print("review_ready:", bool(ir.success and ir.request_completed))
    tech2 = dict(mir.technical_details or {})
    safe_tech = {
        k: v
        for k, v in tech2.items()
        if "api_key" not in str(k).lower() and "secret" not in str(k).lower()
    }
    print("technical_details:", json.dumps(safe_tech, indent=2, default=str)[:1200])

    out_dir = ROOT / "data" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "image": image_path.name,
        "model_id": model_id,
        "direct_ok": raw.ok,
        "direct_technical": tech,
        "direct_parsed": (
            {
                "total_count": raw.parsed.total_count,
                "detections": raw.parsed.detections,
                "warnings": raw.parsed.warnings,
                "summary": raw.parsed.summary,
            }
            if raw.parsed
            else None
        ),
        "adapter_success": mir.success,
        "adapter_final_count": mir.final_count,
        "adapter_detections": dets,
        "adapter_error": mir.error_message,
        "review_ready": bool(
            mir.success and mir.inference_result and mir.inference_result.success
        ),
    }
    report_path = out_dir / "openrouter_vlm_e2e_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0 if mir.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
