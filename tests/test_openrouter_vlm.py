"""Unit tests for direct OpenRouter VLM object detection / parsing."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from openrouter_vlm import (
    OpenRouterVLMError,
    build_inventory_count_prompt,
    call_openrouter_vlm,
    detections_from_parsed,
    extract_message_content,
    parse_inventory_count_json,
)
from schemas import Detection


def test_prompt_requires_object_detection_boxes():
    prompt = build_inventory_count_prompt(
        ["fence panel", "gate"], image_width=800, image_height=600
    )
    assert "ONE entry per individual" in prompt
    assert "x1" in prompt and "y2" in prompt
    assert "800x600" in prompt
    assert "fence panel" in prompt


def test_prompt_requires_complete_individual_counting_for_any_class():
    for classes in (
        ["traffic cone", "road cone"],
        ["fence panel"],
        ["wooden pallet"],
        ["custom widget"],
    ):
        prompt = build_inventory_count_prompt(classes)
        assert "complete individual counting" in prompt
        assert "stacked" in prompt.lower()
        assert "scattered" in prompt.lower()
        assert "Do NOT treat a stack, pile, bundle, nest, or group as a single object" in prompt
        assert "ONE entry per individual physical item" in prompt
        assert "SEPARATE item type" in prompt


def test_valid_json_string_response_with_boxes():
    parsed = parse_inventory_count_json(
        json.dumps(
            {
                "detections": [
                    {
                        "class_name": "fence_panel",
                        "confidence": 0.9,
                        "x1": 10,
                        "y1": 20,
                        "x2": 110,
                        "y2": 220,
                    }
                ],
                "total_count": 1,
                "warnings": [],
                "summary": "one panel",
            }
        ),
        image_width=400,
        image_height=400,
    )
    assert parsed.total_count == 1
    assert parsed.detections[0]["class_name"] == "fence_panel"
    assert parsed.detections[0]["count_only"] is False
    dets = detections_from_parsed(
        parsed, model_name="OpenRouter VLM Detector", image_name="x.jpg"
    )
    assert len(dets) == 1
    assert not dets[0].count_only
    assert dets[0].width > 0 and dets[0].height > 0


def test_fenced_json_response():
    content = """Here you go:
```json
{"detections":[{"class_name":"box","confidence":0.7,"x1":1,"y1":2,"x2":40,"y2":50}],"total_count":1,"warnings":[],"summary":"ok"}
```
"""
    parsed = parse_inventory_count_json(content, image_width=100, image_height=100)
    assert parsed.total_count == 1


def test_content_block_array_response():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "detections": [
                                        {
                                            "class_name": "pallet",
                                            "confidence": 0.8,
                                            "bbox": [5, 5, 55, 55],
                                        }
                                    ],
                                    "total_count": 1,
                                    "warnings": [],
                                    "summary": "one",
                                }
                            ),
                        }
                    ]
                }
            }
        ]
    }
    text, stage = extract_message_content(payload)
    assert stage == "content_blocks"
    parsed = parse_inventory_count_json(text, image_width=100, image_height=100)
    assert parsed.total_count == 1
    assert parsed.detections[0]["x2"] == 55


def test_normalized_box_denormalized_to_pixels():
    parsed = parse_inventory_count_json(
        json.dumps(
            {
                "detections": [
                    {
                        "class_name": "box",
                        "confidence": 0.9,
                        "x1": 0.1,
                        "y1": 0.2,
                        "x2": 0.5,
                        "y2": 0.8,
                    }
                ],
                "total_count": 1,
                "warnings": [],
                "summary": "norm",
            }
        ),
        image_width=200,
        image_height=100,
    )
    d = parsed.detections[0]
    assert abs(d["x1"] - 20.0) < 0.01
    assert abs(d["y1"] - 20.0) < 0.01
    assert abs(d["x2"] - 100.0) < 0.01
    assert abs(d["y2"] - 80.0) < 0.01


def test_empty_response():
    with pytest.raises(OpenRouterVLMError):
        parse_inventory_count_json("")
    text, stage = extract_message_content({"choices": []})
    assert text == ""
    assert stage == "empty_choices"


def test_invalid_json():
    with pytest.raises(OpenRouterVLMError) as exc:
        parse_inventory_count_json("not-json {")
    assert "could not be parsed into a valid inventory count" in str(exc.value)


def test_provider_error_response(monkeypatch):
    class FakeResp:
        status_code = 500
        headers = {"content-type": "application/json"}
        text = '{"error":{"message":"upstream failed"}}'

        def json(self):
            return {"error": {"message": "upstream failed"}}

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    img = Image.new("RGB", (32, 32), color=(200, 180, 160))
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    out = call_openrouter_vlm(
        api_key="sk-or-v1-" + "e" * 32,
        image_bytes=buf.getvalue(),
        class_names=["box"],
        model_id="openai/gpt-5.6-luna",
    )
    assert not out.ok
    assert out.technical.http_status == 500
    assert out.technical.parser_stage == "provider_error"


def test_zero_detections():
    parsed = parse_inventory_count_json(
        '{"detections":[],"total_count":0,"warnings":[],"summary":"none"}'
    )
    assert parsed.total_count == 0
    dets = detections_from_parsed(
        parsed, model_name="OpenRouter VLM Detector", image_name="x.jpg"
    )
    assert dets == []


def test_multiple_individual_objects():
    parsed = parse_inventory_count_json(
        json.dumps(
            {
                "detections": [
                    {
                        "class_name": "fence_panel",
                        "confidence": 0.8,
                        "x1": 10,
                        "y1": 10,
                        "x2": 40,
                        "y2": 80,
                    },
                    {
                        "class_name": "gate",
                        "confidence": 0.6,
                        "x": 100,
                        "y": 50,
                        "width": 40,
                        "height": 60,
                    },
                ],
                "total_count": 2,
                "warnings": [],
                "summary": "mixed",
            }
        ),
        image_width=200,
        image_height=200,
    )
    dets = detections_from_parsed(
        parsed, model_name="OpenRouter VLM Detector", image_name="yard.jpg"
    )
    assert len(dets) == 2
    assert all(not d.count_only for d in dets)
    assert all(d.width > 0 and d.height > 0 for d in dets)
    assert all(d.item_count == 1 for d in dets)


def test_legacy_count_rows_expanded_without_boxes():
    parsed = parse_inventory_count_json(
        json.dumps(
            {
                "detections": [
                    {"class_name": "box", "count": 2, "confidence": 0.5},
                    {"class_name": "box", "count": 3, "confidence": 0.5},
                ],
                "total_count": 99,
                "warnings": [],
                "summary": "legacy",
            }
        )
    )
    assert parsed.total_count == 5
    assert any("no bounding box" in w.lower() for w in parsed.warnings)
    assert any("recalculated" in w.lower() for w in parsed.warnings)
    dets = detections_from_parsed(
        parsed, model_name="OpenRouter VLM Detector", image_name="x.jpg"
    )
    assert len(dets) == 5
    assert all(d.count_only for d in dets)


def test_invalid_confidence_and_unknown_fields_ignored():
    parsed = parse_inventory_count_json(
        json.dumps(
            {
                "detections": [
                    {
                        "class_name": "pole",
                        "confidence": 150,
                        "x1": 1,
                        "y1": 1,
                        "x2": 20,
                        "y2": 30,
                        "extra_field": "ok",
                    }
                ],
                "total_count": 1,
                "mystery": True,
                "warnings": "not-a-list",
                "summary": "ok",
            }
        ),
        image_width=100,
        image_height=100,
    )
    assert parsed.total_count == 1
    assert parsed.detections[0]["confidence"] == 0.5
    assert any("confidence" in w.lower() for w in parsed.warnings)


def test_detection_dataclass_supports_count_only():
    d = Detection(
        detection_id="1",
        class_name="box",
        confidence=0.5,
        x1=0,
        y1=0,
        x2=0,
        y2=0,
        center_x=0,
        center_y=0,
        width=0,
        height=0,
        source_model="m",
        source_image="i",
        count_only=True,
        item_count=7,
    )
    assert d.counted_items == 7


def test_adapter_predict_uses_direct_openrouter_not_workflow(monkeypatch):
    """Streamlit Analyze path must not route OpenRouter through Workflow parsing."""
    from types import SimpleNamespace

    from model_adapters import InferenceOptions, OpenRouterVLMAdapter
    from openrouter_vlm import OpenRouterVLMTechnicalDetails, ParsedInventoryCount
    from schemas import InferenceResult, ModelConfig

    called = {"direct": 0, "workflow": 0}

    def fake_run(**kwargs):
        called["direct"] += 1
        parsed = ParsedInventoryCount(
            detections=[
                {
                    "class_name": "box",
                    "confidence": 0.8,
                    "notes": "",
                    "x1": 1,
                    "y1": 2,
                    "x2": 30,
                    "y2": 40,
                    "count_only": False,
                    "item_count": 1,
                },
                {
                    "class_name": "box",
                    "confidence": 0.7,
                    "notes": "",
                    "x1": 50,
                    "y1": 2,
                    "x2": 80,
                    "y2": 40,
                    "count_only": False,
                    "item_count": 1,
                },
            ],
            total_count=2,
            warnings=[],
            summary="two",
        )
        ir = InferenceResult(
            image_name="x.jpg",
            model_name="OpenRouter VLM Detector",
            prompt="box",
            inference_mode="openrouter_vlm_detection",
            deduplication_strategy="None",
            detections=detections_from_parsed(
                parsed, model_name="OpenRouter VLM Detector", image_name="x.jpg"
            ),
            raw_count=2,
            final_count=2,
            duplicates_removed=0,
            avg_confidence=0.75,
            min_confidence=0.7,
            max_confidence=0.8,
            suspected_overlap_count=0,
            suspected_occlusion_count=0,
            processing_time_seconds=0.1,
            warnings=[],
            success=True,
            source="openrouter_vlm",
            request_completed=True,
            predictions_found=True,
            invocation_mode="openrouter_chat_completions",
        )
        return ir, OpenRouterVLMTechnicalDetails(
            selected_model="openai/gpt-5.6-luna",
            http_status=200,
            parser_stage="parsed_ok",
        )

    monkeypatch.setattr(
        "openrouter_vlm.run_openrouter_vlm_on_prepared_image", fake_run
    )
    monkeypatch.setattr(
        "detector.RoboflowDetector._run_byok_workflow",
        lambda *a, **k: called.__setitem__("workflow", called["workflow"] + 1),
    )

    model = ModelConfig(
        name="OpenRouter VLM Detector",
        kind="workflow",
        enabled=True,
        workspace_name="hariram-s-mzhvc",
        workflow_id="playground-gpt-5-6-luna-od",
        provider="openrouter",
        requires_user_api_key=True,
        supports_prompt=True,
        dynamic_classes=True,
    )
    adapter = OpenRouterVLMAdapter(model, model_api_key="sk-or-v1-" + "e" * 32)
    prepared = SimpleNamespace(
        image_name="x.jpg",
        content_hash="abc",
        inference=None,
        original=None,
        working_image=None,
    )
    prepared.inference = Image.new("RGB", (8, 8), color=(10, 20, 30))
    prepared.original = prepared.inference

    mir = adapter.predict(prepared, InferenceOptions(prompt="box"))
    assert called["direct"] == 1
    assert called["workflow"] == 0
    assert mir.success
    assert mir.final_count == 2
    assert not mir.detections[0].count_only
    assert mir.technical_details.get("parser_stage") == "parsed_ok"
