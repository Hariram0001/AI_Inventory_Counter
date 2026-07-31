"""Offline tests for safe dynamic YOLO-World workflow prompt injection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from detector import (
    DetectorError,
    RoboflowDetector,
    classify_dynamic_prompt_status,
    inject_class_names_into_workflow_spec,
    prompt_to_class_names,
    sanitize_workflow_specification_for_debug,
    verify_dynamic_prompt_propagation,
)
from schemas import ModelConfig


YOLO_SPEC = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        }
    ],
    "steps": [
        {
            "type": "roboflow_core/yolo_world_model@v1",
            "name": "model",
            "class_names": ["wood fence"],
            "confidence": 0.4,
            "images": "$inputs.image",
        }
    ],
}


def _workflow_model() -> ModelConfig:
    return ModelConfig(
        name="YOLO-World",
        kind="workflow",
        workspace_name="hariram-s-mzhvc",
        workflow_id="custom-workflow",
        image_input_name="image",
        supports_prompt=True,
        dynamic_classes=True,
        enabled=True,
    )


def test_compatible_yolo_world_step_found_and_ids_recorded():
    result = inject_class_names_into_workflow_spec(
        YOLO_SPEC, ["traffic cone", "road cone", "safety cone"]
    )
    assert result.injected is True
    assert result.matched_step_count == 1
    assert result.matched_step_ids == ["model"]
    assert "yolo_world" in result.matched_step_types[0]
    assert result.field_used == "class_names"
    assert result.class_names == ["traffic cone", "road cone", "safety cone"]
    assert result.specification["steps"][0]["class_names"] == [
        "traffic cone",
        "road cone",
        "safety cone",
    ]
    # Default fence terms must not remain after injection
    assert result.specification["steps"][0]["class_names"] != ["wood fence"]


def test_prompt_injection_success_dict_shape():
    result = inject_class_names_into_workflow_spec(YOLO_SPEC, ["chair"])
    blob = result.to_dict()
    assert blob == {
        "injected": True,
        "matched_step_count": 1,
        "matched_step_ids": ["model"],
        "matched_step_types": ["roboflow_core/yolo_world_model@v1"],
        "field_used": "class_names",
        "class_names": ["chair"],
    }


def test_no_compatible_step_found():
    spec = {
        "steps": [{"type": "roboflow_core/some_other@v1", "name": "other"}],
        "outputs": [{"name": "predictions"}],
    }
    result = inject_class_names_into_workflow_spec(spec, ["traffic cone"])
    assert result.injected is False
    assert result.matched_step_count == 0
    assert result.field_used is None


def test_traffic_cones_prompts_not_replaced_by_fence():
    names = prompt_to_class_names("traffic cone, road cone, safety cone")
    assert names == ["traffic cone", "road cone", "safety cone"]
    result = inject_class_names_into_workflow_spec(YOLO_SPEC, names)
    assert "wood fence" not in result.class_names
    assert "fence" not in " ".join(result.class_names).lower()
    assert result.specification["steps"][0]["class_names"] == names


def test_dynamic_run_cannot_use_unmodified_fallback(monkeypatch):
    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    client = MagicMock()
    det._client = client

    monkeypatch.setattr(
        det,
        "_fetch_published_workflow_specification",
        lambda *_a, **_k: json.loads(json.dumps(YOLO_SPEC)),
    )

    def _fail_injected(**kwargs):
        assert "specification" in kwargs
        assert kwargs["specification"]["steps"][0]["class_names"] == ["traffic cone"]
        raise RuntimeError("injected spec boom")

    client.run_workflow.side_effect = _fail_injected

    with pytest.raises(DetectorError, match="Injected workflow specification"):
        det.run_workflow(model, "img.jpg", prompt="traffic cone")

    assert det.last_empty_draft_fallback is False
    assert det.last_dynamic_prompt_status == "EXECUTION_FAILED"
    # Must not retry workflow_id / unmodified published defaults
    assert client.run_workflow.call_count == 1


def test_dynamic_run_fails_when_injection_impossible(monkeypatch):
    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    det._client = MagicMock()
    bad_spec = {
        "steps": [{"type": "other", "name": "x"}],
        "outputs": [{"name": "predictions"}],
    }
    monkeypatch.setattr(
        det,
        "_fetch_published_workflow_specification",
        lambda *_a, **_k: bad_spec,
    )
    with pytest.raises(DetectorError, match="Dynamic prompts could not be applied"):
        det.run_workflow(model, "img.jpg", prompt="traffic cone, road cone")
    assert det.last_injection_result is not None
    assert det.last_injection_result["injected"] is False
    assert det.last_dynamic_prompt_status in {
        "WORKFLOW_NOT_DYNAMIC",
        "INJECTION_FAILED",
    }
    det._client.run_workflow.assert_not_called()


def test_fixed_prompt_legacy_may_use_explicit_fallback(monkeypatch):
    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    client = MagicMock()
    det._client = client

    empty = [{}]
    published_ok = [
        {
            "predictions": {
                "predictions": [
                    {
                        "class": "wood fence",
                        "confidence": 0.9,
                        "x": 50,
                        "y": 50,
                        "width": 40,
                        "height": 40,
                    }
                ],
                "image": {"width": 100, "height": 100},
            }
        }
    ]

    def _side_effect(**kwargs):
        if kwargs.get("specification") is not None:
            return published_ok
        return empty

    client.run_workflow.side_effect = _side_effect
    monkeypatch.setattr(
        det,
        "_fetch_published_workflow_specification",
        lambda *_a, **_k: json.loads(json.dumps(YOLO_SPEC)),
    )

    # No prompt → empty-draft fallback permitted
    payload = det.run_workflow(model, "img.jpg", prompt=None, allow_unmodified_fallback=True)
    assert payload == published_ok
    assert det.last_empty_draft_fallback is True
    assert det.last_invocation_mode == "published_specification"


def test_fixed_prompt_legacy_can_disable_fallback(monkeypatch):
    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    client = MagicMock()
    det._client = client
    client.run_workflow.return_value = [{}]
    with pytest.raises(DetectorError, match="empty draft"):
        det.run_workflow(model, "img.jpg", prompt="", allow_unmodified_fallback=False)


def test_injected_specification_execution_success(monkeypatch):
    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    client = MagicMock()
    det._client = client
    monkeypatch.setattr(
        det,
        "_fetch_published_workflow_specification",
        lambda *_a, **_k: json.loads(json.dumps(YOLO_SPEC)),
    )
    client.run_workflow.return_value = [
        {
            "predictions": {
                "predictions": [
                    {
                        "class": "traffic cone",
                        "confidence": 0.8,
                        "x": 10,
                        "y": 10,
                        "width": 5,
                        "height": 5,
                    }
                ],
                "image": {"width": 100, "height": 100},
            }
        }
    ]
    payload = det.run_workflow(
        model, "img.jpg", prompt="traffic cone, road cone, safety cone"
    )
    assert det.last_invocation_mode == "published_specification_with_prompt"
    assert det.last_empty_draft_fallback is False
    assert det.last_injection_result["injected"] is True
    assert det.last_dynamic_prompt_status == "VERIFIED_DYNAMIC"
    kwargs = client.run_workflow.call_args.kwargs
    assert kwargs["specification"]["steps"][0]["class_names"] == [
        "traffic cone",
        "road cone",
        "safety cone",
    ]
    assert kwargs.get("parameters") is None
    assert payload[0]["predictions"]["predictions"][0]["class"] == "traffic cone"


def test_diagnostic_statuses():
    assert (
        classify_dynamic_prompt_status(
            injected=False,
            invocation_mode=None,
            fallback_used=False,
            request_completed=False,
            parse_ok=False,
            raw_count=0,
            error_type="workflow_not_dynamic",
        )
        == "WORKFLOW_NOT_DYNAMIC"
    )
    assert (
        classify_dynamic_prompt_status(
            injected=False,
            invocation_mode=None,
            fallback_used=False,
            request_completed=False,
            parse_ok=False,
            raw_count=0,
        )
        == "INJECTION_FAILED"
    )
    assert (
        classify_dynamic_prompt_status(
            injected=True,
            invocation_mode="published_specification_with_prompt",
            fallback_used=True,
            request_completed=True,
            parse_ok=True,
            raw_count=1,
        )
        == "EXECUTION_FAILED"
    )
    assert (
        classify_dynamic_prompt_status(
            injected=True,
            invocation_mode="published_specification_with_prompt",
            fallback_used=False,
            request_completed=True,
            parse_ok=True,
            raw_count=0,
        )
        == "SUCCESSFUL_ZERO_DETECTIONS"
    )
    assert (
        classify_dynamic_prompt_status(
            injected=True,
            invocation_mode="published_specification_with_prompt",
            fallback_used=False,
            request_completed=True,
            parse_ok=True,
            raw_count=2,
        )
        == "VERIFIED_DYNAMIC"
    )


def test_successful_zero_detections_status(monkeypatch, tmp_path: Path):
    from PIL import Image

    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(img_path)

    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    client = MagicMock()
    det._client = client
    monkeypatch.setattr(
        det,
        "_fetch_published_workflow_specification",
        lambda *_a, **_k: json.loads(json.dumps(YOLO_SPEC)),
    )
    client.run_workflow.return_value = [
        {
            "predictions": {
                "predictions": [],
                "image": {"width": 64, "height": 64},
            }
        }
    ]
    report = verify_dynamic_prompt_propagation(
        det,
        model,
        str(img_path),
        class_names=["traffic cone", "road cone", "safety cone"],
        inventory_key="Traffic Cones",
    )
    assert report["status"] == "SUCCESSFUL_ZERO_DETECTIONS"
    assert report["fallback_used"] is False
    assert report["invocation_mode"] == "published_specification_with_prompt"
    assert report["injected_class_names"] == [
        "traffic cone",
        "road cone",
        "safety cone",
    ]
    assert report["raw_count"] == 0


def test_returned_classes_preserved(monkeypatch, tmp_path: Path):
    from PIL import Image

    img_path = tmp_path / "x.jpg"
    Image.new("RGB", (80, 80), color=(200, 180, 160)).save(img_path)
    det = RoboflowDetector(demo_mode=False, api_key="test-key")
    model = _workflow_model()
    client = MagicMock()
    det._client = client
    monkeypatch.setattr(
        det,
        "_fetch_published_workflow_specification",
        lambda *_a, **_k: json.loads(json.dumps(YOLO_SPEC)),
    )
    client.run_workflow.return_value = [
        {
            "predictions": {
                "predictions": [
                    {
                        "class": "wooden gate",
                        "confidence": 0.77,
                        "x": 40,
                        "y": 40,
                        "width": 20,
                        "height": 30,
                    }
                ],
                "image": {"width": 80, "height": 80},
            }
        }
    ]
    report = verify_dynamic_prompt_propagation(
        det,
        model,
        str(img_path),
        class_names=["wooden gate", "driveway gate"],
        inventory_key="Custom Item",
    )
    assert report["status"] == "VERIFIED_DYNAMIC"
    assert report["raw_returned_classes"] == ["wooden gate"]
    assert report["normalized_count"] == 1


def test_no_secrets_in_debug_snapshot():
    dirty = {
        "steps": [
            {
                "type": "roboflow_core/yolo_world_model@v1",
                "name": "model",
                "class_names": ["wood fence"],
                "api_key": "SECRET_KEY_VALUE",
                "authorization": "Bearer SECRET",
            }
        ],
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "outputs": [{"name": "predictions", "type": "JsonField"}],
    }
    snap = sanitize_workflow_specification_for_debug(
        dirty,
        workspace_name="hariram-s-mzhvc",
        workflow_id="custom-workflow",
    )
    blob = json.dumps(snap)
    assert "SECRET_KEY_VALUE" not in blob
    assert "Bearer SECRET" not in blob
    assert "api_key" not in blob.lower() or "***" in blob.lower()
    assert snap["has_yolo_world"] is True
    assert snap["steps"][0]["compatible_for_injection"] is True
    assert snap["class_names_fields"][0]["class_names"] == ["wood fence"]
