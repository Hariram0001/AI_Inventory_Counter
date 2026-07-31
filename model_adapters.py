"""Common detection-model adapter interface wrapping existing inference code."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from schemas import Detection, InferenceResult, ModelConfig


@dataclass
class ModelValidationResult:
    ok: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceOptions:
    prompt: str = ""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    inference_mode: str = "Whole-image inference"
    tile_size: int = 800
    tile_overlap: float = 0.25
    deduplication_strategy: str = "Conservative"


@dataclass
class ModelInferenceResult:
    """Canonical per-model result for UI comparison and review."""

    model_key: str
    model_display_name: str
    provider: str
    success: bool
    response_source: str
    processing_time_seconds: float
    raw_count: int
    final_count: int
    avg_confidence: float
    max_confidence: float
    classes: list[str]
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    detections: list[Detection] = field(default_factory=list)
    annotated_image_bytes: bytes | None = None
    inference_result: InferenceResult | None = None
    technical_details: dict[str, Any] = field(default_factory=dict)
    model_source: str = ""
    task_type: str = "object_detection"
    effective_prompt: list[str] = field(default_factory=list)
    effective_threshold: float | None = None

    def to_canonical_dict(self) -> dict[str, Any]:
        """Common result schema for UI — provider parsing stays in adapters."""
        dets = [d.to_dict() for d in self.detections]
        counted = [d for d in self.detections if getattr(d, "included_in_count", True)]
        return {
            "model_key": self.model_key,
            "model_display_name": self.model_display_name,
            "model_source": self.model_source or self.provider,
            "task_type": self.task_type,
            "success": self.success,
            "response_source": self.response_source,
            "processing_time_seconds": self.processing_time_seconds,
            "raw_predictions": self.technical_details.get("raw_predictions") or [],
            "normalized_detections": dets,
            "final_detections": [d.to_dict() for d in counted],
            "raw_count": self.raw_count,
            "final_count": self.final_count,
            "average_confidence": self.avg_confidence if self.detections else None,
            "maximum_confidence": self.max_confidence if self.detections else None,
            "classes": list(self.classes),
            "warnings": list(self.warnings),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "effective_prompt": list(self.effective_prompt),
            "effective_threshold": self.effective_threshold,
        }

    @classmethod
    def from_inference_result(
        cls,
        result: InferenceResult,
        *,
        model: ModelConfig,
        provider: str,
        effective_prompt: str | list[str] | None = None,
        effective_threshold: float | None = None,
        model_source: str = "",
        task_type: str = "object_detection",
    ) -> "ModelInferenceResult":
        failed = bool(result.errors) or result.error_type in {
            "api_error",
            "empty_workflow_output",
        }
        if isinstance(effective_prompt, str):
            prompt_list = [p.strip() for p in effective_prompt.split(",") if p.strip()]
        else:
            prompt_list = list(effective_prompt or [])
        return cls(
            model_key=model_key(model),
            model_display_name=model.name,
            provider=provider,
            success=not failed and result.request_completed,
            response_source=result.source or "unknown",
            processing_time_seconds=result.processing_time_seconds,
            raw_count=result.raw_prediction_count or result.raw_count,
            final_count=result.final_count,
            avg_confidence=result.avg_confidence,
            max_confidence=result.max_confidence,
            classes=sorted({d.class_name for d in result.detections}),
            warnings=list(result.warnings),
            error_type=result.error_type,
            error_message=result.error_message or ("; ".join(result.errors) if result.errors else None),
            detections=list(result.detections),
            annotated_image_bytes=result.annotated_image_bytes,
            inference_result=result,
            technical_details={
                "invocation_mode": result.invocation_mode,
                "normalized_prediction_count": result.normalized_prediction_count,
                "duplicates_removed": result.duplicates_removed,
                "api_calls_used": result.api_calls_used,
            },
            model_source=model_source or provider,
            task_type=task_type,
            effective_prompt=prompt_list,
            effective_threshold=effective_threshold,
        )

    @classmethod
    def failed(
        cls,
        model: ModelConfig,
        *,
        provider: str,
        error_type: str,
        error_message: str,
        processing_time_seconds: float = 0.0,
    ) -> "ModelInferenceResult":
        return cls(
            model_key=model_key(model),
            model_display_name=model.name,
            provider=provider,
            success=False,
            response_source="error",
            processing_time_seconds=processing_time_seconds,
            raw_count=0,
            final_count=0,
            avg_confidence=0.0,
            max_confidence=0.0,
            classes=[],
            error_type=error_type,
            error_message=error_message,
        )


def model_key(model: ModelConfig) -> str:
    """Stable key for caching and registry — never invents remote IDs."""
    kind = (model.kind or "model").lower()
    if kind == "workflow":
        return f"workflow:{(model.workspace_name or '').strip()}/{(model.workflow_id or '').strip()}"
    if kind == "local":
        return f"local:{(model.model_id or model.name).strip()}"
    return f"model:{(model.model_id or model.name).strip()}"


def provider_for(model: ModelConfig) -> str:
    kind = (model.kind or "").lower()
    if kind == "workflow" or kind == "model":
        if model.is_demo_model_id():
            return "Demo"
        return "Roboflow"
    if kind == "local":
        return "Local"
    return kind or "Unknown"


class DetectionModelAdapter(Protocol):
    model: ModelConfig

    def validate_configuration(self) -> ModelValidationResult: ...

    def predict(self, prepared_image: Any, options: InferenceOptions) -> ModelInferenceResult: ...


class UnsupportedModelAdapter:
    """Placeholder when no verified execution route exists."""

    def __init__(self, model: ModelConfig, reason: str = "No verified adapter") -> None:
        self.model = model
        self.reason = reason

    def validate_configuration(self) -> ModelValidationResult:
        return ModelValidationResult(
            ok=False,
            message=self.reason,
            details={"adapter_type": "none", "provider": provider_for(self.model)},
        )

    def predict(self, prepared_image: Any, options: InferenceOptions) -> ModelInferenceResult:
        return ModelInferenceResult.failed(
            self.model,
            provider=provider_for(self.model),
            error_type="unavailable",
            error_message=self.reason,
        )


class RoboflowWorkflowAdapter:
    """Adapter for YOLO-World Workflow, hosted OD models, and local classical."""

    def __init__(
        self,
        model: ModelConfig,
        detector: Any | None = None,
        *,
        adapter_type: str = "yolo_world_workflow",
    ) -> None:
        from detector import RoboflowDetector

        self.model = model
        self.detector = detector or RoboflowDetector()
        self.adapter_type = adapter_type

    def validate_configuration(self) -> ModelValidationResult:
        errors = self.model.validation_errors(allow_demo_ids=False)
        if errors:
            return ModelValidationResult(ok=False, message="; ".join(errors))
        kind = (self.model.kind or "").lower()
        if kind == "local":
            return ModelValidationResult(
                ok=True,
                message="Local classical detector is configured (no API key required).",
                details={"provider": "Local", "model_id": self.model.model_id},
            )
        if kind == "model" and not (self.model.model_id or "").strip():
            return ModelValidationResult(
                ok=False,
                message="Hosted object-detection model is missing model_id.",
            )
        ok, msg = self.detector.test_connectivity()
        return ModelValidationResult(
            ok=ok,
            message=msg,
            details={
                "provider": provider_for(self.model),
                "workspace": self.model.workspace_name,
                "workflow_id": self.model.workflow_id,
                "model_id": self.model.model_id,
                "adapter_type": self.adapter_type,
            },
        )

    def predict(self, prepared_image: Any, options: InferenceOptions) -> ModelInferenceResult:
        from detector import DetectorError, run_inference_on_prepared_image
        from detection_ids import assign_stable_detection_ids

        started = time.perf_counter()
        # Fixed-class models must not receive arbitrary inventory prompts.
        prompt = options.prompt
        if not (self.model.supports_prompt or self.model.dynamic_classes):
            prompt = ""
        conf = float(options.confidence_threshold)
        iou = float(options.iou_threshold)
        try:
            result = run_inference_on_prepared_image(
                self.detector,
                prepared_image,
                self.model,
                prompt=prompt,
                confidence_threshold=conf,
                iou_threshold=iou,
                inference_mode=options.inference_mode,
                tile_size=options.tile_size,
                tile_overlap=options.tile_overlap,
                deduplication_strategy=options.deduplication_strategy,
            )
            image_hash = getattr(prepared_image, "content_hash", "") or ""
            result.detections = assign_stable_detection_ids(
                result.detections,
                image_hash=image_hash,
                model_key=model_key(self.model),
            )
            source = "foundation" if (self.model.kind or "").lower() == "workflow" else "workspace"
            if self.model.demo_only or self.model.is_demo_model_id():
                source = "demo"
            elif (self.model.kind or "").lower() == "local":
                source = "local"
            effective = (
                prompt
                if (self.model.supports_prompt or self.model.dynamic_classes)
                else list(self.model.allowed_classes or [])
            )
            return ModelInferenceResult.from_inference_result(
                result,
                model=self.model,
                provider=provider_for(self.model),
                effective_prompt=effective,
                effective_threshold=conf,
                model_source=source,
                task_type="object_detection",
            )
        except DetectorError as exc:
            import traceback

            traceback.print_exc()
            return ModelInferenceResult.failed(
                self.model,
                provider=provider_for(self.model),
                error_type="api_error",
                error_message=f"{type(exc).__name__}: {exc}",
                processing_time_seconds=time.perf_counter() - started,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            return ModelInferenceResult.failed(
                self.model,
                provider=provider_for(self.model),
                error_type="unexpected_error",
                error_message=f"{type(exc).__name__}: {str(exc)[:800]}",
                processing_time_seconds=time.perf_counter() - started,
            )


class OpenRouterVLMAdapter(RoboflowWorkflowAdapter):
    """Adapter for the OpenRouter VLM detector workflow.

    Inference is billed to the user's own OpenRouter account. The key is held
    only for the duration of the call chain and is never persisted or logged.
    """

    def __init__(
        self,
        model: ModelConfig,
        detector: Any | None = None,
        *,
        model_api_key: str = "",
    ) -> None:
        from detector import RoboflowDetector

        if detector is None:
            detector = RoboflowDetector(model_api_key=model_api_key)
        elif model_api_key:
            detector.model_api_key = model_api_key
        super().__init__(model, detector=detector, adapter_type="openrouter_vlm_detector")
        self.model_api_key = str(model_api_key or "")

    def validate_configuration(self) -> ModelValidationResult:
        if not self.model_api_key and not getattr(self.detector, "model_api_key", ""):
            return ModelValidationResult(
                ok=False,
                message=(
                    "Add and verify your OpenRouter API key on the API "
                    "Connections page to use this model."
                ),
                details={"adapter_type": "openrouter_vlm_detector", "provider": "OpenRouter"},
            )
        errors = self.model.validation_errors(allow_demo_ids=False)
        if errors:
            return ModelValidationResult(ok=False, message="; ".join(errors))
        return ModelValidationResult(
            ok=True,
            message="OpenRouter model is configured and a verified key is present.",
            details={
                "adapter_type": "openrouter_vlm_detector",
                "provider": "OpenRouter",
                "workspace": self.model.workspace_name,
                "workflow_id": self.model.workflow_id,
            },
        )

    def predict(self, prepared_image: Any, options: InferenceOptions) -> ModelInferenceResult:
        result = super().predict(prepared_image, options)
        result.provider = "OpenRouter"
        result.model_source = "openrouter"
        return result


def resolve_adapter_type(model: ModelConfig) -> str:
    """Route by catalog adapter_type when available; else infer from ModelConfig."""
    try:
        from model_catalog import (
            ADAPTER_LOCAL,
            ADAPTER_NONE,
            ADAPTER_ROBOFLOW_OD,
            ADAPTER_YOLO_WORLD,
            canonicalize_adapter_type,
            get_all_catalog_models,
        )

        key = model_key(model)
        for entry in get_all_catalog_models():
            if entry.key == key or entry.display_name == model.name:
                return canonicalize_adapter_type(
                    entry.adapter_type,
                    kind=entry.kind or model.kind,
                    workflow_id=entry.workflow_id or model.workflow_id,
                    dynamic=bool(
                        entry.dynamic_prompts
                        or entry.dynamic_classes
                        or model.dynamic_classes
                        or model.supports_prompt
                    ),
                )
        return canonicalize_adapter_type(
            None,
            kind=model.kind,
            workflow_id=model.workflow_id,
            dynamic=bool(model.dynamic_classes or model.supports_prompt),
        )
    except Exception:  # noqa: BLE001
        kind = (model.kind or "").lower()
        if kind == "local":
            return "local_classical"
        if kind == "workflow" and (model.dynamic_classes or model.supports_prompt):
            return "yolo_world_workflow"
        if kind == "model" and (model.model_id or "").strip():
            return "roboflow_object_detection"
        if kind == "workflow":
            return "yolo_world_workflow"
        return "none"


def get_adapter(
    model: ModelConfig,
    detector: Any | None = None,
    *,
    model_api_key: str = "",
) -> RoboflowWorkflowAdapter | UnsupportedModelAdapter:
    """Factory routed by adapter type.

    Implemented today:
    - yolo_world_workflow (dynamic prompts + injected Workflow spec)
    - roboflow_object_detection (hosted model_id via existing detector)
    - local_classical (Local Picket Counter)
    - openrouter_vlm_detector (bring-your-own-key OpenRouter workflow)
    """
    adapter_type = resolve_adapter_type(model)

    if adapter_type == "openrouter_vlm_detector" or getattr(
        model, "requires_user_api_key", False
    ):
        return OpenRouterVLMAdapter(
            model, detector=detector, model_api_key=model_api_key
        )

    if adapter_type in {"none", "unsupported"}:
        return UnsupportedModelAdapter(
            model,
            reason="No verified adapter for this model in the current POC.",
        )
    kind = (model.kind or "").lower()
    if adapter_type == "roboflow_object_detection" and kind == "model":
        if not (model.model_id or "").strip():
            return UnsupportedModelAdapter(
                model,
                reason="Hosted object-detection model is missing a model_id.",
            )
    return RoboflowWorkflowAdapter(model, detector=detector, adapter_type=adapter_type)
