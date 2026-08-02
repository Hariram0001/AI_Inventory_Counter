"""Internal detection schemas and result containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Detection:
    detection_id: str
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    source_model: str
    source_image: str
    tile_id: str | None = None
    scale_id: str | None = None
    is_edge_detection: bool = False
    suspected_overlap: bool = False
    suspected_occlusion: bool = False
    included_in_count: bool = True
    contributing_models: list[str] = field(default_factory=list)
    agreement_count: int = 1
    merged_from: list[str] = field(default_factory=list)
    is_manual: bool = False
    marker_number: int | None = None
    # Photo-preparation region filtering (optional; older records omit these)
    inside_include_area: bool | None = None
    inside_exclude_area: bool | None = None
    region_overlap_ratio: float | None = None
    excluded_by_region: bool = False
    region_exclusion_reason: str | None = None
    # OpenRouter / VLM count-only rows (no bounding boxes invented).
    count_only: bool = False
    item_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def counted_items(self) -> int:
        if not self.included_in_count:
            return 0
        try:
            n = int(self.item_count)
        except (TypeError, ValueError):
            n = 1
        return max(0, n)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 0.0
        return self.width / self.height


@dataclass
class InferenceResult:
    """Result of a single model/strategy run on one image."""

    image_name: str
    model_name: str
    prompt: str
    inference_mode: str
    deduplication_strategy: str
    detections: list[Detection]
    raw_count: int
    final_count: int
    duplicates_removed: int
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    suspected_overlap_count: int
    suspected_occlusion_count: int
    processing_time_seconds: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    used_resized_copy: bool = False
    annotated_image_bytes: bytes | None = None
    strategy_counts: dict[str, int] = field(default_factory=dict)
    api_calls_used: int = 0
    tile_failures: int = 0
    scale_id: str | None = None
    # Pipeline provenance / diagnostics (never secrets)
    success: bool = True
    source: str = ""
    request_completed: bool = True
    predictions_found: bool = False
    error_type: str | None = None
    error_message: str | None = None
    raw_prediction_count: int = 0
    normalized_prediction_count: int = 0
    invocation_mode: str | None = None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "inference_mode": self.inference_mode,
            "deduplication_strategy": self.deduplication_strategy,
            "raw_count": self.raw_count,
            "final_count": self.final_count,
            "duplicates_removed": self.duplicates_removed,
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "suspected_overlap_count": self.suspected_overlap_count,
            "suspected_occlusion_count": self.suspected_occlusion_count,
            "processing_time_seconds": self.processing_time_seconds,
            "warnings": self.warnings,
            "errors": self.errors,
            "used_resized_copy": self.used_resized_copy,
            "strategy_counts": self.strategy_counts,
            "api_calls_used": self.api_calls_used,
            "tile_failures": self.tile_failures,
            "success": self.success,
            "source": self.source,
            "request_completed": self.request_completed,
            "predictions_found": self.predictions_found,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "raw_prediction_count": self.raw_prediction_count,
            "normalized_prediction_count": self.normalized_prediction_count,
            "invocation_mode": self.invocation_mode,
        }


@dataclass
class ConsensusResult:
    consensus_detections: list[Detection]
    consensus_count: int
    multi_model_supported: int
    single_model_only: int
    min_agreement: int
    model_results: list[InferenceResult]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    name: str
    kind: str  # "model", "workflow", or "local"
    enabled: bool = False
    model_id: str | None = None
    workspace_name: str | None = None
    workflow_id: str | None = None
    image_input_name: str = "image"
    prompt_parameter_name: str = "classes"
    supported_inventory_types: list[str] = field(default_factory=list)
    allowed_classes: list[str] = field(default_factory=list)
    supports_prompt: bool = False
    session_only: bool = False
    # Optional registry metadata (backward compatible)
    key: str | None = None
    provider: str | None = None
    is_default: bool = False
    default_confidence: float | None = None
    default_iou: float | None = None
    counting_strategy: str | None = None
    annotation_support: bool = True
    segmentation_support: bool = False
    demo_only: bool = False
    dynamic_classes: bool = False
    # Bring-your-own-key models receive the caller's key as a workflow parameter.
    requires_user_api_key: bool = False
    api_key_parameter_name: str = "model_api_key"

    def is_demo_model_id(self) -> bool:
        if self.demo_only:
            return True
        mid = (self.model_id or "").strip().lower()
        return mid.startswith("demo-") or mid in {"demo-fence-panels/1"}

    def validation_errors(self, *, allow_demo_ids: bool = True) -> list[str]:
        errors: list[str] = []
        if not self.name.strip():
            errors.append("Model display name is required.")
        kind = (self.kind or "").lower().strip()
        if kind not in {"model", "workflow", "local"}:
            errors.append("Integration type must be 'model', 'workflow', or 'local'.")
        if kind == "model":
            if not self.model_id or "replace-with" in (self.model_id or ""):
                errors.append(f"{self.name}: model_id is missing or still a placeholder.")
            elif "/" not in (self.model_id or ""):
                errors.append(
                    f"{self.name}: model_id should look like 'project-id/version'."
                )
            elif not allow_demo_ids and self.is_demo_model_id():
                errors.append(
                    f"{self.name}: demo model_id cannot be used when DEMO_MODE is false."
                )
        if kind == "workflow":
            if not self.workspace_name or "replace-with" in (self.workspace_name or ""):
                errors.append(
                    f"{self.name}: workspace_name is missing or still a placeholder."
                )
            if not self.workflow_id or "replace-with" in (self.workflow_id or ""):
                errors.append(
                    f"{self.name}: workflow_id is missing or still a placeholder."
                )
            if not self.image_input_name:
                errors.append(f"{self.name}: image_input_name is required for workflows.")
        if kind == "local":
            mid = (self.model_id or "").strip().lower()
            if mid not in {"local-picket-counter", "picket-counter", "local_picket"}:
                errors.append(
                    f"{self.name}: local model_id must be 'local-picket-counter'."
                )
        return errors

    def is_valid(self, *, allow_demo_ids: bool = True) -> bool:
        return not self.validation_errors(allow_demo_ids=allow_demo_ids)

    def supports_inventory(self, inventory_type: str) -> bool:
        if not self.supported_inventory_types:
            return True
        return inventory_type in self.supported_inventory_types

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], session_only: bool = False) -> "ModelConfig":
        return cls(
            name=str(data.get("name", "Unnamed Model")),
            kind=str(data.get("kind", "model")).lower(),
            enabled=bool(data.get("enabled", False)),
            model_id=data.get("model_id"),
            workspace_name=data.get("workspace_name"),
            workflow_id=data.get("workflow_id"),
            image_input_name=str(data.get("image_input_name", "image")),
            prompt_parameter_name=str(data.get("prompt_parameter_name", "classes")),
            supported_inventory_types=list(data.get("supported_inventory_types") or []),
            allowed_classes=list(data.get("allowed_classes") or []),
            supports_prompt=bool(data.get("supports_prompt", False)),
            session_only=session_only or bool(data.get("session_only", False)),
            key=data.get("key"),
            provider=data.get("provider"),
            is_default=bool(data.get("is_default", False)),
            default_confidence=(
                float(data["default_confidence"])
                if data.get("default_confidence") is not None
                else None
            ),
            default_iou=(
                float(data["default_iou"]) if data.get("default_iou") is not None else None
            ),
            counting_strategy=data.get("counting_strategy"),
            annotation_support=bool(data.get("annotation_support", True)),
            segmentation_support=bool(data.get("segmentation_support", False)),
            demo_only=bool(data.get("demo_only", False)),
            dynamic_classes=bool(
                data.get("dynamic_classes", data.get("supports_prompt", False))
            ),
            requires_user_api_key=bool(data.get("requires_user_api_key", False)),
            api_key_parameter_name=str(
                data.get("api_key_parameter_name") or "model_api_key"
            ),
        )
