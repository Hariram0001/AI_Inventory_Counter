"""Roboflow detection, response normalization, and inference orchestration."""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from schemas import Detection, InferenceResult, ModelConfig
from overlap import (
    deduplicate,
    mark_overlap_and_occlusion,
    strategy_comparison_counts,
)
from image_processing import (
    PreparedImage,
    annotate_image,
    create_tiles,
    image_to_png_bytes,
    map_box_to_original,
    save_temp_image,
    translate_tile_detections,
)
from config import (
    DEMO_MODE,
    INFERENCE_TIMEOUT_SECONDS,
    MAX_API_CALLS_PER_IMAGE,
    MAX_TILES_PER_IMAGE,
    MOCK_RESPONSE_PATH,
    ROBOFLOW_API_KEY,
    ROBOFLOW_API_URL,
)

logger = logging.getLogger(__name__)


class DetectorError(Exception):
    """User-facing detection error (no secrets)."""


_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:api_key|apikey|token|key)=)[^&\s\"']+"
)
_SECRET_HEADER_RE = re.compile(
    r"(?i)(authorization|api[-_]?key|bearer)\s*[:=]\s*\S+"
)


def sanitize_exception_text(text: str, *, max_len: int = 800) -> str:
    """Keep the real error text, but strip API keys / tokens from URLs and headers."""
    cleaned = _SECRET_QUERY_RE.sub(r"\1***REDACTED***", str(text or ""))
    cleaned = _SECRET_HEADER_RE.sub(r"\1=***REDACTED***", cleaned)
    cleaned = cleaned.replace("\x00", "").strip()
    if len(cleaned) > max_len:
        return cleaned[: max_len - 3] + "..."
    return cleaned


def format_exception_for_user(exc: BaseException) -> str:
    """Type + message for UI/logs — never invent 'SDK not installed'."""
    return f"{type(exc).__name__}: {sanitize_exception_text(str(exc))}"


def log_exception_details(exc: BaseException, *, context: str) -> None:
    """Print full traceback plus type/message (secrets redacted in the message line)."""
    traceback.print_exc()
    logger.error(
        "%s | %s | %s",
        context,
        type(exc).__name__,
        sanitize_exception_text(str(exc)),
    )

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _normalize_confidence(value: Any, default: float = 0.0) -> float:
    """Normalize confidence from 0–1, 0–100, or numeric strings (never double-scale 0–1)."""
    conf = _safe_float(value, default=default)
    if conf > 1.0 and conf <= 100.0:
        conf = conf / 100.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def sanitize_payload_for_debug(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """Strip secrets and bulky binary fields for debug JSON dumps."""
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            lk = str(key).lower()
            if any(s in lk for s in ("api_key", "apikey", "token", "secret", "authorization")):
                out[str(key)] = "***REDACTED***"
            elif lk in {"image_bytes", "bytes", "data_uri", "encoded_image"} and not isinstance(
                value, (dict, list)
            ):
                out[str(key)] = f"<{type(value).__name__} omitted>"
            else:
                out[str(key)] = sanitize_payload_for_debug(value, depth + 1, max_depth)
        return out
    if isinstance(obj, list):
        return [sanitize_payload_for_debug(item, depth + 1, max_depth) for item in obj[:500]]
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return type(obj).__name__


def save_last_live_response(payload: Any, path: Path | None = None) -> Path:
    """Persist a sanitized live response under data/debug/."""
    from config import DATA_DIR, ensure_data_dir

    ensure_data_dir()
    debug_dir = DATA_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out = path or (debug_dir / "last_live_response.json")
    out.write_text(
        json.dumps(sanitize_payload_for_debug(payload), indent=2),
        encoding="utf-8",
    )
    return out


def summarize_response_shape(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Sanitized structural summary for logging (no large payloads)."""
    if depth >= max_depth:
        return type(obj).__name__
    if isinstance(obj, dict):
        return {str(k): summarize_response_shape(v, depth + 1, max_depth) for k, v in list(obj.items())[:30]}
    if isinstance(obj, list):
        if not obj:
            return []
        return [summarize_response_shape(obj[0], depth + 1, max_depth), f"...({len(obj)} items)"]
    return type(obj).__name__


def _extract_predictions_recursive(payload: Any) -> list[dict[str, Any]]:
    """Defensively find prediction dicts in nested Roboflow / Workflow responses."""
    found: list[dict[str, Any]] = []

    def looks_like_prediction(d: dict[str, Any]) -> bool:
        keys = {str(k).lower() for k in d.keys()}
        has_class = bool(keys & {"class", "class_name", "label", "name"})
        has_box = bool(
            keys
            & {
                "x",
                "y",
                "width",
                "height",
                "x1",
                "y1",
                "x2",
                "y2",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "confidence",
                "points",
                "bbox",
            }
        )
        return has_class and has_box

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in (
                "predictions",
                "detections",
                "predictions_detections",
                "model_predictions",
                "object_detection_predictions",
            ):
                if key not in node:
                    continue
                value = node[key]
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            # Nested Roboflow image wrapper: {"predictions":[...], "image":{...}}
                            if "predictions" in item and isinstance(item["predictions"], list):
                                for inner in item["predictions"]:
                                    if isinstance(inner, dict):
                                        found.append(inner)
                            elif looks_like_prediction(item) or any(
                                k in item for k in ("x", "y", "width", "height", "confidence")
                            ):
                                found.append(item)
                elif isinstance(value, dict):
                    # {"predictions": {"predictions": [...]}} or class-keyed maps
                    if "predictions" in value and isinstance(value["predictions"], list):
                        for item in value["predictions"]:
                            if isinstance(item, dict):
                                found.append(item)
                    else:
                        walk(value)
            if looks_like_prediction(node):
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in found:
        oid = id(item)
        if oid not in seen:
            seen.add(oid)
            unique.append(item)
    return unique


def _maybe_denormalize_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    """If all coords look normalized [0,1], convert to pixel space."""
    vals = (x1, y1, x2, y2)
    if (
        image_width > 1.5
        and image_height > 1.5
        and all(0.0 <= v <= 1.0 for v in vals)
    ):
        return (
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height,
        )
    return x1, y1, x2, y2


def _box_from_prediction(
    pred: dict[str, Any],
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float] | None:
    """Parse center/wh, xyxy, nested bbox, or polygon into pixel-space xyxy."""
    # Explicit corners
    if all(k in pred for k in ("x1", "y1", "x2", "y2")):
        box = (
            _safe_float(pred["x1"]),
            _safe_float(pred["y1"]),
            _safe_float(pred["x2"]),
            _safe_float(pred["y2"]),
        )
        return _maybe_denormalize_box(*box, image_width, image_height)
    if all(k in pred for k in ("xmin", "ymin", "xmax", "ymax")):
        box = (
            _safe_float(pred["xmin"]),
            _safe_float(pred["ymin"]),
            _safe_float(pred["xmax"]),
            _safe_float(pred["ymax"]),
        )
        return _maybe_denormalize_box(*box, image_width, image_height)

    # Nested bbox / bounding_box
    bbox = pred.get("bbox") or pred.get("bounding_box") or pred.get("box")
    if isinstance(bbox, dict):
        return _box_from_prediction(bbox, image_width, image_height)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        box = (
            _safe_float(bbox[0]),
            _safe_float(bbox[1]),
            _safe_float(bbox[2]),
            _safe_float(bbox[3]),
        )
        return _maybe_denormalize_box(*box, image_width, image_height)

    # Center format (Roboflow default); support normalized center boxes too
    if all(k in pred for k in ("x", "y", "width", "height")):
        cx = _safe_float(pred["x"])
        cy = _safe_float(pred["y"])
        w = _safe_float(pred["width"])
        h = _safe_float(pred["height"])
        if (
            image_width > 1.5
            and image_height > 1.5
            and 0.0 <= cx <= 1.0
            and 0.0 <= cy <= 1.0
            and 0.0 < w <= 1.0
            and 0.0 < h <= 1.0
        ):
            cx *= image_width
            cy *= image_height
            w *= image_width
            h *= image_height
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    # Polygon / points
    points = pred.get("points") or pred.get("polygon")
    if isinstance(points, list) and points:
        xs: list[float] = []
        ys: list[float] = []
        for pt in points:
            if isinstance(pt, dict):
                xs.append(_safe_float(pt.get("x")))
                ys.append(_safe_float(pt.get("y")))
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                xs.append(_safe_float(pt[0]))
                ys.append(_safe_float(pt[1]))
        if xs and ys:
            box = (min(xs), min(ys), max(xs), max(ys))
            return _maybe_denormalize_box(*box, image_width, image_height)

    return None


def normalize_predictions(
    payload: Any,
    *,
    source_model: str,
    source_image: str,
    image_width: float,
    image_height: float,
    confidence_threshold: float = 0.0,
    allowed_classes: list[str] | None = None,
    tile_id: str | None = None,
    scale_id: str | None = None,
    map_scale_x: float = 1.0,
    map_scale_y: float = 1.0,
    original_width: float | None = None,
    original_height: float | None = None,
) -> list[Detection]:
    """Normalize Roboflow-style responses into Detection objects."""
    preds = _extract_predictions_recursive(payload)
    if not preds and payload is not None:
        logger.warning(
            "No predictions found. Response shape: %s",
            summarize_response_shape(payload),
        )

    allowed = {c.strip().lower() for c in (allowed_classes or []) if c}
    ow = original_width if original_width is not None else image_width
    oh = original_height if original_height is not None else image_height

    detections: list[Detection] = []
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        conf = _normalize_confidence(
            pred.get("confidence", pred.get("score", pred.get("conf"))),
            default=0.0,
        )
        if conf < confidence_threshold:
            continue

        class_name = (
            pred.get("class")
            or pred.get("class_name")
            or pred.get("label")
            or pred.get("name")
            or "object"
        )
        class_name = str(class_name)
        if allowed and class_name.strip().lower() not in allowed:
            continue

        box = _box_from_prediction(pred, image_width, image_height)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        # Map from inference/tile space to original if scales provided
        if map_scale_x != 1.0 or map_scale_y != 1.0 or original_width is not None:
            x1, y1, x2, y2 = map_box_to_original(
                x1, y1, x2, y2, map_scale_x, map_scale_y, int(ow), int(oh)
            )
        else:
            x1 = max(0.0, min(float(ow), x1))
            x2 = max(0.0, min(float(ow), x2))
            y1 = max(0.0, min(float(oh), y1))
            y2 = max(0.0, min(float(oh), y2))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue

        detections.append(
            Detection(
                detection_id=str(pred.get("detection_id") or uuid.uuid4()),
                class_name=class_name,
                confidence=conf,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=(x1 + x2) / 2.0,
                center_y=(y1 + y2) / 2.0,
                width=width,
                height=height,
                source_model=source_model,
                source_image=source_image,
                tile_id=tile_id,
                scale_id=scale_id,
            )
        )
    return detections


def build_workflow_parameters(
    model: ModelConfig,
    image_path: str,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Isolate Workflow argument construction."""
    params: dict[str, Any] = {
        model.image_input_name or "image": image_path,
    }
    if model.supports_prompt and prompt and model.prompt_parameter_name:
        params[model.prompt_parameter_name] = prompt
    return params


def prompt_to_class_names(prompt: str | None) -> list[str]:
    """Turn a user detection prompt into YOLO-World class_names."""
    import re

    text = (prompt or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[,;\n]+", text) if p.strip()]
    return parts or [text]


def inject_class_names_into_workflow_spec(
    spec: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    """Override YOLO-World class_names in a published workflow specification."""
    import copy

    if not class_names:
        return spec
    updated = copy.deepcopy(spec)
    injected = False
    for step in updated.get("steps") or []:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("type") or "").lower()
        if "yolo_world" in step_type or "class_names" in step:
            step["class_names"] = list(class_names)
            injected = True
    if not injected:
        logger.warning(
            "Could not find a YOLO-World step to inject class_names=%s",
            class_names,
        )
    else:
        logger.info("Injected class_names into workflow specification: %s", class_names)
    return updated


def load_mock_response() -> Any:
    path = MOCK_RESPONSE_PATH
    if not path.exists():
        raise DetectorError("Demo Mode is enabled but mock_detection.json is missing.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DetectorError("Failed to load demo mock detection response.") from exc


def sanitize_model_id(model_id: str | None) -> str:
    """Hide most of a model id for safe logging/reporting."""
    if not model_id:
        return "(missing)"
    parts = model_id.split("/")
    if len(parts) != 2:
        return "(invalid-format)"
    project, version = parts
    if len(project) <= 4:
        masked = project[0] + "***" if project else "***"
    else:
        masked = project[:3] + "***" + project[-2:]
    return f"{masked}/{version}"


def response_shape_summary(payload: Any) -> dict[str, Any]:
    """Sanitized structural summary — never includes image bytes or large arrays."""
    summary: dict[str, Any] = {
        "top_level_type": type(payload).__name__,
        "top_level_keys": [],
        "nested_output_keys": [],
        "prediction_count": 0,
        "prediction_fields": [],
        "class_names": [],
    }
    if isinstance(payload, dict):
        summary["top_level_keys"] = sorted(str(k) for k in payload.keys())
        nested: set[str] = set()
        for key in ("outputs", "output", "result", "results", "data"):
            node = payload.get(key)
            if isinstance(node, dict):
                nested.update(str(k) for k in node.keys())
            elif isinstance(node, list) and node and isinstance(node[0], dict):
                nested.update(str(k) for k in node[0].keys())
        summary["nested_output_keys"] = sorted(nested)
    elif isinstance(payload, list):
        summary["top_level_keys"] = [f"list[{len(payload)}]"]
        if payload and isinstance(payload[0], dict):
            summary["nested_output_keys"] = sorted(str(k) for k in payload[0].keys())

    preds = _extract_predictions_recursive(payload)
    summary["prediction_count"] = len(preds)
    field_names: set[str] = set()
    classes: set[str] = set()
    for pred in preds[:50]:
        if isinstance(pred, dict):
            field_names.update(str(k) for k in pred.keys())
            cls = (
                pred.get("class")
                or pred.get("class_name")
                or pred.get("label")
                or pred.get("name")
            )
            if cls:
                classes.add(str(cls))
    summary["prediction_fields"] = sorted(field_names)
    summary["class_names"] = sorted(classes)
    return summary


class RoboflowDetector:
    """Hosted Roboflow inference via inference-sdk (or Demo Mode mocks)."""

    def __init__(
        self,
        api_key: str | None = None,
        demo_mode: bool | None = None,
        api_url: str | None = None,
    ) -> None:
        # Always re-read settings so .env edits apply without restarting Python
        from config import reload_settings

        reload_settings()
        import config as cfg

        self.demo_mode = cfg.DEMO_MODE if demo_mode is None else demo_mode
        self.api_key = (api_key if api_key is not None else cfg.ROBOFLOW_API_KEY) or ""
        self.api_url = api_url or cfg.ROBOFLOW_API_URL
        self._client = None
        self.last_source: str = "unset"  # "demo_mock" | "live_roboflow" | "local_classical"
        self.last_invocation_mode: str | None = None
        self.last_empty_draft_fallback: bool = False
        self.last_raw_prediction_count: int = 0
        self.last_local_warnings: list[str] = []

    def _get_client(self):
        if self.demo_mode:
            return None
        if not self.api_key:
            raise DetectorError(
                "ROBOFLOW_API_KEY is missing. Configure it in the environment "
                "or enable DEMO_MODE=true."
            )
        if self._client is None:
            logger.info("Creating client...")
            # Do not catch Exception and replace with a generic install message.
            # Import / construct failures must surface the ORIGINAL exception.
            try:
                from inference_sdk import InferenceHTTPClient
            except Exception as exc:  # noqa: BLE001
                log_exception_details(exc, context="inference_sdk import failed")
                raise
            try:
                self._client = InferenceHTTPClient(
                    api_url=self.api_url,
                    api_key=self.api_key,
                )
            except Exception as exc:  # noqa: BLE001
                log_exception_details(
                    exc, context="InferenceHTTPClient construction failed"
                )
                raise
            logger.info(
                "Initialized InferenceHTTPClient api_url=%s key_configured=yes",
                self.api_url,
            )
        return self._client

    def test_connectivity(self) -> tuple[bool, str]:
        if self.demo_mode:
            return True, "Demo Mode active — live API not required."
        if not self.api_key:
            return False, "API key missing."
        try:
            client = self._get_client()
            # Prefer a lightweight server probe when available
            if hasattr(client, "get_server_info"):
                try:
                    info = client.get_server_info()
                    info_type = type(info).__name__
                    keys = (
                        sorted(str(k) for k in info.keys())
                        if isinstance(info, dict)
                        else []
                    )
                    logger.info(
                        "Connectivity probe ok type=%s keys=%s",
                        info_type,
                        keys[:12],
                    )
                    return True, "Authenticated to Roboflow Serverless Hosted API."
                except Exception as probe_exc:  # noqa: BLE001
                    log_exception_details(
                        probe_exc, context="get_server_info probe failed"
                    )
                    msg = format_exception_for_user(probe_exc)
                    if "401" in msg or "unauthorized" in msg.lower():
                        return False, msg
                    return True, (
                        "Client initialized. Server info probe skipped; "
                        f"probe error was: {msg}"
                    )
            return True, "Client initialized successfully. Run Analyze to validate a model."
        except Exception as exc:  # noqa: BLE001
            log_exception_details(exc, context="Connectivity test failed")
            return False, format_exception_for_user(exc)

    def _classify_api_error(self, exc: Exception) -> str:
        """Legacy helper — prefer format_exception_for_user for UI text.

        Kept for tests that may still call it; returns a short category hint
        PLUS the original exception (secrets redacted), never 'SDK not installed'.
        """
        original = format_exception_for_user(exc)
        text = str(exc).lower()
        if "401" in text or "unauthorized" in text or "invalid api" in text:
            return f"Invalid or unauthorized Roboflow API key. ({original})"
        if "403" in text or "forbidden" in text:
            return f"Roboflow request forbidden. Check workspace permissions. ({original})"
        if "404" in text or "not found" in text:
            return f"Model or Workflow not found. Check model_id / workflow_id. ({original})"
        if "429" in text or "rate" in text:
            return f"Roboflow rate limit exceeded. Wait and retry. ({original})"
        if "credit" in text or "quota" in text or "payment" in text:
            return f"Insufficient Roboflow credits or quota. ({original})"
        if "timeout" in text or "timed out" in text:
            return f"Roboflow request timed out. ({original})"
        if "workflow" in text and ("input" in text or "parameter" in text):
            return (
                "Workflow input parameters do not match this configuration. "
                f"Check image_input_name and prompt_parameter_name. ({original})"
            )
        logger.error("API failure class=%s", type(exc).__name__)
        return original

    def run_direct_model(
        self,
        model: ModelConfig,
        image_path: str,
        confidence: float = 0.4,
    ) -> Any:
        if self.demo_mode:
            self.last_source = "demo_mock"
            return load_mock_response()
        if model.is_demo_model_id():
            raise DetectorError(
                "Cannot call a demo model_id while DEMO_MODE is false. "
                "Enable a real Roboflow model in models.json."
            )
        client = self._get_client()
        try:
            if not hasattr(client, "infer"):
                raise DetectorError(
                    "Installed inference-sdk does not expose a known infer method."
                )
            logger.info("Sending request... (direct model)")
            logger.info("Waiting...")
            payload = client.infer(image_path, model_id=model.model_id)
            logger.info("Response received...")
            self.last_source = "live_roboflow"
            self.last_invocation_mode = "direct_model"
            self.last_empty_draft_fallback = False
            self.last_raw_prediction_count = len(_extract_predictions_recursive(payload))
            shape = response_shape_summary(payload)
            logger.info(
                "Live inference ok model=%s shape=%s",
                sanitize_model_id(model.model_id),
                shape,
            )
            try:
                save_last_live_response(payload)
            except Exception:  # noqa: BLE001
                logger.debug("Could not persist last_live_response.json", exc_info=True)
            return payload
        except DetectorError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_exception_details(
                exc,
                context=f"Direct model inference failed model={sanitize_model_id(model.model_id)}",
            )
            raise DetectorError(format_exception_for_user(exc)) from exc

    def _fetch_published_workflow_specification(
        self,
        workspace_name: str,
        workflow_id: str,
    ) -> dict[str, Any] | None:
        """Load the published Workflow specification from Roboflow API (no secrets logged)."""
        try:
            import requests
        except Exception as exc:  # noqa: BLE001
            log_exception_details(exc, context="requests import failed")
            return None
        try:
            resp = requests.get(
                f"https://api.roboflow.com/{workspace_name}/workflows/{workflow_id}",
                params={"api_key": self.api_key},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Could not fetch workflow specification status=%s body=%s",
                    resp.status_code,
                    sanitize_exception_text(resp.text[:300]),
                )
                return None
            payload = resp.json()
            workflow = payload.get("workflow") if isinstance(payload, dict) else None
            if not isinstance(workflow, dict):
                return None
            raw_cfg = workflow.get("lastVersionConfig") or workflow.get("config")
            if isinstance(raw_cfg, str):
                raw_cfg = json.loads(raw_cfg)
            if not isinstance(raw_cfg, dict):
                return None
            spec = raw_cfg.get("specification", raw_cfg)
            if isinstance(spec, str):
                spec = json.loads(spec)
            if not isinstance(spec, dict):
                return None
            steps = spec.get("steps") or []
            outputs = spec.get("outputs") or []
            if not steps or not outputs:
                logger.warning(
                    "Published workflow specification missing steps/outputs "
                    "(steps=%s outputs=%s)",
                    len(steps),
                    len(outputs),
                )
                return None
            return spec
        except Exception as exc:  # noqa: BLE001
            log_exception_details(
                exc, context="Failed fetching published workflow specification"
            )
            return None

    @staticmethod
    def _is_empty_workflow_payload(payload: Any) -> bool:
        return (
            isinstance(payload, list)
            and len(payload) == 1
            and isinstance(payload[0], dict)
            and len(payload[0]) == 0
        )

    def run_workflow(
        self,
        model: ModelConfig,
        image_path: str,
        prompt: str | None = None,
    ) -> Any:
        if self.demo_mode:
            self.last_source = "demo_mock"
            return load_mock_response()
        client = self._get_client()
        image_key = model.image_input_name or "image"
        # This published workflow only declares an `image` input. User detection
        # classes are applied by rewriting YOLO-World class_names in the spec —
        # do not send class_names as a workflow parameter (API rejects unknown inputs).
        parameters: dict[str, Any] = {}
        images = {image_key: image_path}
        class_names = prompt_to_class_names(prompt)
        try:
            if not hasattr(client, "run_workflow"):
                raise DetectorError("Installed inference-sdk does not support run_workflow.")

            payload = None
            invocation_mode = "workflow_id"
            used_empty_fallback = False

            # When the user provides detection classes, run the published specification
            # with class_names injected. This workflow has no runtime prompt input.
            if class_names:
                spec = self._fetch_published_workflow_specification(
                    model.workspace_name or "",
                    model.workflow_id or "",
                )
                if spec is None:
                    raise DetectorError(
                        "Could not load the published workflow specification to apply "
                        "your detection classes. Check workspace/workflow access."
                    )
                spec = inject_class_names_into_workflow_spec(spec, class_names)
                logger.info("Sending request... (YOLO-World workflow / published spec)")
                logger.info("Waiting...")
                payload = client.run_workflow(
                    specification=spec,
                    images=images,
                    parameters=None,
                    use_cache=False,
                )
                logger.info("Response received...")
                invocation_mode = "published_specification_with_prompt"
            else:
                # 1) Preferred: workspace + workflow_id
                try:
                    logger.info("Sending request... (workflow_id)")
                    logger.info("Waiting...")
                    payload = client.run_workflow(
                        workspace_name=model.workspace_name,
                        workflow_id=model.workflow_id,
                        images=images,
                        parameters=parameters or None,
                        use_cache=False,
                    )
                    logger.info("Response received...")
                except TypeError:
                    params = build_workflow_parameters(model, image_path, prompt=prompt)
                    logger.info("Sending request... (workflow_id legacy parameters)")
                    logger.info("Waiting...")
                    payload = client.run_workflow(
                        workspace_name=model.workspace_name,
                        workflow_id=model.workflow_id,
                        parameters=params,
                    )
                    logger.info("Response received...")
                    invocation_mode = "workflow_id_legacy_parameters"

                # 2) Fallback: published lastVersionConfig specification
                # Some Workspaces keep an empty draft config while lastVersionConfig is valid.
                if self._is_empty_workflow_payload(payload):
                    logger.warning(
                        "Workflow ID call returned empty output; retrying with published specification."
                    )
                    spec = self._fetch_published_workflow_specification(
                        model.workspace_name or "",
                        model.workflow_id or "",
                    )
                    if spec is not None:
                        logger.info("Sending request... (published specification fallback)")
                        logger.info("Waiting...")
                        payload = client.run_workflow(
                            specification=spec,
                            images=images,
                            parameters=parameters or None,
                            use_cache=False,
                        )
                        logger.info("Response received...")
                        invocation_mode = "published_specification"
                        used_empty_fallback = True

            self.last_source = "live_roboflow"
            self.last_invocation_mode = invocation_mode
            self.last_empty_draft_fallback = used_empty_fallback
            self.last_raw_prediction_count = len(_extract_predictions_recursive(payload))
            shape = response_shape_summary(payload)
            logger.info(
                "Live workflow ok workspace=%s workflow=%s mode=%s classes=%s shape=%s",
                model.workspace_name,
                model.workflow_id,
                invocation_mode,
                class_names or "(workflow default)",
                shape,
            )
            if self._is_empty_workflow_payload(payload):
                logger.warning(
                    "Workflow still returned an empty output object. In Roboflow, ensure "
                    "YOLO-World predictions are connected to Outputs, then Save and Deploy."
                )
            try:
                save_last_live_response(payload)
            except Exception:  # noqa: BLE001
                logger.debug("Could not persist last_live_response.json", exc_info=True)
            return payload
        except DetectorError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_exception_details(exc, context="Workflow inference failed")
            raise DetectorError(format_exception_for_user(exc)) from exc

    def run_local(
        self,
        model: ModelConfig,
        image_path: str,
    ) -> Any:
        """Run a local classical detector (no Roboflow API call)."""
        from PIL import Image

        from picket_counter import detect_fence_pickets, local_picket_response_payload

        mid = (model.model_id or "").strip().lower()
        if mid not in {"local-picket-counter", "picket-counter", "local_picket"}:
            raise DetectorError(f"Unsupported local model_id: {model.model_id!r}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise DetectorError("Local detector could not open the image file.") from exc

        detections, _warnings = detect_fence_pickets(
            image,
            source_image=Path(image_path).name,
            source_model=model.name,
        )
        self.last_source = "local_classical"
        self.last_invocation_mode = "local_picket_counter"
        self.last_empty_draft_fallback = False
        self.last_raw_prediction_count = len(detections)
        # Stash warnings for run_inference to pick up
        self.last_local_warnings = list(_warnings)
        return local_picket_response_payload(detections)

    def infer_image_path(
        self,
        model: ModelConfig,
        image_path: str,
        prompt: str | None = None,
        confidence: float = 0.4,
    ) -> Any:
        kind = (model.kind or "model").lower()
        if kind == "local":
            return self.run_local(model, image_path)
        if kind == "workflow":
            return self.run_workflow(model, image_path, prompt=prompt)
        return self.run_direct_model(model, image_path, confidence=confidence)

    def _infer_pil(
        self,
        model: ModelConfig,
        image,
        prompt: str | None,
        confidence: float,
        temp_dir: Path,
    ) -> Any:
        path = save_temp_image(image, temp_dir)
        try:
            return self.infer_image_path(model, str(path), prompt=prompt, confidence=confidence)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def _confidence_stats(detections: list[Detection]) -> tuple[float, float, float]:
    if not detections:
        return 0.0, 0.0, 0.0
    confs = [d.confidence for d in detections]
    return sum(confs) / len(confs), min(confs), max(confs)


def estimate_api_calls(
    *,
    num_images: int,
    num_models: int,
    inference_mode: str,
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_overlap: float,
    max_tiles: int = MAX_TILES_PER_IMAGE,
) -> int:
    mode = (inference_mode or "whole").lower()
    per_image_model = 1
    if "thorough" in mode or "multi-scale" in mode:
        # whole + 1024/25% + 640/30%
        from image_processing import estimate_tile_count

        t1 = min(max_tiles, estimate_tile_count(image_width, image_height, 1024, 0.25))
        t2 = min(max_tiles, estimate_tile_count(image_width, image_height, 640, 0.30))
        per_image_model = 1 + t1 + t2
    elif "tile" in mode:
        from image_processing import estimate_tile_count

        per_image_model = min(
            max_tiles,
            estimate_tile_count(image_width, image_height, tile_size, tile_overlap),
        )
    return num_images * num_models * per_image_model


def run_inference_on_prepared_image(
    detector: RoboflowDetector,
    prepared: PreparedImage,
    model: ModelConfig,
    *,
    prompt: str = "",
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    inference_mode: str = "whole",
    tile_size: int = 800,
    tile_overlap: float = 0.25,
    deduplication_strategy: str = "conservative",
    progress_callback: Callable[[str], None] | None = None,
) -> InferenceResult:
    """Full inference pipeline for one image and one model."""
    started = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []
    api_calls = 0
    tile_failures = 0
    raw_detections: list[Detection] = []
    extracted_prediction_count = 0
    saw_empty_workflow_payload = False
    request_completed = False
    error_type: str | None = None
    error_message: str | None = None

    if prepared.used_resized_copy:
        warnings.append(
            "A resized copy was used for inference; boxes are mapped back to the original."
        )

    mode = (inference_mode or "whole").lower()
    temp_root = Path(tempfile.mkdtemp(prefix="aic_"))

    def _note_payload(payload: Any) -> None:
        nonlocal extracted_prediction_count, saw_empty_workflow_payload, request_completed
        request_completed = True
        if RoboflowDetector._is_empty_workflow_payload(payload):
            saw_empty_workflow_payload = True
        extracted_prediction_count += len(_extract_predictions_recursive(payload))

    def progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    try:
        if "thorough" in mode or "multi-scale" in mode:
            warnings.append(
                "Thorough multi-scale analysis is slower and uses more API requests."
            )
            passes = [
                ("whole", None, None, "scale_whole"),
                ("tiled", 1024, 0.25, "scale_1024_25"),
                ("tiled", 640, 0.30, "scale_640_30"),
            ]
            for pass_name, tsize, toverlap, scale_id in passes:
                if pass_name == "whole":
                    progress(f"{model.name}: whole-image pass")
                    try:
                        payload = detector._infer_pil(
                            model,
                            prepared.inference,
                            prompt or None,
                            confidence_threshold,
                            temp_root,
                        )
                        api_calls += 1
                        _note_payload(payload)
                        dets = normalize_predictions(
                            payload,
                            source_model=model.name,
                            source_image=prepared.image_name,
                            image_width=prepared.inference_width,
                            image_height=prepared.inference_height,
                            confidence_threshold=confidence_threshold,
                            allowed_classes=model.allowed_classes,
                            scale_id=scale_id,
                            map_scale_x=prepared.scale_x,
                            map_scale_y=prepared.scale_y,
                            original_width=prepared.original_width,
                            original_height=prepared.original_height,
                        )
                        raw_detections.extend(dets)
                    except DetectorError as exc:
                        errors.append(str(exc))
                        error_type = error_type or "api_error"
                        error_message = error_message or str(exc)
                        warnings.append(f"Whole-image pass failed: {exc}")
                else:
                    assert tsize is not None and toverlap is not None
                    tiles, tile_warns = create_tiles(
                        prepared.inference,
                        tile_size=tsize,
                        overlap=toverlap,
                        max_tiles=MAX_TILES_PER_IMAGE,
                    )
                    warnings.extend(tile_warns)
                    for i, tile in enumerate(tiles, start=1):
                        progress(
                            f"{model.name}: {scale_id} tile {i} of {len(tiles)}"
                        )
                        if api_calls >= MAX_API_CALLS_PER_IMAGE:
                            warnings.append(
                                f"Stopped at hard max of {MAX_API_CALLS_PER_IMAGE} API calls."
                            )
                            break
                        try:
                            payload = detector._infer_pil(
                                model,
                                tile.image,
                                prompt or None,
                                confidence_threshold,
                                temp_root,
                            )
                            api_calls += 1
                            _note_payload(payload)
                            local = normalize_predictions(
                                payload,
                                source_model=model.name,
                                source_image=prepared.image_name,
                                image_width=tile.width,
                                image_height=tile.height,
                                confidence_threshold=confidence_threshold,
                                allowed_classes=model.allowed_classes,
                                tile_id=tile.tile_id,
                                scale_id=scale_id,
                            )
                            mapped = translate_tile_detections(
                                local,
                                tile,
                                scale_x=prepared.scale_x,
                                scale_y=prepared.scale_y,
                                original_width=prepared.original_width,
                                original_height=prepared.original_height,
                            )
                            raw_detections.extend(mapped)
                        except DetectorError as exc:
                            tile_failures += 1
                            error_type = error_type or "api_error"
                            error_message = error_message or str(exc)
                            warnings.append(
                                f"Tile {tile.tile_id} failed; continuing. ({exc})"
                            )
        elif "tile" in mode:
            tiles, tile_warns = create_tiles(
                prepared.inference,
                tile_size=tile_size,
                overlap=tile_overlap,
                max_tiles=MAX_TILES_PER_IMAGE,
            )
            warnings.extend(tile_warns)
            for i, tile in enumerate(tiles, start=1):
                progress(f"Processing tile {i} of {len(tiles)}")
                if api_calls >= MAX_API_CALLS_PER_IMAGE:
                    warnings.append(
                        f"Stopped at hard max of {MAX_API_CALLS_PER_IMAGE} API calls."
                    )
                    break
                try:
                    payload = detector._infer_pil(
                        model,
                        tile.image,
                        prompt or None,
                        confidence_threshold,
                        temp_root,
                    )
                    api_calls += 1
                    _note_payload(payload)
                    local = normalize_predictions(
                        payload,
                        source_model=model.name,
                        source_image=prepared.image_name,
                        image_width=tile.width,
                        image_height=tile.height,
                        confidence_threshold=confidence_threshold,
                        allowed_classes=model.allowed_classes,
                        tile_id=tile.tile_id,
                        scale_id=f"tile_{tile_size}",
                    )
                    mapped = translate_tile_detections(
                        local,
                        tile,
                        scale_x=prepared.scale_x,
                        scale_y=prepared.scale_y,
                        original_width=prepared.original_width,
                        original_height=prepared.original_height,
                    )
                    raw_detections.extend(mapped)
                except DetectorError as exc:
                    tile_failures += 1
                    error_type = error_type or "api_error"
                    error_message = error_message or str(exc)
                    warnings.append(f"Tile {tile.tile_id} failed; continuing. ({exc})")
            if tile_failures:
                warnings.append(
                    f"Partial results: {tile_failures} tile(s) failed."
                )
        else:
            progress(f"{model.name}: whole-image inference")
            try:
                payload = detector._infer_pil(
                    model,
                    prepared.inference,
                    prompt or None,
                    confidence_threshold,
                    temp_root,
                )
                api_calls += 1
                _note_payload(payload)
                raw_detections = normalize_predictions(
                    payload,
                    source_model=model.name,
                    source_image=prepared.image_name,
                    image_width=prepared.inference_width,
                    image_height=prepared.inference_height,
                    confidence_threshold=confidence_threshold,
                    allowed_classes=model.allowed_classes,
                    scale_id="whole",
                    map_scale_x=prepared.scale_x,
                    map_scale_y=prepared.scale_y,
                    original_width=prepared.original_width,
                    original_height=prepared.original_height,
                )
            except DetectorError as exc:
                errors.append(str(exc))
                error_type = "api_error"
                error_message = str(exc)
                request_completed = False
                raise
    finally:
        # Cleanup temp dir
        try:
            for p in temp_root.glob("*"):
                p.unlink(missing_ok=True)
            temp_root.rmdir()
        except Exception:  # noqa: BLE001
            pass

    if getattr(detector, "last_local_warnings", None):
        for w in detector.last_local_warnings:
            if w not in warnings:
                warnings.append(w)
        detector.last_local_warnings = []

    if detector.last_empty_draft_fallback:
        warnings.append(
            "Workflow ID returned an empty draft output; used the published "
            "workflow specification instead. Save/Deploy the draft in Roboflow "
            "to avoid this fallback."
        )

    if saw_empty_workflow_payload and extracted_prediction_count == 0:
        error_type = error_type or "empty_workflow_output"
        error_message = error_message or (
            "Workflow returned an empty output object. Check Outputs wiring and Deploy."
        )
        warnings.append(error_message)
    elif not raw_detections and not errors and request_completed:
        if extracted_prediction_count > 0:
            warnings.append(
                f"API returned {extracted_prediction_count} raw prediction(s), but none "
                f"passed local filters (confidence ≥ {confidence_threshold:.2f}"
                + (
                    f", allowed_classes={model.allowed_classes}"
                    if model.allowed_classes
                    else ""
                )
                + ")."
            )
            error_type = error_type or "filtered_to_zero"
        else:
            warnings.append(
                "No detections returned for this image/model. "
                "This is a valid zero result — the model found no matching objects."
            )

    strategy_counts = strategy_comparison_counts(raw_detections, iou_threshold)
    final = deduplicate(raw_detections, deduplication_strategy, iou_threshold)
    final = mark_overlap_and_occlusion(
        final,
        image_width=prepared.original_width,
        image_height=prepared.original_height,
    )

    # Number order: sort by center y then x for stable review
    final = sorted(final, key=lambda d: (d.center_y, d.center_x))

    avg_c, min_c, max_c = _confidence_stats(final)
    overlap_n = sum(1 for d in final if d.suspected_overlap)
    occ_n = sum(1 for d in final if d.suspected_occlusion)

    annotated = annotate_image(prepared.original, final, model_name=model.name)
    elapsed = time.perf_counter() - started

    if deduplication_strategy.lower() in {"none", "none/debug", "debug"}:
        warnings.append("Debug mode: duplicate removal disabled; counts may include duplicates.")

    success = not errors and error_type not in {"empty_workflow_output", "api_error"}
    return InferenceResult(
        image_name=prepared.image_name,
        model_name=model.name,
        prompt=prompt,
        inference_mode=inference_mode,
        deduplication_strategy=deduplication_strategy,
        detections=final,
        raw_count=len(raw_detections),
        final_count=len(final),
        duplicates_removed=max(0, len(raw_detections) - len(final)),
        avg_confidence=avg_c,
        min_confidence=min_c,
        max_confidence=max_c,
        suspected_overlap_count=overlap_n,
        suspected_occlusion_count=occ_n,
        processing_time_seconds=elapsed,
        warnings=warnings,
        errors=errors,
        used_resized_copy=prepared.used_resized_copy,
        annotated_image_bytes=image_to_png_bytes(annotated),
        strategy_counts=strategy_counts,
        api_calls_used=api_calls,
        tile_failures=tile_failures,
        success=success,
        source=detector.last_source,
        request_completed=request_completed,
        predictions_found=extracted_prediction_count > 0,
        error_type=error_type,
        error_message=error_message,
        raw_prediction_count=extracted_prediction_count,
        normalized_prediction_count=len(raw_detections),
        invocation_mode=detector.last_invocation_mode,
    )
