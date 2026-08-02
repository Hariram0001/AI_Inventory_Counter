"""Direct OpenRouter VLM object detection (not Roboflow Workflow parsing).

Calls OpenRouter ``/api/v1/chat/completions`` with an image + strict JSON
schema (one detection per object with a bounding box), then converts into the
same ``Detection`` objects used by YOLO-World / Review & Save.
Never logs API keys.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import config
from schemas import Detection, InferenceResult
from security import redact_text

DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.6-luna"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# YOLO-style: one entry per visible object, with a bounding box.
DETECTION_JSON_SCHEMA_HINT = """{
  "detections": [
    {
      "class_name": "fence_picket",
      "confidence": 0.85,
      "x1": 120,
      "y1": 40,
      "x2": 180,
      "y2": 520
    }
  ],
  "total_count": 1,
  "warnings": [],
  "summary": "brief result"
}"""

# Backward-compatible alias used by older imports/tests.
COUNT_JSON_SCHEMA_HINT = DETECTION_JSON_SCHEMA_HINT


@dataclass
class OpenRouterVLMTechnicalDetails:
    selected_model: str = ""
    http_status: int | None = None
    response_type: str = ""
    response_preview: str = ""
    parser_stage: str = ""
    retryable: bool = False
    top_level_keys: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedInventoryCount:
    """Parsed VLM inventory response (object-detection style)."""

    detections: list[dict[str, Any]]
    total_count: int
    warnings: list[str] = field(default_factory=list)
    summary: str = ""
    raw_content: str = ""


@dataclass
class OpenRouterVLMCallResult:
    ok: bool
    parsed: ParsedInventoryCount | None = None
    technical: OpenRouterVLMTechnicalDetails = field(
        default_factory=OpenRouterVLMTechnicalDetails
    )
    error_message: str = ""


class OpenRouterVLMError(Exception):
    """User-facing OpenRouter VLM failure with sanitized technical details."""

    def __init__(
        self,
        message: str,
        *,
        technical: OpenRouterVLMTechnicalDetails | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.technical = technical or OpenRouterVLMTechnicalDetails(
            retryable=retryable, error_message=message
        )
        self.technical.retryable = retryable
        if not self.technical.error_message:
            self.technical.error_message = message


def configured_openrouter_model_id() -> str:
    return str(
        getattr(config, "OPENROUTER_MODEL_ID", "") or DEFAULT_OPENROUTER_MODEL
    ).strip() or DEFAULT_OPENROUTER_MODEL


def build_inventory_detection_prompt(
    class_names: list[str],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> str:
    classes = [str(c).strip() for c in class_names if str(c).strip()]
    if not classes:
        classes = ["inventory_item"]
    class_list = ", ".join(classes)
    size_line = ""
    if image_width and image_height and image_width > 0 and image_height > 0:
        size_line = (
            f"- Image size is {int(image_width)}x{int(image_height)} pixels. "
            "Bounding box coordinates MUST use this pixel space "
            "(origin at top-left; x right, y down).\n"
        )
    return (
        "You are an object-detection model for inventory photos. "
        "Your job is to find EVERY individual item that matches these classes: "
        f"{class_list}.\n\n"
        "Goal: complete individual counting — arrangement must not matter.\n"
        "Detect items whether they are stacked, nested, scattered, overlapping, "
        "tilted, sideways, upright, distant, close-up, partially occluded, or "
        "in unusual positions/angles.\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON. No markdown, no prose outside JSON.\n"
        "- Use this exact schema:\n"
        f"{DETECTION_JSON_SCHEMA_HINT}\n"
        "- `detections` is a JSON array with ONE entry per individual physical item "
        "(like YOLO). Never merge many items into one detection or one count row.\n"
        "- If multiple classes are listed, treat each class as a SEPARATE item type. "
        "Never merge different classes into one detection (e.g. do not combine "
        "traffic_cone and barrel).\n"
        "- Do NOT treat a stack, pile, bundle, nest, or group as a single object.\n"
        "- Count every individual item you can identify, including partially hidden "
        "ones (visible edges, rims, bases, corners, collars, or silhouettes).\n"
        "- Scattered / spread-out items each get their own detection.\n"
        "- Overlapping items each get their own detection when separable.\n"
        "- Place a tight bounding box on each individual item "
        "(even if only part of it is visible).\n"
        "- Each detection MUST include a bounding box as pixel coordinates: "
        "`x1`, `y1`, `x2`, `y2` (top-left and bottom-right corners).\n"
        "- You may alternatively provide `bbox` as [x1,y1,x2,y2] or center "
        "format `x`,`y`,`width`,`height`.\n"
        f"{size_line}"
        "- Boxes must stay inside the image.\n"
        "- `confidence` must be between 0 and 1. Use lower confidence when an item "
        "is heavily occluded or the nested count is uncertain, but still emit one "
        "detection per estimated individual item.\n"
        "- `total_count` must equal the number of detection objects.\n"
        "- Prefer the provided class names; normalize spaces to underscores if needed.\n"
        "- If nothing matches, return an empty detections array and total_count 0.\n"
    )


def build_inventory_count_prompt(
    class_names: list[str],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> str:
    """Backward-compatible name — now builds the object-detection prompt."""
    return build_inventory_detection_prompt(
        class_names, image_width=image_width, image_height=image_height
    )


def _guess_mime(image_bytes: bytes, filename: str = "") -> str:
    name = (filename or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def encode_image_data_url(image_bytes: bytes, filename: str = "") -> str:
    mime = _guess_mime(image_bytes, filename)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extract_message_content(payload: Any) -> tuple[str, str]:
    """Return (content_text, parser_stage)."""
    if not isinstance(payload, dict):
        return "", "non_object_response"
    choices = payload.get("choices")
    if choices is None:
        return "", "missing_choices"
    if not isinstance(choices, list) or not choices:
        return "", "empty_choices"
    first = choices[0]
    if not isinstance(first, dict):
        return "", "invalid_choice"
    message = first.get("message") or first.get("delta") or {}
    if not isinstance(message, dict):
        content = first.get("content")
    else:
        content = message.get("content")
    if content is None:
        return "", "missing_content"
    if isinstance(content, str):
        return content.strip(), "string_content"
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif block.get("type") == "output_text" and isinstance(
                    block.get("text"), str
                ):
                    parts.append(block["text"])
        joined = "\n".join(p.strip() for p in parts if p and str(p).strip())
        return joined.strip(), "content_blocks"
    return str(content).strip(), "coerced_content"


def _strip_fences(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    match = JSON_FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def _as_nonneg_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = int(float(value))
        if number < 0:
            return None
        return number
    except (TypeError, ValueError):
        return None


def _as_confidence(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        conf = float(value)
        # Accept 0-100 percentages from some models.
        if 1.0 < conf <= 100.0:
            conf = conf / 100.0
        if conf < 0.0 or conf > 1.0:
            return None
        return conf
    except (TypeError, ValueError):
        return None


def _parse_error(stage: str, preview: str = "", *, retryable: bool = True) -> OpenRouterVLMError:
    return OpenRouterVLMError(
        "OpenRouter returned a response, but it could not be parsed into a "
        "valid inventory count.",
        technical=OpenRouterVLMTechnicalDetails(
            parser_stage=stage,
            response_preview=redact_text(preview, max_len=400),
            retryable=retryable,
        ),
        retryable=retryable,
    )


def _item_has_box_fields(item: dict[str, Any]) -> bool:
    if all(k in item for k in ("x1", "y1", "x2", "y2")):
        return True
    if all(k in item for k in ("xmin", "ymin", "xmax", "ymax")):
        return True
    if all(k in item for k in ("x", "y", "width", "height")):
        return True
    bbox = item.get("bbox") or item.get("bounding_box") or item.get("box")
    if isinstance(bbox, dict):
        return _item_has_box_fields(bbox)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return True
    return False


def parse_inventory_count_json(
    content: str,
    *,
    image_width: float | None = None,
    image_height: float | None = None,
) -> ParsedInventoryCount:
    """Parse VLM JSON into per-object detections (with boxes when present)."""
    from detector import _box_from_prediction

    cleaned = _strip_fences(content)
    if not cleaned:
        raise _parse_error("empty_content")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        err = _parse_error("invalid_json", cleaned)
        err.technical.error_message = redact_text(str(exc), max_len=160)
        raise err from exc

    if not isinstance(data, dict):
        raise _parse_error("json_not_object", cleaned)

    raw_detections = data.get("detections", [])
    if raw_detections is None:
        raw_detections = []
    if not isinstance(raw_detections, list):
        raise _parse_error("detections_not_list", cleaned)

    warnings: list[str] = []
    if isinstance(data.get("warnings"), list):
        warnings.extend(str(w) for w in data["warnings"] if str(w).strip())
    elif data.get("warnings") not in (None, ""):
        warnings.append("Ignored non-list warnings field from model response.")

    iw = float(image_width or 0.0)
    ih = float(image_height or 0.0)
    # When size unknown, still allow absolute pixel boxes; denorm uses >1.5 check.
    if iw <= 0:
        iw = 1.0
    if ih <= 0:
        ih = 1.0

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_detections):
        if not isinstance(item, dict):
            warnings.append(f"Skipped non-object detection at index {idx}.")
            continue
        class_name = (
            str(
                item.get("class_name")
                or item.get("class")
                or item.get("label")
                or "inventory_item"
            )
            .strip()
            .replace(" ", "_")
        ) or "inventory_item"

        conf = _as_confidence(item.get("confidence"))
        if item.get("confidence") is not None and conf is None:
            warnings.append(
                f"Ignored invalid confidence for '{class_name}' (must be 0–1)."
            )
        confidence = 0.5 if conf is None else conf
        notes = str(item.get("notes") or "").strip()

        # Preferred path: YOLO-style one object + box.
        if _item_has_box_fields(item):
            box = _box_from_prediction(item, iw, ih)
            if box is None:
                warnings.append(
                    f"Skipped detection '{class_name}' at index {idx}: invalid box."
                )
                continue
            x1, y1, x2, y2 = box
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 1.0 or height <= 1.0:
                warnings.append(
                    f"Skipped detection '{class_name}' at index {idx}: zero-area box."
                )
                continue
            normalized.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "notes": notes,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "count_only": False,
                    "item_count": 1,
                }
            )
            continue

        # Legacy count-only row: expand into N placeholder objects (no boxes).
        count = _as_nonneg_int(item.get("count"))
        if count is None:
            warnings.append(
                f"Skipped detection '{class_name}' at index {idx}: "
                "missing bounding box and invalid count."
            )
            continue
        if count == 0:
            continue
        warnings.append(
            f"Detection '{class_name}' had a count ({count}) but no bounding box; "
            "expanded as count-only (no boxes drawn)."
        )
        for _ in range(count):
            normalized.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "notes": notes,
                    "x1": 0.0,
                    "y1": 0.0,
                    "x2": 0.0,
                    "y2": 0.0,
                    "count_only": True,
                    "item_count": 1,
                }
            )

    boxed = [d for d in normalized if not d.get("count_only")]
    count_only = [d for d in normalized if d.get("count_only")]
    # Prefer boxed objects for the total when any exist.
    if boxed and count_only:
        warnings.append(
            "Mixed boxed and count-only rows; total uses boxed object detections."
        )
        normalized = boxed
    summed = len(normalized)
    reported_total = _as_nonneg_int(data.get("total_count"))
    if reported_total is None:
        total = summed
        if data.get("total_count") is not None:
            warnings.append("Ignored invalid total_count; recalculated from detections.")
    elif reported_total != summed:
        warnings.append(
            f"Model total_count ({reported_total}) did not match number of "
            f"detections ({summed}); using recalculated total."
        )
        total = summed
    else:
        total = reported_total

    summary = str(data.get("summary") or "").strip()
    return ParsedInventoryCount(
        detections=normalized,
        total_count=total,
        warnings=warnings,
        summary=summary,
        raw_content=cleaned,
    )


def detections_from_parsed(
    parsed: ParsedInventoryCount,
    *,
    model_name: str,
    image_name: str,
    map_scale_x: float = 1.0,
    map_scale_y: float = 1.0,
    original_width: float | None = None,
    original_height: float | None = None,
) -> list[Detection]:
    """Build YOLO-style Detection rows (one per object)."""
    from image_processing import map_box_to_original

    out: list[Detection] = []
    ow = float(original_width) if original_width else None
    oh = float(original_height) if original_height else None

    for idx, item in enumerate(parsed.detections):
        count_only = bool(item.get("count_only"))
        x1 = float(item.get("x1") or 0.0)
        y1 = float(item.get("y1") or 0.0)
        x2 = float(item.get("x2") or 0.0)
        y2 = float(item.get("y2") or 0.0)

        if not count_only:
            if map_scale_x != 1.0 or map_scale_y != 1.0 or ow is not None:
                x1, y1, x2, y2 = map_box_to_original(
                    x1,
                    y1,
                    x2,
                    y2,
                    map_scale_x,
                    map_scale_y,
                    int(ow or max(x2, 1)),
                    int(oh or max(y2, 1)),
                )
            else:
                if ow is not None:
                    x1 = max(0.0, min(ow, x1))
                    x2 = max(0.0, min(ow, x2))
                if oh is not None:
                    y1 = max(0.0, min(oh, y1))
                    y2 = max(0.0, min(oh, y2))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width <= 0 or height <= 0:
                continue
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
        else:
            width = height = center_x = center_y = 0.0

        out.append(
            Detection(
                detection_id=f"openrouter-od-{idx + 1}",
                class_name=str(item.get("class_name") or "inventory_item"),
                confidence=float(item.get("confidence") or 0.5),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                source_model=model_name,
                source_image=image_name,
                included_in_count=True,
                count_only=count_only,
                item_count=1,
            )
        )
    return out


def call_openrouter_vlm(
    *,
    api_key: str,
    image_bytes: bytes,
    class_names: list[str],
    model_id: str | None = None,
    image_name: str = "upload.jpg",
    image_width: int | None = None,
    image_height: int | None = None,
    timeout: float = 120.0,
) -> OpenRouterVLMCallResult:
    """Perform one real OpenRouter chat completion for object detection."""
    import requests

    selected = (model_id or configured_openrouter_model_id()).strip()
    tech = OpenRouterVLMTechnicalDetails(selected_model=selected)
    key = str(api_key or "").strip()
    if not key:
        tech.parser_stage = "missing_api_key"
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message=(
                "OpenRouter is not configured. An administrator must add an "
                "API key before this model can run."
            ),
        )
    if not image_bytes:
        tech.parser_stage = "missing_image"
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message="No image bytes were provided for OpenRouter analysis.",
        )

    # Infer dimensions from bytes when not provided (helps box denormalization).
    if not image_width or not image_height:
        try:
            import io

            from PIL import Image as PILImage

            with PILImage.open(io.BytesIO(image_bytes)) as im:
                image_width, image_height = im.size
        except Exception:  # noqa: BLE001
            pass

    prompt = build_inventory_detection_prompt(
        class_names, image_width=image_width, image_height=image_height
    )
    data_url = encode_image_data_url(image_bytes, image_name)
    body = {
        "model": selected,
        "temperature": 0.1,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Hariram0001/AI_Inventory_Counter",
        "X-Title": "AI Inventory Counter",
    }

    try:
        response = requests.post(
            OPENROUTER_CHAT_URL,
            headers=headers,
            json=body,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        tech.parser_stage = "network_error"
        tech.retryable = True
        tech.error_message = redact_text(f"{type(exc).__name__}: {exc}", max_len=200)
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message=(
                "Could not reach OpenRouter. Check the network connection and try again."
            ),
        )

    tech.http_status = int(response.status_code)
    tech.response_type = str(response.headers.get("content-type") or "")
    preview = redact_text(response.text or "", max_len=500)
    tech.response_preview = preview

    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = None
    if isinstance(payload, dict):
        tech.top_level_keys = sorted(str(k) for k in payload.keys())
        usage = payload.get("usage")
        if isinstance(usage, dict):
            tech.usage = {
                k: usage.get(k)
                for k in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cost",
                )
                if k in usage
            }

    if response.status_code in (401, 403):
        tech.parser_stage = "auth_rejected"
        tech.retryable = False
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message=(
                "OpenRouter rejected the current session key. Reconnect it in "
                "API Connections."
            ),
        )
    if response.status_code == 429:
        tech.parser_stage = "rate_limited"
        tech.retryable = True
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message="OpenRouter is rate limiting this key. Try again shortly.",
        )
    if response.status_code == 402:
        tech.parser_stage = "insufficient_credit"
        tech.retryable = False
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message=(
                "Your OpenRouter account does not have enough credit for this run."
            ),
        )
    if response.status_code >= 400 or not isinstance(payload, dict):
        tech.parser_stage = "provider_error"
        tech.retryable = response.status_code >= 500
        err = ""
        if isinstance(payload, dict):
            error_obj = payload.get("error")
            if isinstance(error_obj, dict):
                err = str(error_obj.get("message") or error_obj.get("code") or "")
            elif error_obj:
                err = str(error_obj)
        tech.error_message = redact_text(err or preview, max_len=240)
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message=(
                "OpenRouter returned an error while analyzing the image. "
                + (tech.error_message[:180] if tech.error_message else "")
            ).strip(),
        )

    content, stage = extract_message_content(payload)
    tech.parser_stage = stage
    tech.response_preview = redact_text(content or preview, max_len=500)
    if not content:
        tech.retryable = True
        return OpenRouterVLMCallResult(
            ok=False,
            technical=tech,
            error_message=(
                "OpenRouter returned a response, but it could not be parsed into a "
                "valid inventory count."
            ),
        )

    try:
        parsed = parse_inventory_count_json(
            content, image_width=image_width, image_height=image_height
        )
    except OpenRouterVLMError as exc:
        details = exc.technical
        details.selected_model = selected
        details.http_status = tech.http_status
        details.response_type = tech.response_type
        details.top_level_keys = tech.top_level_keys
        details.usage = tech.usage
        if not details.response_preview:
            details.response_preview = tech.response_preview
        return OpenRouterVLMCallResult(
            ok=False,
            technical=details,
            error_message=str(exc),
        )

    tech.parser_stage = "parsed_ok"
    return OpenRouterVLMCallResult(ok=True, parsed=parsed, technical=tech)


def build_count_inference_result(
    *,
    parsed: ParsedInventoryCount,
    model_name: str,
    image_name: str,
    prompt: str,
    processing_time_seconds: float,
    technical: OpenRouterVLMTechnicalDetails | None = None,
    annotated_image_bytes: bytes | None = None,
    map_scale_x: float = 1.0,
    map_scale_y: float = 1.0,
    original_width: float | None = None,
    original_height: float | None = None,
) -> InferenceResult:
    detections = detections_from_parsed(
        parsed,
        model_name=model_name,
        image_name=image_name,
        map_scale_x=map_scale_x,
        map_scale_y=map_scale_y,
        original_width=original_width,
        original_height=original_height,
    )
    total = len(detections)
    confs = [d.confidence for d in detections]
    avg = sum(confs) / len(confs) if confs else 0.0
    warnings = list(parsed.warnings)
    if parsed.summary:
        warnings.append(f"Summary: {parsed.summary}")
    boxed = sum(1 for d in detections if not getattr(d, "count_only", False))
    if boxed:
        warnings.append(
            f"OpenRouter VLM returned {boxed} object detection(s) with bounding boxes."
        )
    elif detections:
        warnings.append(
            "OpenRouter VLM returned count-only results (no bounding boxes)."
        )
    return InferenceResult(
        image_name=image_name,
        model_name=model_name,
        prompt=prompt,
        inference_mode="openrouter_vlm_detection",
        deduplication_strategy="None",
        detections=detections,
        raw_count=total,
        final_count=total,
        duplicates_removed=0,
        avg_confidence=avg,
        min_confidence=min(confs) if confs else 0.0,
        max_confidence=max(confs) if confs else 0.0,
        suspected_overlap_count=0,
        suspected_occlusion_count=0,
        processing_time_seconds=float(processing_time_seconds),
        warnings=warnings,
        annotated_image_bytes=annotated_image_bytes,
        success=True,
        source="openrouter_vlm",
        request_completed=True,
        predictions_found=total > 0,
        raw_prediction_count=total,
        normalized_prediction_count=len(detections),
        invocation_mode="openrouter_chat_completions",
    )


def run_openrouter_vlm_on_prepared_image(
    *,
    api_key: str,
    prepared_image: Any,
    model_name: str,
    class_names: list[str],
    model_id: str | None = None,
) -> tuple[InferenceResult, OpenRouterVLMTechnicalDetails]:
    """Entry point used by OpenRouterVLMAdapter (same path as Streamlit Analyze)."""
    import io

    from PIL import Image as PILImage

    started = time.perf_counter()
    image_name = str(getattr(prepared_image, "image_name", "") or "upload.jpg")
    working = (
        getattr(prepared_image, "inference", None)
        or getattr(prepared_image, "original", None)
        or getattr(prepared_image, "working_image", None)
    )
    if working is None:
        raise OpenRouterVLMError(
            "No image data available for OpenRouter analysis.",
            technical=OpenRouterVLMTechnicalDetails(parser_stage="missing_image"),
        )
    if not isinstance(working, PILImage.Image):
        working = PILImage.fromarray(working)
    buf = io.BytesIO()
    working.convert("RGB").save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()
    iw, ih = working.size

    result = call_openrouter_vlm(
        api_key=api_key,
        image_bytes=bytes(image_bytes),
        class_names=class_names,
        model_id=model_id,
        image_name=image_name,
        image_width=iw,
        image_height=ih,
    )
    elapsed = time.perf_counter() - started
    if not result.ok or result.parsed is None:
        raise OpenRouterVLMError(
            result.error_message
            or (
                "OpenRouter returned a response, but it could not be parsed into a "
                "valid inventory count."
            ),
            technical=result.technical,
            retryable=result.technical.retryable,
        )

    original = getattr(prepared_image, "original", None) or working
    ow = int(getattr(prepared_image, "original_width", None) or original.size[0])
    oh = int(getattr(prepared_image, "original_height", None) or original.size[1])
    scale_x = float(getattr(prepared_image, "scale_x", None) or (ow / float(iw or 1)))
    scale_y = float(getattr(prepared_image, "scale_y", None) or (oh / float(ih or 1)))

    annotated = None
    try:
        from image_processing import annotate_image

        dets = detections_from_parsed(
            result.parsed,
            model_name=model_name,
            image_name=image_name,
            map_scale_x=scale_x,
            map_scale_y=scale_y,
            original_width=ow,
            original_height=oh,
        )
        annotated_img = annotate_image(original, dets, model_name=model_name)
        out = io.BytesIO()
        annotated_img.save(out, format="JPEG", quality=90)
        annotated = out.getvalue()
    except Exception:  # noqa: BLE001
        annotated = None

    inference = build_count_inference_result(
        parsed=result.parsed,
        model_name=model_name,
        image_name=image_name,
        prompt=", ".join(class_names),
        processing_time_seconds=elapsed,
        technical=result.technical,
        annotated_image_bytes=annotated,
        map_scale_x=scale_x,
        map_scale_y=scale_y,
        original_width=ow,
        original_height=oh,
    )
    return inference, result.technical


__all__ = [
    "COUNT_JSON_SCHEMA_HINT",
    "DETECTION_JSON_SCHEMA_HINT",
    "DEFAULT_OPENROUTER_MODEL",
    "OPENROUTER_CHAT_URL",
    "OpenRouterVLMCallResult",
    "OpenRouterVLMError",
    "OpenRouterVLMTechnicalDetails",
    "ParsedInventoryCount",
    "build_inventory_count_prompt",
    "build_inventory_detection_prompt",
    "call_openrouter_vlm",
    "configured_openrouter_model_id",
    "detections_from_parsed",
    "encode_image_data_url",
    "extract_message_content",
    "parse_inventory_count_json",
    "run_openrouter_vlm_on_prepared_image",
]
