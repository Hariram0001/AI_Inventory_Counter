"""Generic detection benchmark: metrics, storage, prompt sets, profile promotion.

Isolated from the inventory wizard session. No Streamlit dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inventory_profiles import (
    MAX_PROMPTS,
    clear_profiles_cache,
    normalize_prompts,
    prompts_to_csv,
    validate_prompts,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARKS_PATH = PROJECT_ROOT / "data" / "benchmarks.json"
PROFILES_PATH = PROJECT_ROOT / "inventory_profiles.json"
PROFILE_BACKUPS_DIR = PROJECT_ROOT / "data" / "inventory_profile_backups"

MAX_PROMPT_SETS = 3
MAX_BATCH_IMAGES = 20
MAX_THRESHOLDS = 8
DEFAULT_SWEEP_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)
INFERENCE_CONFIRM_THRESHOLD = 30
MIN_RECOMMENDATION_RUNS = 2
ADAPTER_VERSION = "yolo_world_benchmark_v1"

DETECTION_LABELS = (
    "correct",
    "false_positive",
    "wrong_class",
    "duplicate",
    "ignore",
)

RECOMMENDED_BENCHMARK_INVENTORIES = (
    "Fence Panel",
    "Traffic Cones",
    "Chairs",
    "Boxes",
    "Pallets",
    "Cars",
    "Bottles",
    "Gates",
    "Poles",
    "Custom Item",
)

PROMPT_QUALITY_GUIDANCE = """
Good detection prompts describe the individual countable object, not the entire scene or structure.

**Better**
- individual traffic cone
- cardboard box
- wooden pallet

**Potentially ambiguous**
- road equipment
- warehouse items
- fence

**Fence Panels:** the term *fence* may produce one box around the whole structure;
*individual fence panel* / *wooden fence panel* may better match the intended counting unit.
Prompt wording alone does not guarantee correct granularity.
""".strip()

THRESHOLD_WARNING = """
**Confidence threshold note:** Low-confidence detections can be valid, but lowering the
threshold may also increase false positives.

One-image evidence (not a universal recommendation): a valid-looking fence detection was
approximately **24.9%**. Threshold **0.25** removed it; threshold **0.20** retained it.
""".strip()

DATASET_GUIDANCE = """
**Dataset size (guidance, not a guarantee):** for each inventory type, collect at least
**5** images for an initial check and **15–30** for a more meaningful POC benchmark.
Vary distance, angle, lighting, background, occlusion, object size, crowding, and stacking.
One successful image does not prove general accuracy.
""".strip()


@dataclass
class BenchmarkMetrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float | None = None
    recall: float | None = None
    count_error: int | None = None
    count_accuracy: float | None = None
    evaluation: str = "not_evaluated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_prompt_sets(
    raw_sets: list[str] | list[list[str]] | None,
    *,
    max_sets: int = MAX_PROMPT_SETS,
) -> tuple[list[list[str]], list[str]]:
    """Parse up to ``max_sets`` independent prompt lists.

    Each entry may be a CSV/newline string or a list of terms.
    """
    errors: list[str] = []
    if not raw_sets:
        return [], ["At least one prompt set is required."]
    if len(raw_sets) > max_sets:
        errors.append(f"At most {max_sets} prompt sets are allowed.")
        raw_sets = list(raw_sets)[:max_sets]

    parsed: list[list[str]] = []
    for i, item in enumerate(raw_sets, start=1):
        if isinstance(item, str):
            terms = normalize_prompts(item)
        else:
            terms = normalize_prompts(list(item or []))
        safe, set_errs = validate_prompts(terms)
        if set_errs and not safe:
            errors.append(f"Prompt set {i}: " + "; ".join(set_errs))
            continue
        for e in set_errs:
            errors.append(f"Prompt set {i}: {e}")
        if safe:
            parsed.append(safe)
    if not parsed:
        errors.append("No valid prompt sets remain.")
    return parsed, errors


def validate_expected_count(value: Any) -> tuple[int | None, list[str]]:
    """Validate ground-truth expected count (non-negative integer)."""
    errors: list[str] = []
    if value is None or value == "":
        return None, ["Expected object count is required."]
    try:
        if isinstance(value, bool):
            raise ValueError("bool")
        n = int(value)
    except (TypeError, ValueError):
        return None, ["Expected object count must be an integer ≥ 0."]
    if n < 0:
        return None, ["Expected object count must be an integer ≥ 0."]
    return n, errors


def compute_detection_counts(
    labels: list[str] | None,
    *,
    missed_count: int = 0,
) -> tuple[int, int, int]:
    """Return (TP, FP, FN) from per-detection labels and missed objects."""
    tp = fp = 0
    for lab in labels or []:
        key = str(lab or "").strip().lower()
        if key == "correct":
            tp += 1
        elif key in {"false_positive", "wrong_class", "duplicate"}:
            fp += 1
        # ignore → neither
    fn = max(0, int(missed_count or 0))
    return tp, fp, fn


def compute_benchmark_metrics(
    *,
    ai_count: int,
    expected_count: int | None,
    labels: list[str] | None = None,
    missed_count: int = 0,
    execution_failed: bool = False,
    request_completed: bool = True,
) -> BenchmarkMetrics:
    """Image-specific benchmark metrics (not universal model accuracy)."""
    m = BenchmarkMetrics()
    if execution_failed or not request_completed:
        m.evaluation = "execution_failed"
        if expected_count is not None:
            m.count_error = abs(int(ai_count) - int(expected_count))
        return m

    tp, fp, fn = compute_detection_counts(labels, missed_count=missed_count)
    m.true_positives = tp
    m.false_positives = fp
    m.false_negatives = fn

    denom_p = tp + fp
    m.precision = (tp / denom_p) if denom_p > 0 else None
    denom_r = tp + fn
    m.recall = (tp / denom_r) if denom_r > 0 else None

    if expected_count is None:
        m.evaluation = "not_evaluated"
        return m

    exp = int(expected_count)
    ai = int(ai_count)
    m.count_error = abs(ai - exp)
    if exp == 0:
        # Safe zero handling: perfect only when AI also reports zero.
        m.count_accuracy = 1.0 if ai == 0 else 0.0
        if ai == 0:
            m.evaluation = "successful_zero_detections"
        else:
            m.evaluation = "overcount"
        return m

    m.count_accuracy = max(0.0, 1.0 - (m.count_error / float(exp)))
    if ai == exp:
        m.evaluation = "exact_count_match"
    elif ai < exp:
        m.evaluation = "undercount"
    else:
        m.evaluation = "overcount"
    if ai == 0 and exp > 0 and request_completed and not execution_failed:
        # Still undercount, but expose successful-zero execution nuance via status helpers.
        pass
    return m


def evaluation_label(evaluation: str) -> str:
    return {
        "exact_count_match": "Exact count match",
        "undercount": "Undercount",
        "overcount": "Overcount",
        "execution_failed": "Execution failed",
        "successful_zero_detections": "Successful zero detections",
        "not_evaluated": "Not evaluated",
    }.get(evaluation, evaluation or "Unknown")


def image_content_hash(data: bytes) -> str:
    return hashlib.sha256(data or b"").hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def stamp_owner(
    record: dict[str, Any],
    *,
    user_id: int | None,
    username: str = "",
) -> dict[str, Any]:
    """Attach the signing-in user to a benchmark record before it is stored."""
    out = dict(record or {})
    out["user_id"] = int(user_id) if user_id is not None else None
    out["username"] = str(username or "")
    return out


def owned_by_user(
    records: list[dict[str, Any]],
    *,
    user_id: int | None,
    is_admin: bool = False,
) -> list[dict[str, Any]]:
    """Restrict benchmark history to the caller's own rows.

    Administrators see everything, including rows written before benchmark
    records carried an owner. Regular users see only rows they own, so one
    user's runs never appear in another's history.
    """
    if is_admin:
        return list(records or [])
    if user_id is None:
        return []
    owner = int(user_id)
    return [
        row
        for row in (records or [])
        if isinstance(row, dict) and _record_owner_id(row) == owner
    ]


def _record_owner_id(record: dict[str, Any]) -> int | None:
    raw = record.get("user_id")
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def load_benchmark_results(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_BENCHMARKS_PATH
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict):
        items = raw.get("results") or []
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    return [r for r in items if isinstance(r, dict)]


def save_benchmark_result(
    result: dict[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append a sanitized benchmark result. Never stores API keys or image bytes."""
    p = path or DEFAULT_BENCHMARKS_PATH
    clean = sanitize_benchmark_record(result)
    if not clean.get("benchmark_id"):
        clean["benchmark_id"] = str(uuid.uuid4())
    if not clean.get("timestamp"):
        clean["timestamp"] = datetime.now(timezone.utc).isoformat()
    existing = load_benchmark_results(p)
    existing.append(clean)
    _atomic_write_json(p, {"version": 1, "results": existing})
    return clean


def sanitize_benchmark_record(result: dict[str, Any]) -> dict[str, Any]:
    banned = {
        "api_key",
        "ROBOFLOW_API_KEY",
        "authorization",
        "token",
        "secret",
        "image_bytes",
        "annotated_image_bytes",
        "raw_payload",
    }
    out: dict[str, Any] = {}
    for key, value in (result or {}).items():
        lk = str(key).lower()
        if key in banned or any(b in lk for b in ("api_key", "secret", "token", "authorization")):
            continue
        if lk.endswith("_bytes") or lk in {"image_bytes", "annotated_image_bytes"}:
            continue
        out[str(key)] = value
    # Canonical fields (may be None)
    defaults = {
        "benchmark_id": out.get("benchmark_id"),
        "timestamp": out.get("timestamp"),
        "image_hash": out.get("image_hash"),
        "image_source": out.get("image_source"),
        "image_name": out.get("image_name"),
        "inventory_key": out.get("inventory_key"),
        "custom_item_name": out.get("custom_item_name"),
        "prompt_set": list(out.get("prompt_set") or []),
        "prompt_set_label": out.get("prompt_set_label"),
        "expected_count": out.get("expected_count"),
        "raw_count": out.get("raw_count"),
        "final_count": out.get("final_count"),
        "true_positives": out.get("true_positives"),
        "false_positives": out.get("false_positives"),
        "false_negatives": out.get("false_negatives"),
        "precision": out.get("precision"),
        "recall": out.get("recall"),
        "count_error": out.get("count_error"),
        "count_accuracy": out.get("count_accuracy"),
        "processing_time": out.get("processing_time"),
        "returned_classes": list(out.get("returned_classes") or []),
        "invocation_mode": out.get("invocation_mode"),
        "fallback_used": bool(out.get("fallback_used")),
        "model_key": out.get("model_key"),
        "notes": out.get("notes") or "",
        "object_definition": out.get("object_definition") or "",
        "evaluation": out.get("evaluation"),
        "detection_labels": list(out.get("detection_labels") or []),
        "technical": dict(out.get("technical") or {}),
        "confidence_threshold": out.get("confidence_threshold"),
        "reviewed": bool(out.get("reviewed", False)),
        "session_id": out.get("session_id"),
        "cached": bool(out.get("cached", False)),
        "record_kind": out.get("record_kind") or "single",
        "user_id": _record_owner_id(out),
        "username": str(out.get("username") or ""),
    }
    # Drop null-only noise from technical secrets
    tech = defaults["technical"]
    if isinstance(tech, dict):
        defaults["technical"] = {
            k: v
            for k, v in tech.items()
            if "api_key" not in str(k).lower() and "secret" not in str(k).lower()
        }
    return defaults


def filter_benchmark_history(
    results: list[dict[str, Any]],
    *,
    inventory_key: str | None = None,
    exact_match: bool = False,
    overcount: bool = False,
    undercount: bool = False,
    failed: bool = False,
) -> list[dict[str, Any]]:
    out = list(results)
    if inventory_key and inventory_key != "(all)":
        out = [r for r in out if r.get("inventory_key") == inventory_key]
    status_filters = []
    if exact_match:
        status_filters.append("exact_count_match")
    if overcount:
        status_filters.append("overcount")
    if undercount:
        status_filters.append("undercount")
    if failed:
        status_filters.append("execution_failed")
    if status_filters:
        out = [r for r in out if r.get("evaluation") in status_filters]
    # Newest first
    out.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    return out


def backup_inventory_profiles(*, reason: str = "prompt_promotion") -> Path | None:
    """Copy current profiles JSON into data/inventory_profile_backups/."""
    if not PROFILES_PATH.exists():
        return None
    PROFILE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", reason)[:40]
    dest = PROFILE_BACKUPS_DIR / f"inventory_profiles_{stamp}_{safe_reason}.json"
    dest.write_bytes(PROFILES_PATH.read_bytes())
    # Also keep a simple rolling backup next to the source file
    side = PROFILES_PATH.with_suffix(".backup.json")
    side.write_bytes(PROFILES_PATH.read_bytes())
    return dest


def update_profile_prompt_terms(
    inventory_key: str,
    prompt_terms: list[str],
    *,
    profiles_path: Path | None = None,
    default_confidence: float | None = None,
    justification_benchmark_id: str | None = None,
) -> tuple[bool, str]:
    """Atomically update one profile's prompt_terms (and optional confidence).

    Does not modify Roboflow Workflow or API credentials.
    """
    path = profiles_path or PROFILES_PATH
    safe, errors = validate_prompts(prompt_terms)
    if errors and not safe:
        return False, "; ".join(errors)
    if not path.exists():
        return False, "inventory_profiles.json not found."

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Could not read profiles: {exc}"

    profiles = list(raw.get("profiles") or [])
    found = False
    for p in profiles:
        if not isinstance(p, dict):
            continue
        if str(p.get("key") or "") != inventory_key:
            continue
        if p.get("is_custom") or inventory_key == "Custom Item":
            return False, "Custom Item prompts are entered at setup; not stored as a preset."
        old_terms = list(p.get("prompt_terms") or [])
        p["prompt_terms"] = list(safe)
        # Keep allowed classes aligned with prompts when previously mirrored
        allowed = list(p.get("allowed_result_classes") or [])
        if not allowed or set(x.casefold() for x in allowed) == set(
            x.casefold() for x in old_terms
        ):
            p["allowed_result_classes"] = list(safe)
        if default_confidence is not None:
            conf = float(default_confidence)
            if conf < 0.01 or conf > 0.95:
                return False, "Default confidence must be between 0.01 and 0.95."
            p["default_confidence"] = conf
        if justification_benchmark_id:
            p["last_benchmark_id"] = str(justification_benchmark_id)
        found = True
        break
    if not found:
        return False, f"Inventory profile not found: {inventory_key}"

    backup_inventory_profiles(reason=f"promote_{inventory_key}")
    payload = dict(raw)
    payload["profiles"] = profiles
    _atomic_write_json(path, payload)
    clear_profiles_cache()
    conf_note = (
        f"; default_confidence={float(default_confidence):.2f}"
        if default_confidence is not None
        else ""
    )
    return True, (
        f"Updated prompt terms for {inventory_key}: {prompts_to_csv(safe)}{conf_note}"
    )


def build_prompt_comparison_row(
    *,
    prompt_set_label: str,
    prompt_set: list[str],
    ai_count: int,
    expected_count: int | None,
    metrics: BenchmarkMetrics,
    processing_time: float,
    returned_classes: list[str],
    status: str,
) -> dict[str, Any]:
    return {
        "Prompt set": prompt_set_label,
        "Prompts": prompts_to_csv(prompt_set),
        "AI count": ai_count,
        "Expected count": expected_count if expected_count is not None else "—",
        "Count difference": (
            (ai_count - expected_count) if expected_count is not None else "—"
        ),
        "True positives": metrics.true_positives,
        "False positives": metrics.false_positives,
        "Missed objects": metrics.false_negatives,
        "Precision": (
            f"{metrics.precision:.2f}" if metrics.precision is not None else "—"
        ),
        "Recall": f"{metrics.recall:.2f}" if metrics.recall is not None else "—",
        "Processing time": f"{processing_time:.2f}s",
        "Returned classes": ", ".join(returned_classes) or "(none)",
        "Status": status,
    }


def parse_benchmark_metadata(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize optional sample-image benchmark metadata."""
    if not isinstance(entry, dict):
        return None
    bench = entry.get("benchmark")
    if bench is None:
        return None
    if not isinstance(bench, dict):
        return None
    expected = bench.get("expected_count")
    try:
        expected_i = int(expected) if expected is not None else None
    except (TypeError, ValueError):
        expected_i = None
    inv = str(bench.get("inventory_key") or "").strip()
    # Normalize common slug forms to app keys
    slug_map = {
        "traffic_cones": "Traffic Cones",
        "fence_panels": "Fence Panel",
        "fence_panel": "Fence Panel",
        "chairs": "Chairs",
        "boxes": "Boxes",
        "pallets": "Pallets",
        "cars": "Cars",
        "bottles": "Bottles",
        "gates": "Gates",
        "poles": "Poles",
        "custom_item": "Custom Item",
    }
    if inv in slug_map:
        inv = slug_map[inv]
    return {
        "inventory_key": inv or None,
        "expected_count": expected_i,
        "object_definition": str(bench.get("object_definition") or "").strip(),
        "verified": bool(bench.get("verified", False)),
    }


@dataclass
class BenchmarkRunOutcome:
    """One independent prompt-set inference for the benchmark UI."""

    prompt_set_label: str
    prompt_set: list[str] = field(default_factory=list)
    success: bool = False
    execution_failed: bool = False
    raw_count: int = 0
    normalized_count: int = 0
    final_count: int = 0
    returned_classes: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    invocation_mode: str | None = None
    fallback_used: bool = False
    matched_step_id: str | None = None
    matched_step_type: str | None = None
    field_injected: str | None = None
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    annotated_image_bytes: bytes | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    technical: dict[str, Any] = field(default_factory=dict)
    detection_labels: list[str] = field(default_factory=list)
    missed_count: int = 0
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    def apply_review(
        self,
        *,
        expected_count: int | None,
        labels: list[str] | None = None,
        missed_count: int | None = None,
    ) -> None:
        if labels is not None:
            self.detection_labels = list(labels)
        if missed_count is not None:
            self.missed_count = max(0, int(missed_count))
        # Pad/truncate labels to detection count
        n = len(self.detections)
        labs = list(self.detection_labels)
        if len(labs) < n:
            labs.extend(["correct"] * (n - len(labs)))
        self.detection_labels = labs[:n] if n else []
        self.metrics = compute_benchmark_metrics(
            ai_count=self.final_count,
            expected_count=expected_count,
            labels=self.detection_labels,
            missed_count=self.missed_count,
            execution_failed=self.execution_failed,
            request_completed=self.success or (not self.execution_failed),
        )

    def to_storage_dict(
        self,
        *,
        inventory_key: str,
        custom_item_name: str | None,
        expected_count: int | None,
        image_hash: str,
        image_source: str,
        image_name: str,
        model_key: str,
        notes: str = "",
        object_definition: str = "",
    ) -> dict[str, Any]:
        self.apply_review(expected_count=expected_count)
        return {
            "inventory_key": inventory_key,
            "custom_item_name": custom_item_name,
            "prompt_set": list(self.prompt_set),
            "prompt_set_label": self.prompt_set_label,
            "expected_count": expected_count,
            "raw_count": self.raw_count,
            "final_count": self.final_count,
            "true_positives": self.metrics.true_positives,
            "false_positives": self.metrics.false_positives,
            "false_negatives": self.metrics.false_negatives,
            "precision": self.metrics.precision,
            "recall": self.metrics.recall,
            "count_error": self.metrics.count_error,
            "count_accuracy": self.metrics.count_accuracy,
            "processing_time": self.processing_time,
            "returned_classes": list(self.returned_classes),
            "invocation_mode": self.invocation_mode,
            "fallback_used": self.fallback_used,
            "model_key": model_key,
            "notes": notes,
            "object_definition": object_definition,
            "evaluation": self.metrics.evaluation,
            "detection_labels": list(self.detection_labels),
            "image_hash": image_hash,
            "image_source": image_source,
            "image_name": image_name,
            "technical": dict(self.technical),
            "record_kind": "single",
        }


# ---------------------------------------------------------------------------
# Batch benchmark + threshold sweep
# ---------------------------------------------------------------------------

SUPPORTED_BENCHMARK_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SESSIONS_PATH = PROJECT_ROOT / "data" / "benchmark_sessions.json"
DEFAULT_RUN_CACHE_PATH = PROJECT_ROOT / "data" / "benchmark_run_cache.json"

RANKING_OBJECTIVES = (
    "lowest_mae",
    "highest_exact_match",
    "highest_recall",
    "highest_precision",
    "balanced_f1",
)


@dataclass
class NamedPromptSet:
    name: str
    prompts: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompts": list(self.prompts),
            "enabled": bool(self.enabled),
        }


@dataclass
class BatchImageSpec:
    image_id: str
    image_name: str
    image_hash: str
    image_source: str
    size_bytes: int
    expected_count: int | None = None
    object_definition: str = ""
    notes: str = ""
    include_in_aggregate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_thresholds(
    values: list[Any] | tuple[Any, ...] | None,
    *,
    max_thresholds: int = MAX_THRESHOLDS,
) -> tuple[list[float], list[str]]:
    """Normalize confidence thresholds: clamp, dedupe, sort, limit."""
    errors: list[str] = []
    if not values:
        return list(DEFAULT_SWEEP_THRESHOLDS), ["At least one threshold is required."]
    cleaned: list[float] = []
    seen: set[float] = set()
    for raw in values:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Invalid threshold: {raw!r}")
            continue
        if v < 0.01 or v > 0.95:
            errors.append(f"Threshold {v} out of range [0.01, 0.95].")
            continue
        # Round lightly to avoid float noise duplicates
        key = round(v, 4)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(key)
    cleaned.sort()
    if len(cleaned) > max_thresholds:
        errors.append(f"At most {max_thresholds} thresholds are allowed.")
        cleaned = cleaned[:max_thresholds]
    if not cleaned:
        errors.append("No valid thresholds remain.")
    return cleaned, errors


def calculate_inference_run_count(
    *,
    image_count: int,
    prompt_set_count: int,
    threshold_count: int,
) -> int:
    return max(0, int(image_count)) * max(0, int(prompt_set_count)) * max(
        0, int(threshold_count)
    )


def parse_named_prompt_sets(
    raw: list[dict[str, Any]] | None,
    *,
    max_sets: int = MAX_PROMPT_SETS,
) -> tuple[list[NamedPromptSet], list[str]]:
    """Parse named prompt sets with enabled flags."""
    errors: list[str] = []
    if not raw:
        return [], ["At least one prompt set is required."]
    if len(raw) > max_sets:
        errors.append(f"At most {max_sets} prompt sets are allowed.")
        raw = list(raw)[:max_sets]
    out: list[NamedPromptSet] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            errors.append(f"Prompt set {i}: invalid shape.")
            continue
        name = str(item.get("name") or f"Set {chr(64 + i)}").strip() or f"Set {chr(64 + i)}"
        enabled = bool(item.get("enabled", True))
        terms_raw = item.get("prompts")
        if isinstance(terms_raw, str):
            terms = normalize_prompts(terms_raw)
        else:
            terms = normalize_prompts(list(terms_raw or []))
        safe, set_errs = validate_prompts(terms)
        if not enabled:
            out.append(NamedPromptSet(name=name, prompts=safe or terms, enabled=False))
            continue
        if set_errs and not safe:
            errors.append(f"{name}: " + "; ".join(set_errs))
            continue
        for e in set_errs:
            errors.append(f"{name}: {e}")
        if safe:
            out.append(NamedPromptSet(name=name, prompts=safe, enabled=True))
    enabled = [p for p in out if p.enabled and p.prompts]
    if not enabled:
        errors.append("At least one enabled prompt set with valid terms is required.")
    return out, errors


def format_upload_size(num_bytes: int) -> str:
    n = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def is_supported_benchmark_image(filename: str) -> bool:
    return Path(filename or "").suffix.lower() in SUPPORTED_BENCHMARK_SUFFIXES


def dedupe_batch_images(
    images: list[dict[str, Any]],
    *,
    max_images: int = MAX_BATCH_IMAGES,
) -> tuple[list[BatchImageSpec], list[str], dict[str, bytes]]:
    """Deduplicate by content hash; return specs, warnings, and hash→bytes map.

    Input dicts: image_name, image_source, image_bytes, optional expected_count, etc.
    """
    warnings: list[str] = []
    specs: list[BatchImageSpec] = []
    bytes_by_hash: dict[str, bytes] = {}
    seen: set[str] = set()
    for item in images or []:
        if len(specs) >= max_images:
            warnings.append(f"Maximum of {max_images} images per batch; extra images ignored.")
            break
        data = item.get("image_bytes")
        name = str(item.get("image_name") or "image.jpg")
        if not isinstance(data, (bytes, bytearray)) or not data:
            warnings.append(f"Skipped empty image: {name}")
            continue
        if not is_supported_benchmark_image(name):
            warnings.append(f"Unsupported format rejected: {name}")
            continue
        h = image_content_hash(bytes(data))
        if h in seen:
            warnings.append(f"Duplicate image skipped (same content hash): {name}")
            continue
        seen.add(h)
        bytes_by_hash[h] = bytes(data)
        expected = item.get("expected_count")
        try:
            expected_i = int(expected) if expected is not None and expected != "" else None
        except (TypeError, ValueError):
            expected_i = None
        specs.append(
            BatchImageSpec(
                image_id=str(item.get("image_id") or h[:12]),
                image_name=name,
                image_hash=h,
                image_source=str(item.get("image_source") or "upload"),
                size_bytes=len(data),
                expected_count=expected_i,
                object_definition=str(item.get("object_definition") or ""),
                notes=str(item.get("notes") or ""),
                include_in_aggregate=bool(item.get("include_in_aggregate", True)),
            )
        )
    return specs, warnings, bytes_by_hash


def validate_batch_ground_truth(
    images: list[BatchImageSpec],
) -> list[str]:
    """Require expected counts for every image included in aggregate / run."""
    errors: list[str] = []
    included = [im for im in images if im.include_in_aggregate]
    if not included:
        return ["Select at least one image with Include in aggregate = Yes."]
    for im in included:
        n, errs = validate_expected_count(im.expected_count)
        if errs:
            errors.append(f"{im.image_name}: " + "; ".join(errs))
        else:
            im.expected_count = n
    return errors


def build_cache_key(
    *,
    image_hash: str,
    model_key: str,
    prompts: list[str],
    confidence_threshold: float,
    workflow_id: str = "",
    adapter_version: str = ADAPTER_VERSION,
    user_id: int | None = None,
) -> str:
    # Sort for order-insensitive cache hits; inference still uses original order.
    # Include user_id so one account cannot reuse another user's cached runs.
    norm = sorted(normalize_prompts(prompts), key=lambda p: p.casefold())
    payload = {
        "image_hash": image_hash,
        "model_key": model_key,
        "prompts": norm,
        "confidence_threshold": round(float(confidence_threshold), 4),
        "workflow_id": workflow_id or "",
        "adapter_version": adapter_version,
        "user_id": int(user_id) if user_id is not None else None,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_cacheable_success(outcome: "BenchmarkRunOutcome | dict[str, Any]") -> bool:
    """Auth / malformed failures must not be cached as successes."""
    if isinstance(outcome, BenchmarkRunOutcome):
        if outcome.execution_failed or not outcome.success:
            return False
        if outcome.fallback_used:
            return False
        err = (outcome.error_message or "").lower()
        if any(x in err for x in ("api key", "unauthorized", "401", "403", "auth")):
            return False
        return True
    if not isinstance(outcome, dict):
        return False
    if outcome.get("execution_failed") or not outcome.get("success"):
        return False
    if outcome.get("fallback_used"):
        return False
    err = str(outcome.get("error_message") or "").lower()
    if any(x in err for x in ("api key", "unauthorized", "401", "403", "auth")):
        return False
    return True


class BenchmarkRunCache:
    """Simple hash→result cache (no image bytes)."""

    def __init__(self, store: dict[str, dict[str, Any]] | None = None) -> None:
        self._store: dict[str, dict[str, Any]] = dict(store or {})

    def get(self, key: str) -> dict[str, Any] | None:
        hit = self._store.get(key)
        return dict(hit) if isinstance(hit, dict) else None

    def put(self, key: str, result: dict[str, Any]) -> None:
        if not is_cacheable_success(result):
            return
        clean = sanitize_benchmark_record(result)
        # Keep detection boxes for review without annotated bytes
        clean["detections"] = list(result.get("detections") or [])[:200]
        clean["success"] = True
        clean["execution_failed"] = False
        clean["cached"] = True
        self._store[key] = clean

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "entries": dict(self._store)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BenchmarkRunCache":
        if not isinstance(data, dict):
            return cls()
        entries = data.get("entries") if "entries" in data else data
        if not isinstance(entries, dict):
            return cls()
        return cls({str(k): v for k, v in entries.items() if isinstance(v, dict)})

    def save(self, path: Path | None = None) -> Path:
        p = path or DEFAULT_RUN_CACHE_PATH
        _atomic_write_json(p, self.to_dict())
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> "BenchmarkRunCache":
        p = path or DEFAULT_RUN_CACHE_PATH
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return cls()


def outcome_to_run_dict(
    outcome: BenchmarkRunOutcome,
    *,
    image: BatchImageSpec,
    confidence_threshold: float,
    model_key: str,
    session_id: str,
    cached: bool = False,
    reviewed: bool = False,
) -> dict[str, Any]:
    metrics = compute_benchmark_metrics(
        ai_count=outcome.final_count,
        expected_count=image.expected_count,
        labels=outcome.detection_labels if reviewed else None,
        missed_count=outcome.missed_count if reviewed else 0,
        execution_failed=outcome.execution_failed,
        request_completed=outcome.success or (not outcome.execution_failed),
    )
    # Unreviewed: keep count metrics; hide precision/recall
    precision = metrics.precision if reviewed else None
    recall = metrics.recall if reviewed else None
    if not reviewed:
        # Count-only evaluation still useful
        pass
    return {
        "run_id": str(uuid.uuid4()),
        "session_id": session_id,
        "record_kind": "batch_run",
        "image_id": image.image_id,
        "image_name": image.image_name,
        "image_hash": image.image_hash,
        "image_source": image.image_source,
        "include_in_aggregate": image.include_in_aggregate,
        "inventory_notes": image.notes,
        "object_definition": image.object_definition,
        "prompt_set_label": outcome.prompt_set_label,
        "prompt_set": list(outcome.prompt_set),
        "confidence_threshold": float(confidence_threshold),
        "expected_count": image.expected_count,
        "raw_count": outcome.raw_count,
        "normalized_count": outcome.normalized_count,
        "final_count": outcome.final_count,
        "ai_count": outcome.final_count,
        "difference": (
            (outcome.final_count - int(image.expected_count))
            if image.expected_count is not None
            else None
        ),
        "count_error": metrics.count_error,
        "count_accuracy": metrics.count_accuracy,
        "evaluation": metrics.evaluation,
        "exact_match": metrics.evaluation == "exact_count_match",
        "true_positives": metrics.true_positives if reviewed else None,
        "false_positives": metrics.false_positives if reviewed else None,
        "false_negatives": metrics.false_negatives if reviewed else None,
        "precision": precision,
        "recall": recall,
        "precision_status": "ok" if reviewed and precision is not None else (
            "not_reviewed" if not reviewed else "undefined"
        ),
        "recall_status": "ok" if reviewed and recall is not None else (
            "not_reviewed" if not reviewed else "undefined"
        ),
        "reviewed": reviewed,
        "detection_labels": list(outcome.detection_labels),
        "missed_count": outcome.missed_count,
        "returned_classes": list(outcome.returned_classes),
        "processing_time": outcome.processing_time,
        "invocation_mode": outcome.invocation_mode,
        "fallback_used": outcome.fallback_used,
        "success": outcome.success,
        "execution_failed": outcome.execution_failed,
        "error_message": outcome.error_message,
        "model_key": model_key,
        "cached": cached,
        "detections": list(outcome.detections),
        "technical": dict(outcome.technical),
        "matched_step_id": outcome.matched_step_id,
        "field_injected": outcome.field_injected,
    }


def apply_visual_review_to_run(
    run: dict[str, Any],
    *,
    labels: list[str],
    missed_count: int,
) -> dict[str, Any]:
    """Update a run dict with visual review; does not copy labels across thresholds."""
    updated = dict(run)
    dets = list(updated.get("detections") or [])
    labs = list(labels or [])
    if len(labs) < len(dets):
        labs.extend(["correct"] * (len(dets) - len(labs)))
    labs = labs[: len(dets)] if dets else []
    updated["detection_labels"] = labs
    updated["missed_count"] = max(0, int(missed_count))
    updated["reviewed"] = True
    metrics = compute_benchmark_metrics(
        ai_count=int(updated.get("final_count") or 0),
        expected_count=updated.get("expected_count"),
        labels=labs,
        missed_count=updated["missed_count"],
        execution_failed=bool(updated.get("execution_failed")),
        request_completed=bool(updated.get("success"))
        or (not bool(updated.get("execution_failed"))),
    )
    updated["true_positives"] = metrics.true_positives
    updated["false_positives"] = metrics.false_positives
    updated["false_negatives"] = metrics.false_negatives
    updated["precision"] = metrics.precision
    updated["recall"] = metrics.recall
    updated["precision_status"] = "ok" if metrics.precision is not None else "undefined"
    updated["recall_status"] = "ok" if metrics.recall is not None else "undefined"
    updated["count_error"] = metrics.count_error
    updated["count_accuracy"] = metrics.count_accuracy
    updated["evaluation"] = metrics.evaluation
    updated["exact_match"] = metrics.evaluation == "exact_count_match"
    return updated


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def aggregate_prompt_threshold(
    runs: list[dict[str, Any]],
    *,
    prompt_set_label: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Aggregate metrics for one prompt set × threshold (included images only)."""
    subset = [
        r
        for r in runs
        if r.get("prompt_set_label") == prompt_set_label
        and abs(float(r.get("confidence_threshold") or 0) - float(confidence_threshold))
        < 1e-9
        and r.get("include_in_aggregate", True)
    ]
    failed = [r for r in subset if r.get("execution_failed")]
    ok = [r for r in subset if not r.get("execution_failed")]
    reviewed = [r for r in ok if r.get("reviewed")]
    abs_errors = [
        float(r["count_error"])
        for r in ok
        if r.get("count_error") is not None
    ]
    exact = sum(1 for r in ok if r.get("exact_match"))
    over = sum(
        1
        for r in ok
        if r.get("expected_count") is not None
        and int(r.get("final_count") or 0) > int(r["expected_count"])
    )
    under = sum(
        1
        for r in ok
        if r.get("expected_count") is not None
        and int(r.get("final_count") or 0) < int(r["expected_count"])
    )
    zero_det = sum(1 for r in ok if int(r.get("final_count") or 0) == 0)
    total_expected = sum(int(r.get("expected_count") or 0) for r in ok)
    total_ai = sum(int(r.get("final_count") or 0) for r in ok)

    sum_tp = sum(int(r.get("true_positives") or 0) for r in reviewed)
    sum_fp = sum(int(r.get("false_positives") or 0) for r in reviewed)
    sum_fn = sum(int(r.get("false_negatives") or 0) for r in reviewed)
    micro_p = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else None
    micro_r = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else None
    macro_p_vals = [
        float(r["precision"]) for r in reviewed if r.get("precision") is not None
    ]
    macro_r_vals = [
        float(r["recall"]) for r in reviewed if r.get("recall") is not None
    ]
    times = [float(r.get("processing_time") or 0) for r in subset]

    return {
        "prompt_set_label": prompt_set_label,
        "confidence_threshold": float(confidence_threshold),
        "images_evaluated": len(ok),
        "total_runs": len(subset),
        "total_expected_objects": total_expected,
        "total_ai_detections": total_ai,
        "exact_match_images": exact,
        "exact_match_rate": (exact / len(ok)) if ok else None,
        "mean_absolute_count_error": _mean(abs_errors),
        "median_absolute_count_error": _median(abs_errors),
        "total_overcount": over,
        "total_undercount": under,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_precision": _mean(macro_p_vals),
        "macro_recall": _mean(macro_r_vals),
        "precision_recall_available": bool(reviewed),
        "reviewed_images": len(reviewed),
        "mean_processing_time": _mean(times),
        "failed_runs": len(failed),
        "zero_detection_runs": zero_det,
        "incomplete_metrics": not bool(reviewed),
    }


def build_comparison_matrix(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compact matrix rows for UI/table rendering."""
    rows = []
    for a in aggregates:
        rows.append(
            {
                "Prompt set": a.get("prompt_set_label"),
                "Threshold": a.get("confidence_threshold"),
                "MAE": a.get("mean_absolute_count_error"),
                "Exact-match rate": a.get("exact_match_rate"),
                "Failed runs": a.get("failed_runs"),
                "Images": a.get("images_evaluated"),
                "Micro P": a.get("micro_precision"),
                "Micro R": a.get("micro_recall"),
                "Incomplete PR": a.get("incomplete_metrics"),
            }
        )
    return rows


def recommend_configuration(
    aggregates: list[dict[str, Any]],
    *,
    objective: str = "lowest_mae",
    min_images: int = MIN_RECOMMENDATION_RUNS,
) -> dict[str, Any] | None:
    """Best configuration for this benchmark dataset (not universally best)."""
    candidates = [
        a
        for a in aggregates
        if int(a.get("images_evaluated") or 0) >= min_images
        and int(a.get("failed_runs") or 0) < int(a.get("total_runs") or 1)
    ]
    if not candidates:
        return None

    obj = (objective or "lowest_mae").lower()

    def score(a: dict[str, Any]) -> tuple:
        mae = a.get("mean_absolute_count_error")
        emr = a.get("exact_match_rate")
        prec = a.get("micro_precision")
        rec = a.get("micro_recall")
        failed = int(a.get("failed_runs") or 0)
        # Lower is better for sort key first component when negated appropriately
        if obj == "highest_exact_match":
            return (-(emr if emr is not None else -1), failed, mae if mae is not None else 1e9)
        if obj == "highest_recall":
            if rec is None:
                return (1e9, failed)
            return (-rec, failed, mae if mae is not None else 1e9)
        if obj == "highest_precision":
            if prec is None:
                return (1e9, failed)
            return (-prec, failed, mae if mae is not None else 1e9)
        if obj == "balanced_f1":
            if prec is None or rec is None or (prec + rec) == 0:
                return (1e9, failed)
            f1 = 2 * prec * rec / (prec + rec)
            return (-f1, failed, mae if mae is not None else 1e9)
        # lowest_mae default
        return (mae if mae is not None else 1e9, failed, -(emr if emr is not None else 0))

    best = sorted(candidates, key=score)[0]
    return {
        "label": "Best configuration for this benchmark dataset",
        "objective": obj,
        "prompt_set_label": best.get("prompt_set_label"),
        "confidence_threshold": best.get("confidence_threshold"),
        "supporting_image_count": best.get("images_evaluated"),
        "exact_match_rate": best.get("exact_match_rate"),
        "mean_absolute_count_error": best.get("mean_absolute_count_error"),
        "precision_recall_available": best.get("precision_recall_available"),
        "micro_precision": best.get("micro_precision"),
        "micro_recall": best.get("micro_recall"),
        "failed_runs": best.get("failed_runs"),
        "aggregate": best,
    }


def build_batch_session(
    *,
    inventory_key: str,
    model_key: str,
    images: list[BatchImageSpec],
    prompt_sets: list[NamedPromptSet],
    thresholds: list[float],
    runs: list[dict[str, Any]],
    custom_item_name: str | None = None,
    recommendation: dict[str, Any] | None = None,
    profile_update: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    enabled_sets = [p for p in prompt_sets if p.enabled]
    aggregates = []
    for ps in enabled_sets:
        for thr in thresholds:
            aggregates.append(
                aggregate_prompt_threshold(
                    runs,
                    prompt_set_label=ps.name,
                    confidence_threshold=thr,
                )
            )
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_kind": "batch_session",
        "inventory_key": inventory_key,
        "custom_item_name": custom_item_name,
        "model_key": model_key,
        "images": [im.to_dict() for im in images],
        "prompt_sets": [p.to_dict() for p in prompt_sets],
        "confidence_thresholds": list(thresholds),
        "runs": list(runs),
        "aggregates": aggregates,
        "recommendation": recommendation,
        "profile_update": profile_update,
        "total_planned_runs": calculate_inference_run_count(
            image_count=len([i for i in images if i.include_in_aggregate]),
            prompt_set_count=len(enabled_sets),
            threshold_count=len(thresholds),
        ),
    }


def save_batch_session(
    session: dict[str, Any],
    *,
    path: Path | None = None,
    history_path: Path | None = None,
    write_history: bool = True,
) -> dict[str, Any]:
    p = path or DEFAULT_SESSIONS_PATH
    existing: list[dict[str, Any]] = []
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = list(raw.get("sessions") or [])
            elif isinstance(raw, list):
                existing = raw
        except (OSError, json.JSONDecodeError):
            existing = []
    # Strip annotated / private bytes if any slipped in; keep detection boxes.
    clean_session = json.loads(json.dumps(session, default=str))
    for run in clean_session.get("runs") or []:
        if isinstance(run, dict):
            run.pop("annotated_image_bytes", None)
            run.pop("image_bytes", None)
            for banned in ("api_key", "ROBOFLOW_API_KEY", "authorization", "token"):
                run.pop(banned, None)
    existing.append(clean_session)
    _atomic_write_json(p, {"version": 1, "sessions": existing})
    # Compact history rows (compatible with old single-image records)
    if write_history:
        try:
            for run in clean_session.get("runs") or []:
                if not isinstance(run, dict):
                    continue
                row = sanitize_benchmark_record(
                    {
                        **run,
                        "benchmark_id": run.get("run_id"),
                        # Runs inherit the session's owner so batch rows stay
                        # attributed even though the run dict has no owner.
                        "user_id": clean_session.get("user_id"),
                        "username": clean_session.get("username"),
                        "notes": (
                            str(run.get("notes") or "")
                            + f" batch_session:{clean_session.get('session_id')}"
                        ).strip(),
                    }
                )
                save_benchmark_result(row, path=history_path)
        except Exception:
            pass
    return clean_session


def load_batch_sessions(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_SESSIONS_PATH
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict):
        return [s for s in (raw.get("sessions") or []) if isinstance(s, dict)]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    return []


def export_session_json(session: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(session, default=str))
    for run in payload.get("runs") or []:
        if isinstance(run, dict):
            run.pop("annotated_image_bytes", None)
            run.pop("image_bytes", None)
            run.pop("api_key", None)
    return json.dumps(payload, indent=2)


def export_session_csv(session: dict[str, Any]) -> str:
    """One row per image × prompt set × threshold."""
    import csv
    import io

    fields = [
        "session_id",
        "image_name",
        "image_hash",
        "expected_count",
        "prompt_set",
        "prompts",
        "confidence_threshold",
        "AI_count",
        "difference",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "count_error",
        "processing_time",
        "status",
        "returned_classes",
        "invocation_mode",
        "fallback_used",
        "reviewed",
        "cached",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    sid = session.get("session_id")
    for run in session.get("runs") or []:
        if not isinstance(run, dict):
            continue
        writer.writerow(
            {
                "session_id": sid,
                "image_name": run.get("image_name"),
                "image_hash": run.get("image_hash"),
                "expected_count": run.get("expected_count"),
                "prompt_set": run.get("prompt_set_label"),
                "prompts": prompts_to_csv(list(run.get("prompt_set") or [])),
                "confidence_threshold": run.get("confidence_threshold"),
                "AI_count": run.get("final_count"),
                "difference": run.get("difference"),
                "TP": run.get("true_positives")
                if run.get("reviewed")
                else "not_reviewed",
                "FP": run.get("false_positives")
                if run.get("reviewed")
                else "not_reviewed",
                "FN": run.get("false_negatives")
                if run.get("reviewed")
                else "not_reviewed",
                "precision": (
                    run.get("precision")
                    if run.get("reviewed")
                    else "not_reviewed"
                ),
                "recall": (
                    run.get("recall") if run.get("reviewed") else "not_reviewed"
                ),
                "count_error": run.get("count_error"),
                "processing_time": run.get("processing_time"),
                "status": run.get("evaluation"),
                "returned_classes": ", ".join(run.get("returned_classes") or []),
                "invocation_mode": run.get("invocation_mode"),
                "fallback_used": run.get("fallback_used"),
                "reviewed": run.get("reviewed"),
                "cached": run.get("cached"),
            }
        )
    return buf.getvalue()


def enumerate_batch_combinations(
    images: list[BatchImageSpec],
    prompt_sets: list[NamedPromptSet],
    thresholds: list[float],
) -> list[tuple[BatchImageSpec, NamedPromptSet, float]]:
    combos = []
    for im in images:
        if not im.include_in_aggregate:
            continue
        for ps in prompt_sets:
            if not ps.enabled:
                continue
            for thr in thresholds:
                combos.append((im, ps, float(thr)))
    return combos
