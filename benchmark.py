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
) -> tuple[bool, str]:
    """Atomically update one profile's prompt_terms after backing up.

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
        found = True
        break
    if not found:
        return False, f"Inventory profile not found: {inventory_key}"

    backup_inventory_profiles(reason=f"promote_{inventory_key}")
    payload = dict(raw)
    payload["profiles"] = profiles
    _atomic_write_json(path, payload)
    clear_profiles_cache()
    return True, f"Updated prompt terms for {inventory_key}: {prompts_to_csv(safe)}"


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
        }
