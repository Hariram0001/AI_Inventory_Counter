"""Inventory prompt profiles, prompt normalization, and analysis run context.

Profiles load from inventory_profiles.json (project-relative). No Streamlit.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
PROFILES_PATH = PROJECT_ROOT / "inventory_profiles.json"

# Prompt validation limits (sensible POC defaults)
MAX_PROMPTS = 8
MAX_PROMPT_LEN = 64
MAX_TOTAL_PROMPT_LEN = 400

# Reject obvious HTML / code fragments in custom prompts
_UNSAFE_PROMPT_RE = re.compile(
    r"[<>`]|javascript:|on\w+\s*=|<\s*script|{\s*}|\$\{",
    re.IGNORECASE,
)

_CUSTOM_KEY = "Custom Item"
_profiles_cache: list[dict[str, Any]] | None = None


def clear_profiles_cache() -> None:
    global _profiles_cache
    _profiles_cache = None


def load_inventory_profiles(*, force_reload: bool = False) -> list[dict[str, Any]]:
    """Load inventory profiles from JSON (cached)."""
    global _profiles_cache
    if _profiles_cache is not None and not force_reload:
        return list(_profiles_cache)
    if not PROFILES_PATH.exists():
        _profiles_cache = []
        return []
    try:
        raw = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _profiles_cache = []
        return []
    profiles = list(raw.get("profiles") or [])
    # Normalize keys
    cleaned: list[dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict) or not str(p.get("key") or "").strip():
            continue
        cleaned.append(
            {
                "key": str(p["key"]).strip(),
                "display_name": str(p.get("display_name") or p["key"]).strip(),
                "description": str(p.get("description") or "").strip(),
                "enabled": bool(p.get("enabled", True)),
                "prompt_terms": [str(x).strip() for x in (p.get("prompt_terms") or []) if str(x).strip()],
                "allowed_result_classes": [
                    str(x).strip()
                    for x in (p.get("allowed_result_classes") or [])
                    if str(x).strip()
                ],
                "default_confidence": float(p.get("default_confidence") or 0.25),
                "counting_unit": str(p.get("counting_unit") or "individual item").strip(),
                "default_model": str(p.get("default_model") or "YOLO-World").strip(),
                "icon": str(p.get("icon") or "").strip(),
                "is_custom": bool(p.get("is_custom", False))
                or str(p.get("key")).strip() == _CUSTOM_KEY,
            }
        )
    _profiles_cache = cleaned
    return list(cleaned)


def get_profile(key: str | None) -> dict[str, Any] | None:
    if not key:
        return None
    for p in load_inventory_profiles():
        if p["key"] == key:
            return dict(p)
    return None


def enabled_profiles() -> list[dict[str, Any]]:
    return [p for p in load_inventory_profiles() if p.get("enabled")]


def selectable_inventory_keys() -> list[str]:
    return [p["key"] for p in enabled_profiles()]


def inventory_type_keys() -> list[str]:
    """All profile keys (enabled or not) for UI grids."""
    return [p["key"] for p in load_inventory_profiles()]


def is_custom_inventory(key: str | None) -> bool:
    if not key:
        return False
    p = get_profile(key)
    return bool(p and p.get("is_custom")) or key == _CUSTOM_KEY


def _title_prompt(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text[:1].upper() + text[1:] if len(text) > 1 else text.upper()


def inventory_display_name(key: str | None, *, custom_item_name: str | None = None) -> str:
    if not key:
        return ""
    if is_custom_inventory(key):
        items = normalize_prompts(custom_item_name)
        if not items:
            return "Custom Item"
        if len(items) == 1:
            return _title_prompt(items[0])
        if len(items) <= 3:
            return ", ".join(_title_prompt(p) for p in items)
        return f"{len(items)} custom items"
    p = get_profile(key)
    if p:
        return str(p.get("display_name") or key)
    # Legacy / unknown keys (history compatibility)
    if key == "Fence Panel":
        return "Fence Panels"
    return key


def counting_unit_for(
    key: str | None,
    *,
    custom_item_name: str | None = None,
) -> str:
    if is_custom_inventory(key):
        items = normalize_prompts(custom_item_name)
        if len(items) == 1:
            return f"individual {items[0].lower()}"
        if len(items) > 1:
            return "individual items by type (counted separately)"
        return "individual item"
    p = get_profile(key)
    if p and p.get("counting_unit"):
        return str(p["counting_unit"])
    if key == "Fence Panel":
        return "individual fence panel"
    return "individual item"


def normalize_prompts(
    terms: list[str] | tuple[str, ...] | str | None,
) -> list[str]:
    """Trim, drop empties, case-insensitive de-dupe; preserve first-seen casing."""
    if terms is None:
        return []
    if isinstance(terms, str):
        raw_parts = re.split(r"[,;\n]+", terms)
    else:
        raw_parts = []
        for t in terms:
            raw_parts.extend(re.split(r"[,;\n]+", str(t)))
    seen: set[str] = set()
    out: list[str] = []
    for part in raw_parts:
        cleaned = " ".join(str(part).strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def prompt_is_unsafe(text: str) -> bool:
    return bool(_UNSAFE_PROMPT_RE.search(text or ""))


def validate_prompts(prompts: list[str]) -> tuple[list[str], list[str]]:
    """Return (normalized_prompts, errors). Empty prompt list is an error."""
    errors: list[str] = []
    normalized = normalize_prompts(prompts)
    if not normalized:
        errors.append("At least one detection prompt is required.")
        return [], errors

    safe: list[str] = []
    for p in normalized:
        if prompt_is_unsafe(p):
            errors.append(f"Prompt rejected (unsupported characters): {p[:40]}")
            continue
        if len(p) > MAX_PROMPT_LEN:
            errors.append(
                f"Prompt exceeds {MAX_PROMPT_LEN} characters: {p[:40]}…"
            )
            continue
        safe.append(p)

    if len(safe) > MAX_PROMPTS:
        errors.append(f"At most {MAX_PROMPTS} detection prompts are allowed.")
        safe = safe[:MAX_PROMPTS]

    total = sum(len(p) for p in safe) + max(0, len(safe) - 1) * 2  # commas+space
    if total > MAX_TOTAL_PROMPT_LEN:
        errors.append(
            f"Combined prompt length exceeds {MAX_TOTAL_PROMPT_LEN} characters."
        )

    if not safe:
        errors.append("No valid detection prompts remain after validation.")
    return safe, errors


@dataclass
class CustomItemSpec:
    """One custom inventory type to detect separately from the others."""

    name: str
    aliases: list[str] = field(default_factory=list)

    @property
    def class_names(self) -> list[str]:
        """Primary name + aliases for model class lists."""
        out = [self.name]
        seen = {self.name.casefold()}
        for alias in self.aliases:
            key = alias.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(alias)
        return out


def parse_custom_item_specs(
    item_name: str | None,
    alternatives: str | None = None,
) -> tuple[list[CustomItemSpec], list[str]]:
    """Parse custom items into separate item types (+ optional aliases).

    - ``item_name``: each line/comma entry is a **separate** item type.
    - ``alternatives``:
        - ``traffic cone: road cone, safety cone`` → aliases for that type
        - with a single item type, free aliases attach to that type
        - with multiple item types, free aliases become additional separate types
          (prefer the ``primary: alias`` form for synonyms)
    """
    notes: list[str] = []
    primaries = normalize_prompts(item_name or "")
    alias_map: dict[str, list[str]] = {p.casefold(): [] for p in primaries}
    primary_by_fold = {p.casefold(): p for p in primaries}
    free_aliases: list[str] = []

    for raw_line in re.split(r"[\n;]+", str(alternatives or "")):
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            heads = normalize_prompts(left)
            aliases = normalize_prompts(right)
            if not heads:
                free_aliases.extend(aliases)
                continue
            head = heads[0]
            fold = head.casefold()
            if fold not in primary_by_fold:
                primaries.append(head)
                primary_by_fold[fold] = head
                alias_map[fold] = []
            for alias in aliases:
                if alias.casefold() == fold:
                    continue
                if alias.casefold() not in {a.casefold() for a in alias_map[fold]}:
                    alias_map[fold].append(alias)
        else:
            free_aliases.extend(normalize_prompts(line))

    if not primaries:
        # Synonyms-only path: each term is its own separate item type.
        primaries = normalize_prompts(free_aliases)
        free_aliases = []
        alias_map = {p.casefold(): [] for p in primaries}
        primary_by_fold = {p.casefold(): p for p in primaries}
    elif len(primaries) == 1:
        fold = primaries[0].casefold()
        for alias in free_aliases:
            if alias.casefold() == fold:
                continue
            if alias.casefold() not in {a.casefold() for a in alias_map[fold]}:
                alias_map[fold].append(alias)
        free_aliases = []
    elif free_aliases:
        notes.append(
            "Unscoped extra terms were added as separate item types. "
            "Use 'item: alias1, alias2' to attach synonyms to one item."
        )
        for alias in free_aliases:
            fold = alias.casefold()
            if fold in primary_by_fold:
                continue
            primaries.append(alias)
            primary_by_fold[fold] = alias
            alias_map[fold] = []

    if not primaries:
        return [], ["Enter at least one custom item to detect."]

    # Validate flat class list, then rebuild specs from surviving terms.
    flat: list[str] = []
    for p in primaries:
        flat.append(p)
        flat.extend(alias_map.get(p.casefold(), []))
    safe, errors = validate_prompts(flat)
    if errors and not safe:
        return [], errors

    safe_fold = {s.casefold(): s for s in safe}
    specs: list[CustomItemSpec] = []
    for primary in primaries:
        kept_name = safe_fold.get(primary.casefold())
        if not kept_name:
            continue
        kept_aliases = [
            safe_fold[a.casefold()]
            for a in alias_map.get(primary.casefold(), [])
            if a.casefold() in safe_fold and a.casefold() != primary.casefold()
        ]
        specs.append(CustomItemSpec(name=kept_name, aliases=kept_aliases))

    if not specs:
        return [], errors or ["Enter at least one custom item to detect."]
    return specs, errors + notes


def parse_custom_prompts(
    item_name: str | None,
    alternatives: str | None = None,
) -> tuple[list[str], list[str]]:
    """Build flat model class list from custom items (+ optional aliases)."""
    specs, errors = parse_custom_item_specs(item_name, alternatives)
    if not specs:
        return [], errors
    flat: list[str] = []
    for spec in specs:
        flat.extend(spec.class_names)
    # Re-validate length after expansion (already validated inside specs parser).
    return flat, errors


def custom_class_alias_map(
    specs: list[CustomItemSpec],
) -> dict[str, str]:
    """Map casefolded class/alias → primary item type name."""
    mapping: dict[str, str] = {}
    for spec in specs:
        mapping[spec.name.casefold()] = spec.name
        for alias in spec.aliases:
            mapping[alias.casefold()] = spec.name
    return mapping


def canonicalize_detection_class(
    class_name: str | None,
    alias_map: dict[str, str] | None,
) -> str:
    raw = str(class_name or "").strip()
    if not raw:
        return "object"
    if not alias_map:
        return raw
    return alias_map.get(raw.casefold()) or alias_map.get(
        raw.replace("_", " ").casefold()
    ) or raw.replace("_", " ")


def counts_by_item_type(
    detections: list[Any],
    *,
    primary_types: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """Count included detections per primary item type (separate totals)."""
    totals: dict[str, int] = {}
    if primary_types:
        for name in primary_types:
            totals[name] = 0
    for det in detections or []:
        if not bool(getattr(det, "included_in_count", True)):
            continue
        if bool(getattr(det, "excluded_by_region", False)):
            continue
        label = canonicalize_detection_class(
            getattr(det, "class_name", None), alias_map
        )
        try:
            n = int(getattr(det, "item_count", 1) or 1)
        except (TypeError, ValueError):
            n = 1
        if bool(getattr(det, "count_only", False)):
            n = max(0, n)
        else:
            n = 1
        totals[label] = totals.get(label, 0) + max(0, n)
    return totals


def effective_prompts_for_inventory(
    inventory_key: str | None,
    *,
    custom_item_name: str | None = None,
    custom_alternatives: str | None = None,
    prompt_override: str | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve validated effective prompts for a selected inventory."""
    if prompt_override and prompt_override.strip():
        return validate_prompts(normalize_prompts(prompt_override))

    if is_custom_inventory(inventory_key):
        return parse_custom_prompts(custom_item_name, custom_alternatives)

    p = get_profile(inventory_key)
    if p:
        return validate_prompts(list(p.get("prompt_terms") or []))

    # Legacy fallbacks
    if inventory_key == "Fence Panel":
        return validate_prompts(
            ["fence panel", "wooden fence panel", "privacy fence panel"]
        )
    if inventory_key:
        return validate_prompts([inventory_key])
    return [], ["Select an inventory type."]


def prompts_to_csv(prompts: list[str]) -> str:
    return ", ".join(prompts)


@dataclass
class AnalysisRunContext:
    """Canonical analysis run context (not scattered session prompt fields)."""

    inventory_key: str
    inventory_display_name: str
    counting_unit: str
    effective_prompts: list[str] = field(default_factory=list)
    selected_model_key: str = ""
    selected_model_display_name: str = "YOLO-World"
    confidence_threshold: float = 0.25
    uploaded_image_ids: list[str] = field(default_factory=list)
    custom_item_name: str | None = None
    allowed_result_classes: list[str] = field(default_factory=list)
    # Custom Item: each entry is a separate detection type.
    primary_item_types: list[str] = field(default_factory=list)
    class_alias_map: dict[str, str] = field(default_factory=dict)

    def prompt_csv(self) -> str:
        return prompts_to_csv(self.effective_prompts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalysisRunContext | None:
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                inventory_key=str(data.get("inventory_key") or ""),
                inventory_display_name=str(data.get("inventory_display_name") or ""),
                counting_unit=str(data.get("counting_unit") or "individual item"),
                effective_prompts=list(data.get("effective_prompts") or []),
                selected_model_key=str(data.get("selected_model_key") or ""),
                selected_model_display_name=str(
                    data.get("selected_model_display_name") or "YOLO-World"
                ),
                confidence_threshold=float(data.get("confidence_threshold") or 0.25),
                uploaded_image_ids=list(data.get("uploaded_image_ids") or []),
                custom_item_name=(
                    str(data["custom_item_name"])
                    if data.get("custom_item_name") not in (None, "")
                    else None
                ),
                allowed_result_classes=list(data.get("allowed_result_classes") or []),
                primary_item_types=list(data.get("primary_item_types") or []),
                class_alias_map={
                    str(k).casefold(): str(v)
                    for k, v in dict(data.get("class_alias_map") or {}).items()
                },
            )
        except (TypeError, ValueError):
            return None


def class_names_for_primary_type(
    run_ctx: AnalysisRunContext | None,
    primary: str,
) -> list[str]:
    """Class names for one primary type (primary + its aliases only)."""
    name = str(primary or "").strip()
    if not name:
        return []
    if run_ctx is None:
        return [name]
    fold = name.casefold()
    alias_map = dict(run_ctx.class_alias_map or {})
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in list(run_ctx.primary_item_types or []) + list(
        run_ctx.effective_prompts or []
    ):
        term = str(candidate or "").strip()
        if not term:
            continue
        target = alias_map.get(term.casefold(), term)
        if target.casefold() != fold and term.casefold() != fold:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(term)
    if not ordered:
        ordered = [name]
    for i, term in enumerate(ordered):
        if term.casefold() == fold:
            if i != 0:
                ordered.insert(0, ordered.pop(i))
            break
    else:
        ordered.insert(0, name)
    return ordered


def preset_primary_and_aliases(
    inventory_key: str,
    prompts: list[str],
    *,
    profile: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str]]:
    """One primary inventory type; synonym prompt terms map onto it internally."""
    prof = profile if isinstance(profile, dict) else (get_profile(inventory_key) or {})
    primary = str(prof.get("display_name") or inventory_key or "").strip()
    if not primary:
        for p in prompts or []:
            term = str(p or "").strip()
            if term:
                primary = term
                break
    if not primary:
        primary = "item"
    alias_map: dict[str, str] = {primary.casefold(): primary}
    for p in prompts or []:
        term = str(p or "").strip()
        if term:
            alias_map[term.casefold()] = primary
    # Also map the raw inventory key when it differs from display_name.
    key = str(inventory_key or "").strip()
    if key:
        alias_map[key.casefold()] = primary
    return primary, alias_map


def build_run_context(
    *,
    inventory_key: str,
    custom_item_name: str | None = None,
    custom_alternatives: str | None = None,
    selected_model_key: str = "",
    selected_model_display_name: str = "YOLO-World",
    confidence_threshold: float | None = None,
    uploaded_image_ids: list[str] | None = None,
    prompt_override: str | None = None,
) -> tuple[AnalysisRunContext | None, list[str]]:
    prompts, errors = effective_prompts_for_inventory(
        inventory_key,
        custom_item_name=custom_item_name,
        custom_alternatives=custom_alternatives,
        prompt_override=prompt_override,
    )
    if errors and not prompts:
        return None, errors

    profile = get_profile(inventory_key) or {}
    conf = confidence_threshold
    if conf is None:
        conf = float(profile.get("default_confidence") or 0.25)

    primary_types: list[str] = []
    alias_map: dict[str, str] = {}
    if is_custom_inventory(inventory_key) and not (
        prompt_override and prompt_override.strip()
    ):
        specs, spec_notes = parse_custom_item_specs(
            custom_item_name, custom_alternatives
        )
        for note in spec_notes:
            if note and note not in errors:
                errors.append(note)
        primary_types = [s.name for s in specs]
        alias_map = custom_class_alias_map(specs)
        # Prefer flat class list from specs (primaries + aliases).
        prompts = []
        for spec in specs:
            prompts.extend(spec.class_names)
    elif prompts:
        # Presets: one primary type; synonyms stay internal for the model.
        primary, alias_map = preset_primary_and_aliases(
            inventory_key, prompts, profile=profile
        )
        primary_types = [primary]

    ctx = AnalysisRunContext(
        inventory_key=inventory_key,
        inventory_display_name=inventory_display_name(
            inventory_key, custom_item_name=custom_item_name
        ),
        counting_unit=counting_unit_for(
            inventory_key, custom_item_name=custom_item_name
        ),
        effective_prompts=prompts,
        selected_model_key=selected_model_key,
        selected_model_display_name=selected_model_display_name or "YOLO-World",
        confidence_threshold=float(conf),
        uploaded_image_ids=list(uploaded_image_ids or []),
        custom_item_name=(custom_item_name or None) if is_custom_inventory(inventory_key) else None,
        allowed_result_classes=list(profile.get("allowed_result_classes") or prompts),
        primary_item_types=primary_types,
        class_alias_map=alias_map,
    )
    return ctx, errors


def recommendation_dict_for(inventory_key: str | None) -> dict[str, Any]:
    """Shape compatible with inventory_config.resolve_recommended_model lookups."""
    p = get_profile(inventory_key)
    if not p:
        if inventory_key == "Fence Panel":
            return {
                "default_model": "YOLO-World",
                "recommended_models": ["YOLO-World", "Local Picket Counter"],
                "alternative_models": ["Local Picket Counter"],
                "prompt": "fence panel, wooden fence panel, privacy fence panel",
                "allowed_classes": [
                    "fence panel",
                    "wooden fence panel",
                    "privacy fence panel",
                    "wood fence",
                ],
                "confidence_threshold": 0.25,
                "counting_strategy": "Object Detection",
            }
        return {}
    prompts = list(p.get("prompt_terms") or [])
    default_model = str(p.get("default_model") or "OpenRouter VLM Detector").strip()
    recommended = [default_model]
    for name in ("OpenRouter VLM Detector", "YOLO-World"):
        if name not in recommended:
            recommended.append(name)
    alternatives = [n for n in recommended[1:]]
    if inventory_key == "Fence Panel" and "Local Picket Counter" not in alternatives:
        alternatives.append("Local Picket Counter")
    return {
        "default_model": default_model,
        "recommended_models": recommended,
        "alternative_models": alternatives,
        "prompt": prompts_to_csv(prompts),
        "allowed_classes": list(p.get("allowed_result_classes") or prompts),
        "confidence_threshold": float(p.get("default_confidence") or 0.25),
        "counting_strategy": "Object Detection",
        "counting_note": p.get("description") or "",
    }


def all_recommendations() -> dict[str, dict[str, Any]]:
    return {p["key"]: recommendation_dict_for(p["key"]) for p in load_inventory_profiles() if not p.get("is_custom")}
