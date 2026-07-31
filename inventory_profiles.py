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


def inventory_display_name(key: str | None, *, custom_item_name: str | None = None) -> str:
    if not key:
        return ""
    if is_custom_inventory(key):
        name = (custom_item_name or "").strip()
        if name:
            # Title-case lightly for display without forcing acronyms
            return name[:1].upper() + name[1:] if len(name) > 1 else name.upper()
        return "Custom Item"
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
        name = (custom_item_name or "").strip()
        if name:
            return f"individual {name.lower()}"
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


def parse_custom_prompts(
    item_name: str | None,
    alternatives: str | None = None,
) -> tuple[list[str], list[str]]:
    """Build effective prompts from custom item name + comma-separated alternatives."""
    name = " ".join((item_name or "").strip().split())
    if not name:
        return [], ["Item name is required for Custom Item."]
    alts = alternatives or ""
    parts: list[str] = [name]
    parts.extend(normalize_prompts(alts))
    return validate_prompts(parts)


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
            )
        except (TypeError, ValueError):
            return None


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
    return {
        "default_model": p.get("default_model") or "YOLO-World",
        "recommended_models": [p.get("default_model") or "YOLO-World"],
        "alternative_models": (
            ["Local Picket Counter"] if inventory_key == "Fence Panel" else []
        ),
        "prompt": prompts_to_csv(prompts),
        "allowed_classes": list(p.get("allowed_result_classes") or prompts),
        "confidence_threshold": float(p.get("default_confidence") or 0.25),
        "counting_strategy": "Object Detection",
        "counting_note": p.get("description") or "",
    }


def all_recommendations() -> dict[str, dict[str, Any]]:
    return {p["key"]: recommendation_dict_for(p["key"]) for p in load_inventory_profiles() if not p.get("is_custom")}
