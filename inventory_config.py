"""Inventory selection rules and recommended AI setup resolution (no Streamlit)."""

from __future__ import annotations

from typing import Any

from inventory_profiles import (
    build_run_context,
    counting_unit_for,
    effective_prompts_for_inventory,
    enabled_profiles,
    get_profile,
    inventory_display_name as profile_display_name,
    is_custom_inventory,
    recommendation_dict_for,
    selectable_inventory_keys,
)
from model_registry import (
    get_default_model,
    get_enabled_valid_models,
    get_selectable_analysis_models,
)
from schemas import ModelConfig

# Re-export for callers that historically imported from this module.
__all__ = [
    "SELECTABLE_INVENTORY_KEY",
    "FIXED_PHOTO_RELATIONSHIP",
    "PHOTO_RELATIONSHIP_NOTE",
    "PHOTO_RELATIONSHIP_NOTE_PHOTOS",
    "INVENTORY_DISPLAY_NAMES",
    "inventory_display_name",
    "is_inventory_selectable",
    "non_demo_enabled_models",
    "get_selectable_analysis_models",
    "resolve_recommended_model",
    "form_updates_from_recommendation",
    "suggest_model_from_trial_rows",
    "is_custom_inventory",
    "counting_unit_for",
    "effective_prompts_for_inventory",
    "build_run_context",
]

# Legacy constant: Fence Panel remains the primary historical key.
SELECTABLE_INVENTORY_KEY = "Fence Panel"
FIXED_PHOTO_RELATIONSHIP = "Separate inventory areas"
PHOTO_RELATIONSHIP_NOTE = "Each photo will be analyzed as a separate inventory set."
PHOTO_RELATIONSHIP_NOTE_PHOTOS = "Each photo is analyzed as a separate inventory set."

INVENTORY_DISPLAY_NAMES: dict[str, str] = {
    p["key"]: p["display_name"] for p in enabled_profiles()
}
# Ensure legacy display name even if JSON temporarily missing
INVENTORY_DISPLAY_NAMES.setdefault("Fence Panel", "Fence Panels")


def inventory_display_name(key: str | None, *, custom_item_name: str | None = None) -> str:
    return profile_display_name(key, custom_item_name=custom_item_name)


def is_inventory_selectable(key: str | None) -> bool:
    if not key:
        return False
    return key in selectable_inventory_keys()


def non_demo_enabled_models(
    models: list[ModelConfig],
    *,
    allow_demo: bool = False,
) -> list[ModelConfig]:
    enabled = get_enabled_valid_models(models, allow_demo_ids=allow_demo)
    if allow_demo:
        return enabled
    return [m for m in enabled if not m.is_demo_model_id()]


def resolve_recommended_model(
    inventory_key: str | None,
    models: list[ModelConfig],
    inventory_recommendations: dict[str, dict[str, Any]] | None = None,
    *,
    allow_demo: bool = False,
    custom_item_name: str | None = None,
    custom_alternatives: str | None = None,
) -> dict[str, Any]:
    """
    Resolve the recommended model for an inventory key.

    Priority:
    1. Inventory-specific default model
    2. First enabled recommended / alternative model
    3. Application-wide default model
    4. First enabled and valid (non-demo) model
    5. None → configuration error
    """
    sources: list[str] = ["inventory profiles"]
    enabled = non_demo_enabled_models(models, allow_demo=allow_demo)
    enabled_by_name = {m.name: m for m in enabled}

    # Merge JSON profile recommendations with any passed map (config.py).
    rec = recommendation_dict_for(inventory_key)
    if inventory_recommendations and inventory_key in inventory_recommendations:
        # Caller overrides win for missing fields only when profile empty
        overlay = inventory_recommendations.get(inventory_key) or {}
        for k, v in overlay.items():
            if v not in (None, "", [], {}):
                rec[k] = v

    # Custom item: always prefer YOLO-World (dynamic prompts)
    if is_custom_inventory(inventory_key):
        rec.setdefault("default_model", "YOLO-World")
        prompts, _ = effective_prompts_for_inventory(
            inventory_key,
            custom_item_name=custom_item_name,
            custom_alternatives=custom_alternatives,
        )
        if prompts:
            rec["prompt"] = ", ".join(prompts)
            rec["allowed_classes"] = list(prompts)

    chosen: ModelConfig | None = None
    reason = ""

    default_name = rec.get("default_model")
    if default_name and default_name in enabled_by_name:
        chosen = enabled_by_name[default_name]
        reason = "inventory-specific default model"
        sources.append("inventory profiles")

    if chosen is None:
        for name in list(rec.get("recommended_models") or []) + list(
            rec.get("alternative_models") or []
        ):
            if name in enabled_by_name:
                chosen = enabled_by_name[name]
                reason = "first enabled recommended model"
                break

    if chosen is None:
        app_default = get_default_model(models)
        if app_default and app_default.name in enabled_by_name:
            chosen = enabled_by_name[app_default.name]
            reason = "application-wide default model"
            sources.append("models.json")

    if chosen is None and enabled:
        # Prefer a dynamic-prompt model for non-fence inventories
        dyn = next((m for m in enabled if m.supports_prompt or m.dynamic_classes), None)
        chosen = dyn or enabled[0]
        reason = "first enabled and valid model"
        sources.append("models.json")

    display = inventory_display_name(inventory_key, custom_item_name=custom_item_name)

    if chosen is None:
        return {
            "ok": False,
            "model": None,
            "model_name": None,
            "reason": "no_valid_model",
            "error": f"No valid AI model is configured for {display or 'this inventory'}.",
            "sources": list(dict.fromkeys(sources + ["models.json"])),
            "prompt": rec.get("prompt") or "",
            "allowed_classes": list(rec.get("allowed_classes") or []),
            "confidence_threshold": float(rec.get("confidence_threshold") or 0.25),
            "counting_strategy": rec.get("counting_strategy") or "",
            "inventory_key": inventory_key,
            "inventory_display": display,
            "provider": None,
            "workflow_id": None,
            "counting_note": rec.get("counting_note") or "",
            "counting_unit": counting_unit_for(
                inventory_key, custom_item_name=custom_item_name
            ),
        }

    prompts, prompt_errs = effective_prompts_for_inventory(
        inventory_key,
        custom_item_name=custom_item_name,
        custom_alternatives=custom_alternatives,
    )
    prompt = ", ".join(prompts) if prompts else (rec.get("prompt") or "").strip()
    allowed = list(rec.get("allowed_classes") or chosen.allowed_classes or prompts)
    if not prompt and allowed:
        prompt = ", ".join(allowed)
    if not prompt and chosen.supports_prompt:
        prompt = inventory_key or ""

    conf = rec.get("confidence_threshold")
    if conf is None:
        conf = chosen.default_confidence if chosen.default_confidence is not None else 0.25
    profile = get_profile(inventory_key) or {}
    if profile.get("default_confidence") is not None and rec.get("confidence_threshold") is None:
        conf = profile["default_confidence"]

    strategy = (
        rec.get("counting_strategy")
        or chosen.counting_strategy
        or "object_detection"
    )

    sources = list(dict.fromkeys(sources + ["models.json"]))
    if (chosen.kind or "").lower() == "workflow":
        sources.append("deployed workflow")

    return {
        "ok": True,
        "model": chosen,
        "model_name": chosen.name,
        "reason": reason,
        "error": None if not prompt_errs else "; ".join(prompt_errs),
        "prompt_errors": prompt_errs,
        "sources": sources,
        "prompt": prompt,
        "effective_prompts": prompts,
        "allowed_classes": allowed,
        "confidence_threshold": float(conf),
        "counting_strategy": strategy,
        "inventory_key": inventory_key,
        "inventory_display": display,
        "provider": chosen.provider
        or ("Roboflow" if (chosen.kind or "").lower() == "workflow" else "Local"),
        "workflow_id": chosen.workflow_id,
        "counting_note": rec.get("counting_note") or profile.get("description") or "",
        "counting_unit": counting_unit_for(
            inventory_key, custom_item_name=custom_item_name
        ),
        "model_key": getattr(chosen, "key", None) or chosen.name,
        "custom_item_name": custom_item_name if is_custom_inventory(inventory_key) else None,
    }


def form_updates_from_recommendation(
    resolved: dict[str, Any],
    *,
    apply_selection: bool = True,
) -> dict[str, Any]:
    """Session form fields derived from a resolved recommendation.

    When apply_selection is False, never overwrite selected_models / selected_mode
    (Analyze page must not reset the user's model pick every rerun).
    """
    updates: dict[str, Any] = {
        "photo_relationship": FIXED_PHOTO_RELATIONSHIP,
        "recommended_setup_resolved": bool(resolved.get("ok")),
        "recommended_model_name": resolved.get("model_name") or "",
        "recommended_setup_error": resolved.get("error") or "",
        "counting_unit": resolved.get("counting_unit") or "",
    }
    if resolved.get("ok") and resolved.get("model_name"):
        if apply_selection:
            updates["selected_models"] = [resolved["model_name"]]
            updates["selected_mode"] = "Single Model"
        prompt = resolved.get("prompt") or ""
        updates["prompt"] = prompt
        updates["prompt_preset"] = prompt
        updates["class_override"] = prompt
        updates["confidence_threshold"] = float(resolved.get("confidence_threshold") or 0.25)
        updates["counting_strategy"] = resolved.get("counting_strategy") or ""
        if resolved.get("effective_prompts") is not None:
            updates["effective_prompts"] = list(resolved.get("effective_prompts") or [])
    elif apply_selection:
        updates["selected_models"] = []
    return updates


def suggest_model_from_trial_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Transparent suggestion from optional model trials (no ground-truth accuracy claims).

    rows: status, model_name, raw_count, final_count, avg_confidence,
          max_confidence, processing_time, warnings (list|str)
    """
    basis: list[str] = []
    eligible = [
        r
        for r in rows
        if (r.get("status") or "").upper() in {"OK", "SUCCESS", "COMPLETE"}
        and not r.get("error_type")
    ]
    if not eligible:
        return {
            "suggested_model": None,
            "label": "Suggested model for this run",
            "basis": ["No successful model trials to compare."],
            "rows": rows,
        }

    def _warn_count(r: dict[str, Any]) -> int:
        w = r.get("warnings") or []
        if isinstance(w, str):
            return 1 if w.strip() else 0
        return len(w)

    with_objects = [r for r in eligible if int(r.get("final_count") or 0) > 0]
    pool = with_objects or eligible
    if with_objects:
        basis.append("Valid object-level detections")

    no_warn = [r for r in pool if _warn_count(r) == 0]
    if no_warn:
        pool = no_warn
        basis.append("No parser warnings")

    pool = sorted(
        pool,
        key=lambda r: (
            -float(r.get("avg_confidence") or 0.0),
            float(r.get("processing_time") or r.get("processing_time_seconds") or 1e9),
        ),
    )
    best = pool[0]
    basis.append("Highest average confidence")
    if len(pool) > 1:
        basis.append("Faster processing time as a tie-breaker")

    return {
        "suggested_model": best.get("model_name"),
        "label": "Suggested model for this run",
        "basis": basis,
        "rows": rows,
    }
