"""Model registry loaded from models.json plus session-only models."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from config import MODELS_JSON_PATH
from schemas import ModelConfig


def load_models_raw(path: Path | None = None) -> list[dict[str, Any]]:
    """Load raw dicts so unknown fields can be preserved on save."""
    models_path = path or MODELS_JSON_PATH
    if not models_path.exists():
        return []
    try:
        raw = json.loads(models_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def load_models_from_file(path: Path | None = None) -> list[ModelConfig]:
    return [ModelConfig.from_dict(item) for item in load_models_raw(path)]


def save_models_to_file(
    models: list[ModelConfig] | list[dict[str, Any]],
    path: Path | None = None,
    *,
    backup: bool = True,
) -> Path:
    """Atomically write models.json, preserving unknown fields from prior file."""
    models_path = path or MODELS_JSON_PATH
    previous = {str(d.get("name")): d for d in load_models_raw(models_path)}
    payload: list[dict[str, Any]] = []
    for item in models:
        if isinstance(item, ModelConfig):
            base = dict(previous.get(item.name) or {})
            base.update(item.to_dict())
            # Never persist API keys
            base.pop("api_key", None)
            base.pop("ROBOFLOW_API_KEY", None)
            payload.append(base)
        else:
            row = dict(item)
            row.pop("api_key", None)
            payload.append(row)

    text = json.dumps(payload, indent=2) + "\n"
    # Validate before replace
    json.loads(text)
    if backup and models_path.exists():
        bak = models_path.with_suffix(".json.bak")
        shutil.copy2(models_path, bak)
    models_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(models_path.parent),
        suffix=".tmp",
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(models_path)
    return models_path


def get_default_model(models: list[ModelConfig]) -> ModelConfig | None:
    """Return the explicitly marked default model, else first enabled valid model."""
    enabled = get_enabled_valid_models(models)
    for m in enabled:
        if m.is_default:
            return m
    return enabled[0] if enabled else None


def merge_session_models(
    file_models: list[ModelConfig],
    session_models: list[ModelConfig] | list[dict[str, Any]],
) -> list[ModelConfig]:
    merged = list(file_models)
    names = {m.name for m in merged}
    for item in session_models:
        model = item if isinstance(item, ModelConfig) else ModelConfig.from_dict(item, session_only=True)
        model.session_only = True
        if model.name in names:
            # Replace existing session entry with same name
            merged = [model if m.name == model.name else m for m in merged]
        else:
            merged.append(model)
            names.add(model.name)
    return merged


def get_enabled_valid_models(
    models: list[ModelConfig],
    *,
    allow_demo_ids: bool | None = None,
) -> list[ModelConfig]:
    """
    Return enabled models that pass validation.
    When allow_demo_ids is None, use config.DEMO_MODE (demo IDs only allowed in Demo Mode).
    """
    if allow_demo_ids is None:
        from config import DEMO_MODE

        allow_demo_ids = bool(DEMO_MODE)
    return [
        m
        for m in models
        if m.enabled and m.is_valid(allow_demo_ids=allow_demo_ids)
    ]


# Legacy display names → current registry names (session / history resilience)
MODEL_NAME_ALIASES: dict[str, str] = {
    "YOLO-World Fence Panel": "YOLO-World",
}


def normalize_model_name(name: str | None) -> str:
    if not name:
        return ""
    return MODEL_NAME_ALIASES.get(name, name)


def sanitize_selected_model_names(
    selected_names: list[str] | None,
    available_names: list[str],
) -> list[str]:
    """Drop stale/deleted model keys; map aliases; never invent replacements."""
    available = set(available_names)
    out: list[str] = []
    for raw in selected_names or []:
        name = normalize_model_name(raw)
        if name in available and name not in out:
            out.append(name)
    return out


def get_selectable_analysis_models(
    models: list[ModelConfig],
    inventory_key: str | None = "Fence Panel",
    *,
    allow_demo: bool = False,
    custom_item: bool | None = None,
) -> list[ModelConfig]:
    """
    Models shown in the Analysis selector.

    Live POC: enabled, live-validated, inventory-compatible object detectors
    with an implemented adapter. Custom Item normally includes only dynamic
    prompt models (YOLO-World).
    """
    if custom_item is None:
        custom_item = (inventory_key or "") == "Custom Item"
    # Prefer catalog-aware selector (excludes local/demo unless allow_demo).
    try:
        from model_catalog import get_selectable_models as catalog_selectable

        return catalog_selectable(
            inventory_key,
            allow_demo=allow_demo,
            custom_item=custom_item,
        )
    except Exception:  # noqa: BLE001
        pass

    enabled = get_enabled_valid_models(models, allow_demo_ids=allow_demo)
    if not allow_demo:
        enabled = [m for m in enabled if not m.is_demo_model_id() and not m.demo_only]

    out: list[ModelConfig] = []
    for m in enabled:
        kind = (m.kind or "").strip().lower()
        # Live Analysis: Roboflow workflow/model + optional Local Picket Counter.
        if kind not in {"workflow", "model", "local"}:
            continue
        if m.is_demo_model_id() and not allow_demo:
            continue
        if m.demo_only and kind != "local" and not allow_demo:
            continue
        if custom_item and not (m.dynamic_classes or m.supports_prompt):
            continue
        supported = list(m.supported_inventory_types or [])
        # Empty supported list = dynamic / any inventory (e.g. YOLO-World)
        if supported and inventory_key and inventory_key not in supported:
            continue
        if not supported and not (m.dynamic_classes or m.supports_prompt) and kind != "local":
            continue
        out.append(m)
    return out


def get_models_for_inventory(
    models: list[ModelConfig],
    inventory_type: str,
    require_enabled: bool = True,
) -> list[ModelConfig]:
    out: list[ModelConfig] = []
    for m in models:
        if require_enabled and not m.enabled:
            continue
        if not m.supports_inventory(inventory_type):
            continue
        out.append(m)
    return out


def validate_selection(
    models: list[ModelConfig],
    selected_names: list[str],
    mode: str,
) -> list[str]:
    errors: list[str] = []
    selected = [m for m in models if m.name in selected_names]
    if not selected:
        errors.append("No model selected.")
        return errors

    for m in selected:
        errors.extend(m.validation_errors())
        if not m.enabled and not m.session_only:
            errors.append(f"{m.name} is not enabled.")

    mode_key = (mode or "").lower()
    if "single" in mode_key and len(selected) != 1:
        errors.append("Single model mode requires exactly one model.")
    if "compare" in mode_key and not (2 <= len(selected) <= 3):
        errors.append("Compare models mode requires 2–3 models.")
    if "consensus" in mode_key and not (2 <= len(selected) <= 3):
        errors.append("Experimental consensus requires 2–3 models.")
    return errors


def summarize_models(models: list[ModelConfig]) -> list[dict[str, Any]]:
    from config import DEMO_MODE
    from model_adapters import model_key, provider_for

    allow_demo = bool(DEMO_MODE)
    rows = []
    for m in models:
        rows.append(
            {
                "name": m.name,
                "key": m.key or model_key(m),
                "provider": m.provider or provider_for(m),
                "kind": m.kind,
                "enabled": m.enabled,
                "default": bool(m.is_default),
                "valid": m.is_valid(allow_demo_ids=allow_demo),
                "session_only": m.session_only,
                "supports_prompt": m.supports_prompt,
                "dynamic_classes": bool(m.dynamic_classes),
                "demo_only": bool(m.demo_only),
                "counting_strategy": m.counting_strategy or "",
                "supported_inventory_types": ", ".join(m.supported_inventory_types) or "(any)",
                "supported_classes": ", ".join(m.allowed_classes) or "(dynamic/open)",
                "issues": "; ".join(m.validation_errors(allow_demo_ids=allow_demo)) or "OK",
            }
        )
    return rows
