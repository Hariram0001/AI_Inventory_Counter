"""Application configuration loaded from environment variables / Streamlit secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_JSON_PATH = PROJECT_ROOT / "models.json"
MOCK_RESPONSE_PATH = PROJECT_ROOT / "sample_responses" / "mock_detection.json"
SAMPLE_IMAGE_DIR = PROJECT_ROOT / "assets" / "sample_images"
SAMPLE_IMAGE_MANIFEST_PATH = SAMPLE_IMAGE_DIR / "manifest.json"

# Mutable runtime settings (refreshed via reload_settings)
DATA_DIR: Path
DB_PATH: Path
DEMO_MODE: bool
ROBOFLOW_API_KEY: str
ROBOFLOW_API_URL: str
ROBOFLOW_WORKSPACE: str
ROBOFLOW_WORKFLOW_ID: str
MAX_UPLOAD_BYTES: int
MAX_INFERENCE_DIMENSION: int
MAX_TILES_PER_IMAGE: int
MAX_API_CALLS_PER_IMAGE: int
API_CALL_CONFIRM_THRESHOLD: int
INFERENCE_TIMEOUT_SECONDS: float


def _truthy(value: Any, *, default: bool = False) -> bool:
    """Parse bool-like env/secret values safely (bool, str, int)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def _secret_get(name: str) -> str | None:
    """Read from Streamlit secrets when available; never crash outside Streamlit."""
    try:
        import streamlit as st  # lazy — pytest / CLI may not have an active runtime

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        # st.secrets supports mapping access; missing keys raise
        try:
            val = secrets[name]
        except Exception:  # noqa: BLE001 — KeyError / StreamlitSecretNotFoundError / etc.
            return None
        if val is None:
            return None
        return str(val).strip()
    except Exception:  # noqa: BLE001 — streamlit missing or secrets unavailable
        return None


def _setting(name: str, default: str = "") -> str:
    """Priority: Streamlit secrets → process env → default."""
    secret = _secret_get(name)
    if secret is not None and secret != "":
        return secret
    env = os.getenv(name)
    if env is not None and str(env).strip() != "":
        return str(env).strip()
    return default


def reload_settings() -> None:
    """Reload .env / secrets into process settings."""
    global DATA_DIR, DB_PATH, DEMO_MODE, ROBOFLOW_API_KEY, ROBOFLOW_API_URL
    global ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID
    global MAX_UPLOAD_BYTES, MAX_INFERENCE_DIMENSION, MAX_TILES_PER_IMAGE
    global MAX_API_CALLS_PER_IMAGE, API_CALL_CONFIRM_THRESHOLD, INFERENCE_TIMEOUT_SECONDS

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    data_dir_raw = _setting("DATA_DIR", str(PROJECT_ROOT / "data"))
    DATA_DIR = Path(data_dir_raw)
    if not DATA_DIR.is_absolute():
        DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()
    DB_PATH = DATA_DIR / "inventory_counts.db"

    DEMO_MODE = _truthy(_setting("DEMO_MODE", "false"), default=False)
    ROBOFLOW_API_KEY = _setting("ROBOFLOW_API_KEY", "")
    ROBOFLOW_API_URL = _setting(
        "ROBOFLOW_API_URL", "https://serverless.roboflow.com"
    )
    ROBOFLOW_WORKSPACE = _setting("ROBOFLOW_WORKSPACE", "hariram-s-mzhvc")
    ROBOFLOW_WORKFLOW_ID = _setting("ROBOFLOW_WORKFLOW_ID", "custom-workflow")
    MAX_UPLOAD_BYTES = int(_setting("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
    MAX_INFERENCE_DIMENSION = int(_setting("MAX_INFERENCE_DIMENSION", "2048"))
    MAX_TILES_PER_IMAGE = int(_setting("MAX_TILES_PER_IMAGE", "60"))
    MAX_API_CALLS_PER_IMAGE = int(_setting("MAX_API_CALLS_PER_IMAGE", "60"))
    API_CALL_CONFIRM_THRESHOLD = int(_setting("API_CALL_CONFIRM_THRESHOLD", "30"))
    INFERENCE_TIMEOUT_SECONDS = float(_setting("INFERENCE_TIMEOUT_SECONDS", "120"))


reload_settings()

# Default tiling
DEFAULT_TILE_SIZE = 800
DEFAULT_TILE_OVERLAP = 0.25

# Default prompts by inventory type (legacy single-string; prefer inventory_profiles.json)
DEFAULT_PROMPTS = {
    "Fence Panel": "fence panel, wooden fence panel, privacy fence panel",
    "Pallets": "wooden pallet, shipping pallet, stacked pallet",
    "Boxes": "cardboard box, shipping box, storage box",
    "Poles": "metal pole, fence pole, steel pole",
    "Gates": "fence gate, metal gate, yard gate",
    "Chairs": "chair, folding chair, outdoor chair",
    "Traffic Cones": "traffic cone, road cone, safety cone",
}

# Legacy in-module profiles kept for import compatibility; source of truth is
# inventory_profiles.json via inventory_profiles.py.
INVENTORY_PROFILES: dict[str, dict[str, Any]] = {
    "Fence Panel": {
        "display_name": "Fence Panels",
        "detection_queries": [
            "fence panel",
            "wooden fence panel",
            "privacy fence panel",
        ],
        "allowed_result_classes": [
            "fence panel",
            "wooden fence panel",
            "privacy fence panel",
            "wood fence",
        ],
        "default_model": "YOLO-World",
        "confidence_threshold": 0.25,
        "counting_strategy": "Object Detection",
        "counting_note": (
            "YOLO-World may detect the whole fence as one object depending on the photo "
            "and prompt. Individual panel counting is not guaranteed."
        ),
    },
}

FENCE_PANEL_PROMPT_PRESETS = [
    "fence panel",
    "wooden fence panel",
    "privacy fence panel",
    "wood fence",
    "fence picket",
    "wooden picket",
    "fence post",
    "temporary fence panel",
]

DETECTION_PROMPT_HINTS = [
    "wood fence",
    "fence panel",
    "traffic cone",
    "wooden pallet",
    "metal pole",
    "chair",
]


def _sync_inventory_from_profiles() -> None:
    """Refresh INVENTORY_PROFILES / recommendations / types from JSON registry."""
    global INVENTORY_PROFILES, INVENTORY_MODEL_RECOMMENDATIONS, INVENTORY_TYPES, DEFAULT_PROMPTS
    try:
        from inventory_profiles import all_recommendations, load_inventory_profiles

        profiles = load_inventory_profiles()
        if not profiles:
            return
        synced: dict[str, dict[str, Any]] = {}
        defaults: dict[str, str] = dict(DEFAULT_PROMPTS)
        keys: list[str] = []
        for p in profiles:
            key = p["key"]
            keys.append(key)
            terms = list(p.get("prompt_terms") or [])
            synced[key] = {
                "display_name": p.get("display_name") or key,
                "detection_queries": terms,
                "allowed_result_classes": list(p.get("allowed_result_classes") or terms),
                "default_model": p.get("default_model") or "YOLO-World",
                "confidence_threshold": float(p.get("default_confidence") or 0.25),
                "counting_strategy": "Object Detection",
                "counting_note": p.get("description") or "",
                "counting_unit": p.get("counting_unit") or "individual item",
                "enabled": bool(p.get("enabled", True)),
                "is_custom": bool(p.get("is_custom", False)),
            }
            if terms:
                defaults[key] = ", ".join(terms)
        INVENTORY_PROFILES = synced
        INVENTORY_MODEL_RECOMMENDATIONS = all_recommendations()
        INVENTORY_TYPES = keys
        DEFAULT_PROMPTS = defaults
    except Exception:
        # Keep module-level fallbacks if JSON cannot be loaded at import time.
        pass


# Inventory → recommended model names (must match models.json display names)
INVENTORY_MODEL_RECOMMENDATIONS: dict[str, dict[str, Any]] = {
    "Fence Panel": {
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
        "counting_note": (
            "YOLO-World may detect the whole fence as one object depending on the photo "
            "and prompt. Individual panel counting is not guaranteed."
        ),
    },
}


def inventory_detection_prompt(inventory_key: str | None) -> str:
    """Comma-separated detection queries for dynamic models."""
    try:
        from inventory_profiles import effective_prompts_for_inventory, prompts_to_csv

        prompts, _ = effective_prompts_for_inventory(inventory_key)
        if prompts:
            return prompts_to_csv(prompts)
    except Exception:
        pass
    profile = INVENTORY_PROFILES.get(inventory_key or "") or {}
    queries = profile.get("detection_queries") or []
    if queries:
        return ", ".join(queries)
    if inventory_key in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[inventory_key]
    return (inventory_key or "").strip()


def inventory_display_name_config(inventory_key: str | None) -> str:
    try:
        from inventory_profiles import inventory_display_name

        return inventory_display_name(inventory_key)
    except Exception:
        pass
    profile = INVENTORY_PROFILES.get(inventory_key or "") or {}
    if profile.get("display_name"):
        return str(profile["display_name"])
    return inventory_key or ""

YARD_OPTIONS = ["LA Yard", "Dallas Yard", "Houston Yard", "Other"]
INVENTORY_TYPES = [
    "Fence Panel",
    "Pallets",
    "Boxes",
    "Poles",
    "Gates",
    "Chairs",
    "Traffic Cones",
    "Custom Item",
]

_sync_inventory_from_profiles()

TILE_SIZES = [512, 640, 800, 1024, 1280]
TILE_OVERLAPS = [0.10, 0.20, 0.25, 0.30, 0.40]

DISCLAIMER = (
    "This is an experimental AI estimate. Closely stacked, partially hidden, "
    "distant, or overlapping objects may be missed. All results must be reviewed "
    "before being saved as an official inventory count."
)

PHOTO_RELATIONSHIP_WARNING = (
    "Automatic cross-photo object identity matching is not included in this POC. "
    "Photographs of the same inventory must not be summed."
)

DEDUP_STRATEGY_EXPLAINER = (
    "Duplicate suppression removes repeated detections of the same object. "
    "It may occasionally remove separate objects that are very close together. "
    "Use Compare strategies when counting densely stacked inventory."
)


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def api_key_configured() -> bool:
    return bool(ROBOFLOW_API_KEY)


def masked_api_key_status() -> str:
    if DEMO_MODE:
        return "Demo Mode (API key not required)"
    if api_key_configured():
        return "Configured (hidden)"
    return "Missing"
