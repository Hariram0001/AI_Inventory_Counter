"""Plain application constants — no Streamlit, no project UI imports.

Import this module freely from app.py / ui_helpers without circular-import risk.
"""

from __future__ import annotations

# Legacy names kept for older imports / docs helpers. Panels are top-level views now.
SETTINGS_SECTIONS = (
    "ai_configuration",
    "history",
    "diagnostics",
    "account",
    "api_keys",
)

SETTINGS_SECTION_LABELS: dict[str, str] = {
    "ai_configuration": "AI Configuration",
    "history": "Inventory History",
    "diagnostics": "Diagnostics",
    "account": "Account",
    "api_keys": "API Keys",
}

USER_SETTINGS_SECTIONS = (
    "ai_configuration",
    "history",
    "diagnostics",
    "account",
)

SETTINGS_LABEL_TO_SECTION: dict[str, str] = {
    label: section for section, label in SETTINGS_SECTION_LABELS.items()
}

PANEL_TITLES: dict[str, str] = {
    "welcome": "AI Inventory Counter",
    "home": "AI Inventory Counter",
    "wizard": "Counting workflow",
    "history": "Inventory History",
    "ai_configuration": "AI Configuration",
    "diagnostics": "Diagnostics",
    "account": "Profile",
    "api_keys": "API Keys",
    "admin": "Administration",
    "shape_detection": "Shape Detection",
}

PANEL_CAPTIONS: dict[str, str] = {
    "welcome": "Count visible inventory items from photos using AI-powered object detection.",
    "home": "Count visible inventory items from photos using AI-powered object detection.",
    "history": "Saved inventory analyses — filter, scan, and export from this panel.",
    "ai_configuration": "Connection status, model catalog, inventory prompts, and detection benchmarks.",
    "diagnostics": "Runtime health and troubleshooting for this deployment.",
    "account": "Your profile, sign-in details, and password.",
    "api_keys": "Roboflow status and the OpenRouter key for this deployment.",
    "admin": "Manage accounts, model access, demo samples, and connectivity for this deployment.",
    "wizard": "Choose inventory, add photos, run detection, then review and save.",
    "shape_detection": (
        "Detect likely visible circular shapes and circular objects using local "
        "computer vision. Testing Phase."
    ),
}

# Wizard stages (aliases map older stage names into this flow)
STAGES = [
    "setup",
    "photos",
    "analyze",
    "running",
    "review",
]

STAGE_LABELS = {
    "setup": "Inventory Setup",
    "photos": "Add Photos",
    "analyze": "Analyze",
    "running": "Running",
    "review": "Review & Save",
}

STAGE_ALIASES = {
    "inventory": "setup",
    "relationship": "setup",
    "details": "setup",
    "inventory_type": "setup",
    "photo_relationship": "setup",
    "upload": "photos",
    "analysis": "analyze",
    "run": "running",
    "running_analysis": "running",
    "save": "review",
    "config": "analyze",
}

# Top-level views. "welcome" is the post-login dashboard; unauthenticated
# visitors only ever see the login screen, which is rendered before dispatch.
VIEWS = (
    "welcome",
    "wizard",
    "history",
    "ai_configuration",
    "diagnostics",
    "account",
    "api_keys",
    "admin",
    "shape_detection",
)

# Views only an administrator may open.
ADMIN_ONLY_VIEWS = ("admin", "api_keys")

VIEW_ALIASES = {
    "home": "welcome",
    # Do not map "setup" here — that name is the wizard Inventory Setup stage.
    "profile": "account",
    "api_connections": "api_keys",
    "connections": "api_keys",
    "settings": "ai_configuration",
    "admin_console": "admin",
    "shapes": "shape_detection",
    "shape": "shape_detection",
}


def settings_sections_for_role(*, is_admin: bool) -> tuple[str, ...]:
    if is_admin:
        return SETTINGS_SECTIONS
    return USER_SETTINGS_SECTIONS


PHOTO_REL_DISPLAY = {
    "Different inventory in each photo": "Separate inventory areas",
    "Same inventory from multiple angles": "Same inventory from different angles",
}

PHOTO_REL_INTERNAL_TO_DISPLAY = {v: k for k, v in PHOTO_REL_DISPLAY.items()}

DEFAULT_REVIEW = {
    "use_direct": False,
    "direct_count": None,
    "false_positives": 0,
    "missed_items": 0,
    "notes": "",
}


def get_settings_section_from_label(label: str) -> str:
    return SETTINGS_LABEL_TO_SECTION.get(label, "ai_configuration")
