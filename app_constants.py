"""Plain application constants — no Streamlit, no project UI imports.

Import this module freely from app.py / ui_helpers without circular-import risk.
"""

from __future__ import annotations

SETTINGS_SECTIONS = (
    "ai_configuration",
    "history",
    "diagnostics",
)

SETTINGS_SECTION_LABELS: dict[str, str] = {
    "ai_configuration": "AI Configuration",
    "history": "Inventory History",
    "diagnostics": "Diagnostics",
}

SETTINGS_LABEL_TO_SECTION: dict[str, str] = {
    label: section for section, label in SETTINGS_SECTION_LABELS.items()
}

# Four-step wizard (aliases map older stage names into this flow)
STAGES = [
    "setup",
    "photos",
    "analyze",
    "review",
]

STAGE_LABELS = {
    "setup": "Inventory Setup",
    "photos": "Add Photos",
    "analyze": "Analyze",
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
    "save": "review",
    "config": "analyze",
}

VIEW_ALIASES = {
    "home": "welcome",
    # Do not map "setup" here — that name is the wizard Inventory Setup stage.
    "history": "settings",
    "diagnostics": "settings",
    "ai_configuration": "settings",
}

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
