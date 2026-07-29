"""Back-compat shim — canonical constants live in app_constants.py."""

from __future__ import annotations

from app_constants import (  # noqa: F401
    SETTINGS_LABEL_TO_SECTION,
    SETTINGS_SECTION_LABELS,
    SETTINGS_SECTIONS,
    get_settings_section_from_label,
)
