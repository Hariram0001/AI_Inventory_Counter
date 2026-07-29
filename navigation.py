"""Back-compat shim — prefer ui_helpers / app_constants."""

from __future__ import annotations

from app_constants import (  # noqa: F401
    DEFAULT_REVIEW,
    SETTINGS_SECTION_LABELS,
    SETTINGS_SECTIONS,
    STAGE_ALIASES,
    STAGE_LABELS,
    STAGES,
    VIEW_ALIASES,
    get_settings_section_from_label,
)
from ui_helpers import (  # noqa: F401
    default_form,
    leave_settings,
    navigate_to,
    normalize_stage,
    normalize_view,
    open_settings,
    reset_active_analysis,
)
