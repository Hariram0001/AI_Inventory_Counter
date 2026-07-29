"""Shared Model confidence formatting and display bands (not accuracy claims)."""

from __future__ import annotations

# UI-only bands. Not used as automatic exclude thresholds.
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.50

CONFIDENCE_HELP = (
    "The model confidence indicates how strongly the model matched this region "
    "to the predicted class. It is not a measured accuracy score."
)

CONFIDENCE_LABEL = "Model confidence"
CONFIDENCE_LABEL_SHORT = "Confidence"


def format_confidence_percent(value: float | None, *, decimals: int = 1) -> str:
    """Format a 0–1 confidence as a percentage string (e.g. 0.552 → 55.2%)."""
    if value is None:
        return "—"
    pct = max(0.0, min(100.0, float(value) * 100.0))
    if decimals <= 0:
        return f"{pct:.0f}%"
    text = f"{pct:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{text}%"


def confidence_band(value: float | None) -> str:
    """Return High / Medium / Low for display only."""
    if value is None:
        return "—"
    v = float(value)
    if v >= CONFIDENCE_HIGH:
        return "High"
    if v >= CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def is_low_confidence_warning(
    value: float | None,
    *,
    warning_threshold: float = CONFIDENCE_MEDIUM,
) -> bool:
    """True when confidence is below the review-warning display threshold."""
    if value is None:
        return False
    return float(value) < float(warning_threshold)
