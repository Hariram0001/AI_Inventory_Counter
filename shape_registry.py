"""Shape detection registry — aliases and enabled detectors only.

UI and detectors resolve shapes through this module; do not hardcode aliases
elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "shape_registry.json"

UNSUPPORTED_SHAPE_MESSAGE = (
    "Only circle detection is available during the current testing phase."
)


@dataclass(frozen=True)
class ShapeSpec:
    key: str
    display_name: str
    enabled: bool
    aliases: tuple[str, ...]
    detector: str = ""
    status: str = ""

    @property
    def coming_soon(self) -> bool:
        return (not self.enabled) or self.status == "coming_soon"


class ShapeResolutionError(ValueError):
    """Raised when a requested shape cannot be resolved to an enabled detector."""

    def __init__(self, message: str, *, requested: str = "", status: str = "") -> None:
        super().__init__(message)
        self.requested = requested
        self.status = status


def _normalize_term(raw: str | None) -> str:
    text = " ".join(str(raw or "").strip().lower().split())
    return text.replace("_", " ").replace("-", " ")


@lru_cache(maxsize=1)
def load_registry() -> dict[str, ShapeSpec]:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    out: dict[str, ShapeSpec] = {}
    for key, raw in data.items():
        aliases = tuple(
            _normalize_term(a)
            for a in (raw.get("aliases") or [key, raw.get("display_name", key)])
            if _normalize_term(a)
        )
        out[str(key)] = ShapeSpec(
            key=str(key),
            display_name=str(raw.get("display_name") or key.title()),
            enabled=bool(raw.get("enabled")),
            aliases=aliases or (_normalize_term(key),),
            detector=str(raw.get("detector") or ""),
            status=str(raw.get("status") or ("enabled" if raw.get("enabled") else "coming_soon")),
        )
    return out


def reload_registry() -> dict[str, ShapeSpec]:
    load_registry.cache_clear()
    return load_registry()


def enabled_shapes() -> list[ShapeSpec]:
    return [s for s in load_registry().values() if s.enabled]


def coming_soon_shapes() -> list[ShapeSpec]:
    return [s for s in load_registry().values() if s.coming_soon]


def preset_options() -> list[str]:
    """Display names for enabled presets (for selectboxes)."""
    return [s.display_name for s in enabled_shapes()]


def resolve_shape(raw: str | None) -> ShapeSpec:
    """Resolve user text / preset to an enabled shape, or raise."""
    term = _normalize_term(raw)
    if not term:
        raise ShapeResolutionError(
            "Enter a shape to detect, or choose Circles.",
            requested=str(raw or ""),
        )

    registry = load_registry()
    # Exact key / display / alias match (case-insensitive via normalize).
    for spec in registry.values():
        candidates = {_normalize_term(spec.key), _normalize_term(spec.display_name)}
        candidates.update(spec.aliases)
        if term in candidates:
            if not spec.enabled:
                raise ShapeResolutionError(
                    UNSUPPORTED_SHAPE_MESSAGE,
                    requested=str(raw or ""),
                    status="coming_soon",
                )
            return spec

    # Unknown term — treat as unsupported for this testing phase.
    raise ShapeResolutionError(
        UNSUPPORTED_SHAPE_MESSAGE,
        requested=str(raw or ""),
        status="unsupported",
    )


def registry_public_dict() -> dict[str, dict[str, Any]]:
    """Stable dict for docs/tests — mirrors the JSON shape."""
    out: dict[str, dict[str, Any]] = {}
    for key, spec in load_registry().items():
        entry: dict[str, Any] = {
            "display_name": spec.display_name,
            "enabled": spec.enabled,
            "aliases": list(spec.aliases),
        }
        if spec.detector:
            entry["detector"] = spec.detector
        if spec.status:
            entry["status"] = spec.status
        out[key] = entry
    return out
