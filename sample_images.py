"""Built-in sample-image library loaded from project-owned assets/sample_images/."""

from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from config import PROJECT_ROOT

SAMPLE_IMAGE_DIR = PROJECT_ROOT / "assets" / "sample_images"
MANIFEST_PATH = SAMPLE_IMAGE_DIR / "manifest.json"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# Manifest inventory_type → app inventory key
INVENTORY_TYPE_MAP = {
    "fence_panels": "Fence Panel",
    "fence_panel": "Fence Panel",
    "Fence Panel": "Fence Panel",
    "traffic_cones": "Traffic Cones",
    "Traffic Cones": "Traffic Cones",
    "chairs": "Chairs",
    "Chairs": "Chairs",
    "boxes": "Boxes",
    "Boxes": "Boxes",
    "pallets": "Pallets",
    "Pallets": "Pallets",
    "cars": "Cars",
    "Cars": "Cars",
    "bottles": "Bottles",
    "Bottles": "Bottles",
    "gates": "Gates",
    "Gates": "Gates",
    "poles": "Poles",
    "Poles": "Poles",
    "custom_item": "Custom Item",
    "Custom Item": "Custom Item",
}


@dataclass
class SampleImage:
    id: str
    filename: str
    title: str
    description: str = ""
    inventory_type: str = "fence_panels"
    enabled: bool = True
    featured: bool = False
    source: str = "project_sample"
    license: str = "Provided for this project"
    path: Path | None = None
    width: int = 0
    height: int = 0
    mime_type: str = "image/jpeg"
    size_bytes: int = 0
    decode_ok: bool = False
    benchmark: dict[str, Any] | None = None

    @property
    def app_inventory_key(self) -> str:
        if self.benchmark and self.benchmark.get("inventory_key"):
            return str(self.benchmark["inventory_key"])
        return INVENTORY_TYPE_MAP.get(self.inventory_type, "")

    @property
    def relative_path(self) -> str:
        if self.path is None:
            return self.filename
        try:
            return str(self.path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return self.filename


@dataclass
class SampleLibraryStatus:
    directory_exists: bool = False
    manifest_exists: bool = False
    manifest_valid: bool = False
    manifest_error: str | None = None
    valid_count: int = 0
    enabled_count: int = 0
    missing_files: list[str] = field(default_factory=list)
    invalid_files: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    samples: list[SampleImage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory_exists": self.directory_exists,
            "manifest_exists": self.manifest_exists,
            "manifest_valid": self.manifest_valid,
            "manifest_error": self.manifest_error,
            "valid_count": self.valid_count,
            "enabled_count": self.enabled_count,
            "missing_files": list(self.missing_files),
            "invalid_files": list(self.invalid_files),
            "duplicate_ids": list(self.duplicate_ids),
            "warnings": list(self.warnings),
        }


_STATUS_CACHE: SampleLibraryStatus | None = None


def clear_sample_library_cache() -> None:
    global _STATUS_CACHE
    _STATUS_CACHE = None


def load_sample_library(*, force_reload: bool = False) -> SampleLibraryStatus:
    """Load and validate the sample-image manifest. Never raises for bad entries."""
    global _STATUS_CACHE
    if _STATUS_CACHE is not None and not force_reload:
        return _STATUS_CACHE

    status = SampleLibraryStatus()
    status.directory_exists = SAMPLE_IMAGE_DIR.is_dir()
    if not status.directory_exists:
        status.warnings.append("Sample image directory is missing.")
        _STATUS_CACHE = status
        return status

    status.manifest_exists = MANIFEST_PATH.is_file()
    if not status.manifest_exists:
        status.warnings.append("Sample image manifest.json is missing.")
        _STATUS_CACHE = status
        return status

    try:
        raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        status.manifest_valid = False
        status.manifest_error = f"Invalid manifest JSON: {exc}"
        status.warnings.append(status.manifest_error)
        _STATUS_CACHE = status
        return status

    if not isinstance(raw, dict) or not isinstance(raw.get("images"), list):
        status.manifest_valid = False
        status.manifest_error = "Manifest must be an object with an 'images' array."
        status.warnings.append(status.manifest_error)
        _STATUS_CACHE = status
        return status

    status.manifest_valid = True
    seen_ids: set[str] = set()

    for entry in raw["images"]:
        if not isinstance(entry, dict):
            status.warnings.append("Skipped non-object manifest entry.")
            continue
        sid = str(entry.get("id") or "").strip()
        filename = str(entry.get("filename") or "").strip()
        if not sid or not filename:
            status.warnings.append("Skipped manifest entry missing id or filename.")
            continue
        if sid in seen_ids:
            status.duplicate_ids.append(sid)
            status.warnings.append(f"Duplicate sample id: {sid}")
            continue
        seen_ids.add(sid)

        path = SAMPLE_IMAGE_DIR / filename
        from benchmark import parse_benchmark_metadata

        sample = SampleImage(
            id=sid,
            filename=filename,
            title=str(entry.get("title") or sid),
            description=str(entry.get("description") or ""),
            inventory_type=str(entry.get("inventory_type") or "fence_panels"),
            enabled=bool(entry.get("enabled", True)),
            featured=bool(entry.get("featured", False)),
            source=str(entry.get("source") or "project_sample"),
            license=str(entry.get("license") or "Provided for this project"),
            path=path,
            benchmark=parse_benchmark_metadata(entry),
        )

        if not path.is_file():
            status.missing_files.append(filename)
            status.warnings.append(f"Missing sample file: {filename}")
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            status.invalid_files.append(filename)
            status.warnings.append(f"Unsupported sample file type: {filename}")
            continue

        sample.mime_type = MIME_BY_SUFFIX.get(
            suffix, mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        try:
            sample.size_bytes = path.stat().st_size
            with Image.open(path) as img:
                sample.width, sample.height = img.size
                img.verify()
            # Re-open after verify
            with Image.open(path) as img2:
                sample.width, sample.height = img2.size
            sample.decode_ok = True
        except Exception:  # noqa: BLE001
            status.invalid_files.append(filename)
            status.warnings.append(f"Cannot decode sample file: {filename}")
            continue

        status.samples.append(sample)
        status.valid_count += 1
        if sample.enabled:
            status.enabled_count += 1

    _STATUS_CACHE = status
    return status


def list_enabled_samples(
    *,
    inventory_key: str | None = "Fence Panel",
    include_disabled: bool = False,
) -> list[SampleImage]:
    """Return valid samples for the gallery (default: Fence Panels only)."""
    status = load_sample_library()
    out: list[SampleImage] = []
    for s in status.samples:
        if not s.decode_ok:
            continue
        if not include_disabled and not s.enabled:
            continue
        if inventory_key:
            if s.app_inventory_key != inventory_key:
                continue
        out.append(s)
    # Featured first, then title
    out.sort(key=lambda s: (not s.featured, s.title.lower()))
    return out


def get_sample_by_id(sample_id: str) -> SampleImage | None:
    status = load_sample_library()
    for s in status.samples:
        if s.id == sample_id and s.decode_ok:
            return s
    return None


def read_sample_bytes(sample: SampleImage) -> bytes:
    if sample.path is None or not sample.path.is_file():
        raise FileNotFoundError(sample.filename)
    return sample.path.read_bytes()


def sample_library_diagnostics_warnings() -> list[str]:
    """Warnings suitable for Diagnostics (no absolute server paths)."""
    status = load_sample_library()
    return list(status.warnings)
