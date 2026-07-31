"""Administrator-managed sample images stored under DATA_DIR.

Uploads are validated by decoding the image (not by trusting the declared
content type), stored under a generated safe filename, and described by a row
in ``admin_samples``. Files never land outside the sample directory.
"""

from __future__ import annotations

import io
import re
import secrets
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

import config
from database import _connect, initialize_database, utc_now_iso

SUPPORTED_FORMATS = {"JPEG": ".jpg", "PNG": ".png"}
MAX_SAMPLE_BYTES = 15 * 1024 * 1024
MIN_DIMENSION = 64
MAX_DIMENSION = 8000


class SampleValidationError(ValueError):
    """Raised when an uploaded sample fails validation."""


@dataclass(frozen=True)
class AdminSample:
    id: int
    sample_id: str
    filename: str
    title: str = ""
    description: str = ""
    inventory_type: str = ""
    expected_count: int | None = None
    is_enabled: bool = True
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    mime_type: str = ""
    uploaded_by: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def path(self) -> Path:
        return samples_dir() / self.filename

    @property
    def exists(self) -> bool:
        try:
            return self.path.is_file()
        except OSError:
            return False


def samples_dir() -> Path:
    """Directory that holds administrator-uploaded sample files."""
    directory = Path(config.DATA_DIR) / "admin_samples"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(raw: str, *, fallback: str = "sample") -> str:
    """ASCII, lowercase, filesystem-safe token with no path separators."""
    text = unicodedata.normalize("NFKD", str(raw or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    text = re.sub(r"_{2,}", "_", text)
    return (text or fallback)[:48]


def validate_image_bytes(data: bytes) -> dict[str, Any]:
    """Decode and check an upload; returns image metadata or raises."""
    if not data:
        raise SampleValidationError("The uploaded file is empty.")
    if len(data) > MAX_SAMPLE_BYTES:
        limit_mb = MAX_SAMPLE_BYTES // (1024 * 1024)
        raise SampleValidationError(f"Sample images must be {limit_mb} MB or smaller.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fmt = (image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise SampleValidationError(
            "That file could not be read as an image. Upload a JPEG or PNG."
        ) from exc

    if fmt not in SUPPORTED_FORMATS:
        raise SampleValidationError("Only JPEG and PNG sample images are supported.")
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise SampleValidationError(
            f"Sample images must be at least {MIN_DIMENSION}×{MIN_DIMENSION} pixels."
        )
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise SampleValidationError(
            f"Sample images must be at most {MAX_DIMENSION}×{MAX_DIMENSION} pixels."
        )

    return {
        "format": fmt,
        "suffix": SUPPORTED_FORMATS[fmt],
        "width": width,
        "height": height,
        "size_bytes": len(data),
        "mime_type": "image/jpeg" if fmt == "JPEG" else "image/png",
    }


def add_sample(
    *,
    data: bytes,
    title: str,
    inventory_type: str,
    description: str = "",
    expected_count: int | None = None,
    uploaded_by: str = "",
    is_enabled: bool = True,
    db_path: str | None = None,
) -> AdminSample:
    """Validate, store and register a new sample image."""
    if not str(title or "").strip():
        raise SampleValidationError("Give the sample a title.")

    meta = validate_image_bytes(data)
    base = slugify(title)
    sample_id = f"{base}_{secrets.token_hex(4)}"
    filename = f"{sample_id}{meta['suffix']}"

    target = samples_dir() / filename
    # Defence in depth: the generated name cannot escape, but verify anyway.
    if target.parent.resolve() != samples_dir().resolve():
        raise SampleValidationError("Invalid sample destination.")

    temp = target.with_suffix(target.suffix + ".tmp")
    try:
        temp.write_bytes(data)
        temp.replace(target)
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise SampleValidationError("The sample file could not be saved.") from exc

    now = utc_now_iso()
    initialize_database(db_path)
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO admin_samples
                    (sample_id, filename, title, description, inventory_type,
                     expected_count, is_enabled, width, height, size_bytes,
                     mime_type, uploaded_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    filename,
                    str(title).strip(),
                    str(description or "").strip(),
                    str(inventory_type or "").strip(),
                    None if expected_count is None else int(expected_count),
                    1 if is_enabled else 0,
                    meta["width"],
                    meta["height"],
                    meta["size_bytes"],
                    meta["mime_type"],
                    str(uploaded_by or ""),
                    now,
                    now,
                ),
            )
            row_id = int(cur.lastrowid)
    except Exception as exc:  # noqa: BLE001 — roll back the orphaned file
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise SampleValidationError("The sample could not be registered.") from exc

    stored = get_sample(row_id, db_path=db_path)
    if stored is None:
        raise SampleValidationError("The sample could not be registered.")
    return stored


def _row_to_sample(row: Any) -> AdminSample:
    data = dict(row)
    return AdminSample(
        id=int(data["id"]),
        sample_id=str(data.get("sample_id") or ""),
        filename=str(data.get("filename") or ""),
        title=str(data.get("title") or ""),
        description=str(data.get("description") or ""),
        inventory_type=str(data.get("inventory_type") or ""),
        expected_count=data.get("expected_count"),
        is_enabled=bool(data.get("is_enabled", 1)),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        size_bytes=int(data.get("size_bytes") or 0),
        mime_type=str(data.get("mime_type") or ""),
        uploaded_by=str(data.get("uploaded_by") or ""),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
    )


def get_sample(row_id: int, *, db_path: str | None = None) -> AdminSample | None:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM admin_samples WHERE id = ?", (int(row_id),)
        ).fetchone()
    return _row_to_sample(row) if row else None


def list_samples(
    *, include_disabled: bool = True, db_path: str | None = None
) -> list[AdminSample]:
    initialize_database(db_path)
    sql = "SELECT * FROM admin_samples"
    if not include_disabled:
        sql += " WHERE is_enabled = 1"
    sql += " ORDER BY created_at DESC"
    with _connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [_row_to_sample(row) for row in rows]


def update_sample(
    row_id: int,
    *,
    title: str | None = None,
    description: str | None = None,
    inventory_type: str | None = None,
    expected_count: int | None = None,
    is_enabled: bool | None = None,
    db_path: str | None = None,
) -> AdminSample | None:
    fields: dict[str, Any] = {}
    if title is not None:
        fields["title"] = str(title).strip()
    if description is not None:
        fields["description"] = str(description).strip()
    if inventory_type is not None:
        fields["inventory_type"] = str(inventory_type).strip()
    if expected_count is not None:
        fields["expected_count"] = int(expected_count)
    if is_enabled is not None:
        fields["is_enabled"] = 1 if is_enabled else 0
    if not fields:
        return get_sample(row_id, db_path=db_path)

    fields["updated_at"] = utc_now_iso()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE admin_samples SET {assignments} WHERE id = ?",
            list(fields.values()) + [int(row_id)],
        )
    return get_sample(row_id, db_path=db_path)


def delete_sample(row_id: int, *, db_path: str | None = None) -> str:
    """Remove the row and its file; returns the sample_id that was deleted."""
    sample = get_sample(row_id, db_path=db_path)
    if sample is None:
        return ""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM admin_samples WHERE id = ?", (int(row_id),))
    try:
        sample.path.unlink(missing_ok=True)
    except OSError:
        pass
    return sample.sample_id


def read_sample_bytes(sample: AdminSample) -> bytes:
    try:
        return sample.path.read_bytes()
    except OSError as exc:
        raise SampleValidationError("The sample file is missing from storage.") from exc
