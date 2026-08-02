"""Persistence for shape-detection runs, items, and feature policy.

Separate from inventory history. Authorization is enforced here — UI filters
alone are not sufficient.
"""

from __future__ import annotations

import csv
import io
import json
import secrets
from pathlib import Path
from typing import Any

import config
from database import DatabaseError, _connect, initialize_database, utc_now_iso
from shape_detection_models import CircleDetection, ShapeDetectionResult

FEATURE_KEY = "shape_detection"
MSG_READ_ONLY = (
    "The result can be reviewed, but it could not be saved to local history."
)


DEFAULT_POLICY: dict[str, Any] = {
    "enabled_for_admins": True,
    "enabled_for_users": True,
    "max_image_bytes": None,  # None → use config.MAX_UPLOAD_BYTES
    "save_history_enabled": True,
    "notes": "Local OpenCV circle detection. No API key or inference quota.",
}


class ShapeAuthError(PermissionError):
    """Raised when a user may not view or mutate a shape-test record."""


class ShapeStorageError(RuntimeError):
    """Raised for persistence failures with a safe message."""


def shape_artifacts_dir() -> Path:
    directory = Path(config.DATA_DIR) / "shape_detection"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def annotated_dir() -> Path:
    directory = shape_artifacts_dir() / "annotated"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Feature policy
# ---------------------------------------------------------------------------


def get_feature_policy(
    feature_key: str = FEATURE_KEY, db_path: str | None = None
) -> dict[str, Any]:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM feature_policies WHERE feature_key = ?",
            (feature_key,),
        ).fetchone()
    if row is None:
        return dict(DEFAULT_POLICY)
    data = dict(row)
    return {
        "enabled_for_admins": bool(data.get("enabled_for_admins", 1)),
        "enabled_for_users": bool(data.get("enabled_for_users", 1)),
        "max_image_bytes": data.get("max_image_bytes"),
        "save_history_enabled": bool(data.get("save_history_enabled", 1)),
        "notes": str(data.get("notes") or ""),
        "updated_at": data.get("updated_at"),
        "updated_by": data.get("updated_by") or "",
    }


def upsert_feature_policy(
    *,
    enabled_for_admins: bool,
    enabled_for_users: bool,
    max_image_bytes: int | None,
    save_history_enabled: bool,
    notes: str = "",
    updated_by: str = "",
    feature_key: str = FEATURE_KEY,
    db_path: str | None = None,
) -> dict[str, Any]:
    initialize_database(db_path)
    now = utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO feature_policies (
                feature_key, display_name, enabled_for_admins, enabled_for_users,
                max_image_bytes, save_history_enabled, notes, updated_at, updated_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feature_key) DO UPDATE SET
                enabled_for_admins = excluded.enabled_for_admins,
                enabled_for_users = excluded.enabled_for_users,
                max_image_bytes = excluded.max_image_bytes,
                save_history_enabled = excluded.save_history_enabled,
                notes = excluded.notes,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (
                feature_key,
                "Shape Detection",
                1 if enabled_for_admins else 0,
                1 if enabled_for_users else 0,
                max_image_bytes,
                1 if save_history_enabled else 0,
                str(notes or ""),
                now,
                str(updated_by or ""),
            ),
        )
    return get_feature_policy(feature_key, db_path=db_path)


def ensure_default_feature_policy(db_path: str | None = None) -> None:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT feature_key FROM feature_policies WHERE feature_key = ?",
            (FEATURE_KEY,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO feature_policies (
                    feature_key, display_name, enabled_for_admins, enabled_for_users,
                    max_image_bytes, save_history_enabled, notes, updated_at, updated_by
                ) VALUES (?, ?, 1, 1, NULL, 1, ?, ?, '')
                """,
                (
                    FEATURE_KEY,
                    "Shape Detection",
                    DEFAULT_POLICY["notes"],
                    utc_now_iso(),
                ),
            )


def shape_detection_allowed(user, *, db_path: str | None = None) -> tuple[bool, str]:
    """Server-side gate for dashboard button and direct page access."""
    if user is None:
        return False, "Sign in to use Shape Detection."
    ensure_default_feature_policy(db_path=db_path)
    policy = get_feature_policy(db_path=db_path)
    if user.is_admin:
        if not policy.get("enabled_for_admins", True):
            return (
                False,
                "Shape Detection is currently disabled for administrators.",
            )
        return True, ""
    if not policy.get("enabled_for_users", True):
        return False, "Shape Detection is currently unavailable for your account."
    if not getattr(user, "is_active", True):
        return False, "Your account is not active."
    return True, ""


def max_image_bytes_for_policy(db_path: str | None = None) -> int:
    policy = get_feature_policy(db_path=db_path)
    configured = policy.get("max_image_bytes")
    if configured is not None and int(configured) > 0:
        return int(configured)
    return int(getattr(config, "MAX_UPLOAD_BYTES", 25 * 1024 * 1024))


# ---------------------------------------------------------------------------
# Runs / items
# ---------------------------------------------------------------------------


def _can_view_run(user, row: dict[str, Any]) -> bool:
    if user is None:
        return False
    if user.is_admin:
        return True
    owner = row.get("owner_user_id")
    if owner is None:
        return False  # legacy / unowned — admin only
    return int(owner) == int(user.user_id)


def save_shape_test(
    result: ShapeDetectionResult,
    *,
    user,
    source_type: str,
    original_filename: str = "",
    annotated_bytes: bytes | None = None,
    notes: str = "",
    db_path: str | None = None,
) -> int:
    """Persist a reviewed shape test. Returns run id."""
    if user is None:
        raise ShapeAuthError("Sign in to save a shape test.")
    allowed, message = shape_detection_allowed(user)
    if not allowed:
        raise ShapeAuthError(message)

    policy = get_feature_policy(db_path=db_path)
    if not policy.get("save_history_enabled", True):
        raise ShapeStorageError(MSG_READ_ONLY)

    initialize_database(db_path)
    annotated_rel = ""
    if annotated_bytes:
        token = secrets.token_hex(8)
        fname = f"shape_{token}.png"
        target = annotated_dir() / fname
        try:
            target.write_bytes(annotated_bytes)
            annotated_rel = f"shape_detection/annotated/{fname}"
        except OSError as exc:
            raise ShapeStorageError(MSG_READ_ONLY) from exc

    counts = {
        "detected": result.detected_count,
        "excluded": result.excluded_count,
        "manual": int(result.manually_added_count or 0),
        "final": result.final_count,
    }
    now = utc_now_iso()
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO shape_detection_runs (
                    owner_user_id, owner_username_snapshot, created_at,
                    requested_shape, normalized_shape, source_type,
                    original_filename_sanitized, image_hash,
                    original_width, original_height,
                    processed_width, processed_height,
                    detection_mode, target_type, parameter_summary_json,
                    detected_count, excluded_count, manually_added_count,
                    final_count, processing_time, notes,
                    annotated_image_path, status
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    int(user.user_id),
                    str(user.username),
                    now,
                    result.requested_shape,
                    result.normalized_shape,
                    str(source_type or "upload"),
                    str(original_filename or "")[:120],
                    result.image_hash,
                    int(result.original_width),
                    int(result.original_height),
                    int(result.processed_width),
                    int(result.processed_height),
                    str((result.settings or {}).get("mode") or "balanced"),
                    str((result.settings or {}).get("target_type") or "both"),
                    json.dumps(result.settings or {}, separators=(",", ":")),
                    counts["detected"],
                    counts["excluded"],
                    counts["manual"],
                    counts["final"],
                    float(result.processing_time_seconds),
                    str(notes or result.manual_notes or ""),
                    annotated_rel,
                    "saved",
                ),
            )
            run_id = int(cur.lastrowid)
            for det in result.detections:
                conn.execute(
                    """
                    INSERT INTO shape_detection_items (
                        run_id, sequence_number, center_x, center_y,
                        radius, diameter, partial, methods_json,
                        quality_score, included, review_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        int(det.sequence_number),
                        float(det.center_x),
                        float(det.center_y),
                        float(det.radius),
                        float(det.diameter),
                        1 if det.partial else 0,
                        json.dumps(list(det.detection_methods)),
                        float(det.quality_score),
                        1 if det.included else 0,
                        str(det.review_status or "unreviewed"),
                    ),
                )
            return run_id
    except DatabaseError as exc:
        raise ShapeStorageError(MSG_READ_ONLY) from exc
    except Exception as exc:  # noqa: BLE001
        raise ShapeStorageError(MSG_READ_ONLY) from exc


def list_shape_tests(
    user,
    *,
    limit: int = 100,
    owner_user_id: int | None = None,
    detection_mode: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_final_count: int | None = None,
    max_final_count: int | None = None,
    partial_present: bool | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    if user is None:
        return []
    initialize_database(db_path)
    sql = "SELECT * FROM shape_detection_runs WHERE 1=1"
    params: list[Any] = []

    if user.is_admin:
        if owner_user_id is not None:
            sql += " AND owner_user_id = ?"
            params.append(int(owner_user_id))
    else:
        sql += " AND owner_user_id = ?"
        params.append(int(user.user_id))

    if detection_mode:
        sql += " AND detection_mode = ?"
        params.append(str(detection_mode))
    if date_from:
        sql += " AND created_at >= ?"
        params.append(str(date_from))
    if date_to:
        sql += " AND created_at <= ?"
        params.append(str(date_to))
    if min_final_count is not None:
        sql += " AND final_count >= ?"
        params.append(int(min_final_count))
    if max_final_count is not None:
        sql += " AND final_count <= ?"
        params.append(int(max_final_count))

    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    with _connect(db_path) as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if partial_present is None:
        return rows

    # Filter by whether any included item is partial
    filtered: list[dict[str, Any]] = []
    for row in rows:
        items = get_shape_test_items(int(row["id"]), user, db_path=db_path)
        has_partial = any(i.get("partial") and i.get("included") for i in items)
        if has_partial == bool(partial_present):
            filtered.append(row)
    return filtered


def get_shape_test(
    run_id: int, user, *, db_path: str | None = None
) -> dict[str, Any] | None:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM shape_detection_runs WHERE id = ?",
            (int(run_id),),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    if not _can_view_run(user, data):
        raise ShapeAuthError("You do not have permission to open this shape test.")
    return data


def get_shape_test_items(
    run_id: int, user, *, db_path: str | None = None
) -> list[dict[str, Any]]:
    # Authorization via parent run
    get_shape_test(run_id, user, db_path=db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM shape_detection_items WHERE run_id = ? "
            "ORDER BY sequence_number ASC",
            (int(run_id),),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["methods"] = json.loads(item.get("methods_json") or "[]")
        except json.JSONDecodeError:
            item["methods"] = []
        item["partial"] = bool(item.get("partial"))
        item["included"] = bool(item.get("included"))
        out.append(item)
    return out


def load_annotated_image_bytes(
    run: dict[str, Any],
) -> bytes | None:
    rel = str(run.get("annotated_image_path") or "").strip()
    if not rel:
        return None
    # Only allow paths under DATA_DIR/shape_detection
    path = (Path(config.DATA_DIR) / rel).resolve()
    root = shape_artifacts_dir().resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def result_from_saved_run(
    run: dict[str, Any], items: list[dict[str, Any]]
) -> ShapeDetectionResult:
    """Rebuild a result model from saved rows without re-running detection."""
    detections: list[CircleDetection] = []
    for item in items:
        methods = item.get("methods") or []
        if isinstance(methods, str):
            try:
                methods = json.loads(methods)
            except json.JSONDecodeError:
                methods = []
        cx = float(item.get("center_x") or 0)
        cy = float(item.get("center_y") or 0)
        radius = float(item.get("radius") or 0)
        seq = int(item.get("sequence_number") or 0)
        from shape_detection_models import BoundingBox

        detections.append(
            CircleDetection(
                id=f"shape-{seq}",
                center_x=cx,
                center_y=cy,
                radius=radius,
                diameter=float(item.get("diameter") or radius * 2),
                bounding_box=BoundingBox(
                    x1=cx - radius,
                    y1=cy - radius,
                    x2=cx + radius,
                    y2=cy + radius,
                ),
                detection_methods=list(methods),
                quality_score=float(item.get("quality_score") or 0),
                partial=bool(item.get("partial")),
                included=bool(item.get("included")),
                review_status=str(item.get("review_status") or "unreviewed"),
                sequence_number=seq,
            )
        )
    try:
        settings = json.loads(run.get("parameter_summary_json") or "{}")
    except json.JSONDecodeError:
        settings = {}
    return ShapeDetectionResult(
        requested_shape=str(run.get("requested_shape") or ""),
        normalized_shape=str(run.get("normalized_shape") or "circle"),
        detections=detections,
        processing_time_seconds=float(run.get("processing_time") or 0),
        original_width=int(run.get("original_width") or 0),
        original_height=int(run.get("original_height") or 0),
        processed_width=int(run.get("processed_width") or 0),
        processed_height=int(run.get("processed_height") or 0),
        settings=settings,
        image_hash=str(run.get("image_hash") or ""),
        manually_added_count=int(run.get("manually_added_count") or 0),
        manual_notes=str(run.get("notes") or ""),
    )


def export_csv(run: dict[str, Any], items: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "run_id",
            "sequence_number",
            "center_x",
            "center_y",
            "radius",
            "diameter",
            "partial",
            "methods",
            "shape_quality",
            "included",
            "review_status",
        ]
    )
    for item in items:
        methods = item.get("methods") or []
        writer.writerow(
            [
                run.get("id"),
                item.get("sequence_number"),
                item.get("center_x"),
                item.get("center_y"),
                item.get("radius"),
                item.get("diameter"),
                bool(item.get("partial")),
                ";".join(methods) if isinstance(methods, list) else methods,
                item.get("quality_score"),
                bool(item.get("included")),
                item.get("review_status"),
            ]
        )
    return buf.getvalue()


def export_json(run: dict[str, Any], items: list[dict[str, Any]]) -> str:
    result = result_from_saved_run(run, items)
    payload = result.public_export_dict()
    payload["run_id"] = run.get("id")
    payload["created_at"] = run.get("created_at")
    payload["source_type"] = run.get("source_type")
    payload["original_filename_sanitized"] = run.get("original_filename_sanitized")
    # Never include storage paths or credentials
    return json.dumps(payload, indent=2)
