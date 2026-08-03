"""SQLite persistence for inventory counts, users, policies and audit events.

Schema changes go through the versioned migration list at the bottom of this
module. Migrations are idempotent, run inside a transaction, and take a file
backup before the first upgrade of an existing database.
"""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple

import config
from config import ensure_data_dir


def current_db_path() -> str:
    """Resolve the database path on every call.

    ``config.reload_settings()`` can point DATA_DIR somewhere new at runtime, so
    binding the path at import time would silently keep writing to the old file.
    """
    return str(config.DB_PATH)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inventory_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    yard TEXT NOT NULL,
    inventory_type TEXT NOT NULL,
    photo_relationship TEXT,
    number_of_photos INTEGER,
    selected_mode TEXT,
    accepted_model TEXT,
    selected_prompt TEXT,
    inference_mode TEXT,
    tile_size INTEGER,
    tile_overlap REAL,
    deduplication_strategy TEXT,
    confidence_threshold REAL,
    iou_threshold REAL,
    raw_ai_count INTEGER,
    ai_count INTEGER,
    reviewed_count INTEGER,
    false_positive_adjustment INTEGER,
    missed_item_adjustment INTEGER,
    average_confidence REAL,
    suspected_overlap_count INTEGER,
    suspected_occlusion_count INTEGER,
    processing_time_seconds REAL,
    percentage_error REAL,
    notes TEXT
);
"""

INVENTORY_COLUMNS = [
    "created_at",
    "yard",
    "inventory_type",
    "photo_relationship",
    "number_of_photos",
    "selected_mode",
    "accepted_model",
    "selected_prompt",
    "inference_mode",
    "tile_size",
    "tile_overlap",
    "deduplication_strategy",
    "confidence_threshold",
    "iou_threshold",
    "raw_ai_count",
    "ai_count",
    "reviewed_count",
    "false_positive_adjustment",
    "missed_item_adjustment",
    "average_confidence",
    "suspected_overlap_count",
    "suspected_occlusion_count",
    "processing_time_seconds",
    "percentage_error",
    "notes",
    "user_id",
    "username",
]


class DatabaseError(Exception):
    """Raised when a database operation fails."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    ensure_data_dir()
    path = db_path or current_db_path()
    conn: sqlite3.Connection | None = None
    try:
        # timeout + WAL let multiple signed-in users share one SQLite file
        # without serializing the whole app behind a single lock.
        conn = sqlite3.connect(path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error:
            # Some filesystems reject WAL; continue with the default journal.
            pass
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
        raise DatabaseError(f"Database error: {exc}") from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Migration framework
# ---------------------------------------------------------------------------


class Migration(NamedTuple):
    version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, ddl: str
) -> None:
    if column not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _current_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT,
            applied_at TEXT NOT NULL
        )
        """
    )
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    value = row[0] if row else None
    return int(value) if value is not None else 0


def _migration_001_inventory_counts(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_TABLE_SQL)


def _migration_002_auth_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT,
            display_name TEXT,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            is_active INTEGER NOT NULL DEFAULT 1,
            force_password_change INTEGER NOT NULL DEFAULT 0,
            failed_login_count INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            last_login_at TEXT,
            last_activity_at TEXT,
            password_changed_at TEXT,
            session_version INTEGER NOT NULL DEFAULT 1,
            auth_provider TEXT NOT NULL DEFAULT 'local',
            external_subject TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_by TEXT
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            actor_user_id INTEGER,
            actor_username TEXT,
            event_type TEXT NOT NULL,
            target_type TEXT,
            target_id TEXT,
            outcome TEXT NOT NULL DEFAULT 'success',
            detail TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(event_type)"
    )


def _migration_003_inventory_ownership(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "inventory_counts", "user_id", "INTEGER")
    _add_column_if_missing(conn, "inventory_counts", "username", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inventory_user ON inventory_counts(user_id)"
    )


def _migration_004_model_policies(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_access_policies (
            model_key TEXT PRIMARY KEY,
            display_name TEXT,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            allowed_roles TEXT NOT NULL DEFAULT 'admin,user',
            requires_user_api_key INTEGER NOT NULL DEFAULT 0,
            requires_cost_confirmation INTEGER NOT NULL DEFAULT 0,
            maximum_runs_per_user_per_day INTEGER,
            notes TEXT,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_key TEXT NOT NULL,
            usage_date TEXT NOT NULL,
            run_count INTEGER NOT NULL DEFAULT 0,
            last_run_at TEXT,
            UNIQUE(user_id, model_key, usage_date)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_usage_user_date "
        "ON user_model_usage(user_id, usage_date)"
    )


def _migration_005_admin_samples(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            title TEXT,
            description TEXT,
            inventory_type TEXT,
            expected_count INTEGER,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            width INTEGER,
            height INTEGER,
            size_bytes INTEGER,
            mime_type TEXT,
            uploaded_by TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _migration_006_deployment_secrets(conn: sqlite3.Connection) -> None:
    """Admin-managed OpenRouter key shared by the deployment (not per user)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deployment_secrets (
            name TEXT PRIMARY KEY,
            secret_value TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            updated_by TEXT NOT NULL DEFAULT ''
        )
        """
    )
    # OpenRouter is now admin-supplied; drop per-user BYOK / cost flags on seeds.
    conn.execute(
        """
        UPDATE model_access_policies
        SET requires_user_api_key = 0,
            requires_cost_confirmation = 0,
            notes = CASE
                WHEN model_key LIKE '%playground-gpt%' OR model_key LIKE '%openrouter%'
                THEN 'Uses the administrator-configured OpenRouter key. Enable the '
                     || 'model here to let users run it; they never see the key.'
                ELSE notes
            END
        WHERE requires_user_api_key = 1
           OR model_key LIKE '%playground-gpt%'
           OR model_key LIKE '%openrouter%'
        """
    )


def _migration_007_shape_detection(conn: sqlite3.Connection) -> None:
    """Experimental local shape detection history + feature policy."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_policies (
            feature_key TEXT PRIMARY KEY,
            display_name TEXT,
            enabled_for_admins INTEGER NOT NULL DEFAULT 1,
            enabled_for_users INTEGER NOT NULL DEFAULT 1,
            max_image_bytes INTEGER,
            save_history_enabled INTEGER NOT NULL DEFAULT 1,
            notes TEXT,
            updated_at TEXT NOT NULL,
            updated_by TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shape_detection_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_user_id INTEGER,
            owner_username_snapshot TEXT,
            created_at TEXT NOT NULL,
            requested_shape TEXT NOT NULL,
            normalized_shape TEXT NOT NULL,
            source_type TEXT,
            original_filename_sanitized TEXT,
            image_hash TEXT,
            original_width INTEGER,
            original_height INTEGER,
            processed_width INTEGER,
            processed_height INTEGER,
            detection_mode TEXT,
            target_type TEXT,
            parameter_summary_json TEXT,
            detected_count INTEGER,
            excluded_count INTEGER,
            manually_added_count INTEGER,
            final_count INTEGER,
            processing_time REAL,
            notes TEXT,
            annotated_image_path TEXT,
            status TEXT NOT NULL DEFAULT 'saved'
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_shape_runs_owner "
        "ON shape_detection_runs(owner_user_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_shape_runs_created "
        "ON shape_detection_runs(created_at DESC)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shape_detection_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            sequence_number INTEGER NOT NULL,
            center_x REAL,
            center_y REAL,
            radius REAL,
            diameter REAL,
            partial INTEGER NOT NULL DEFAULT 0,
            methods_json TEXT,
            quality_score REAL,
            included INTEGER NOT NULL DEFAULT 1,
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            FOREIGN KEY(run_id) REFERENCES shape_detection_runs(id)
                ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_shape_items_run "
        "ON shape_detection_items(run_id)"
    )
    # Extend admin samples for optional Shape Detection classification.
    _add_column_if_missing(
        conn, "admin_samples", "sample_kind", "TEXT NOT NULL DEFAULT 'inventory'"
    )
    _add_column_if_missing(conn, "admin_samples", "expected_shape", "TEXT")
    _add_column_if_missing(conn, "admin_samples", "verified_count", "INTEGER")
    _add_column_if_missing(conn, "admin_samples", "difficulty", "TEXT")


def _migration_008_signup_and_reset_requests(conn: sqlite3.Connection) -> None:
    """Self-signup pending approval + admin-authorized password reset requests."""
    _add_column_if_missing(
        conn, "users", "account_status", "TEXT NOT NULL DEFAULT 'active'"
    )
    # Existing inactive accounts become disabled (not pending self-signups).
    conn.execute(
        """
        UPDATE users
        SET account_status = 'disabled'
        WHERE COALESCE(is_active, 1) = 0
          AND account_status = 'active'
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            user_id INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            requested_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewed_by TEXT NOT NULL DEFAULT '',
            detail TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_password_reset_status "
        "ON password_reset_requests(status, requested_at DESC)"
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "inventory_counts table", _migration_001_inventory_counts),
    Migration(2, "users and audit_events tables", _migration_002_auth_tables),
    Migration(3, "inventory ownership columns", _migration_003_inventory_ownership),
    Migration(4, "model access policies and usage", _migration_004_model_policies),
    Migration(5, "admin managed samples", _migration_005_admin_samples),
    Migration(6, "admin OpenRouter deployment key", _migration_006_deployment_secrets),
    Migration(7, "shape detection and feature policies", _migration_007_shape_detection),
    Migration(8, "signup approval and password reset requests", _migration_008_signup_and_reset_requests),
)

SCHEMA_VERSION = MIGRATIONS[-1].version


def _backup_database(path: str) -> str | None:
    """Copy the database file next to itself before the first schema upgrade."""
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = source.with_name(f"{source.name}.bak.{stamp}")
    try:
        shutil.copy2(source, target)
        return str(target)
    except Exception:  # noqa: BLE001 — a failed backup must not block startup
        return None


def apply_migrations(db_path: str | None = None) -> int:
    """Bring the database up to ``SCHEMA_VERSION``; returns the applied version."""
    ensure_data_dir()
    path = db_path or current_db_path()

    with _connect(path) as conn:
        version = _current_version(conn)
        pending = [m for m in MIGRATIONS if m.version > version]
        if not pending:
            return version

    if version > 0:
        _backup_database(path)

    with _connect(path) as conn:
        conn.execute("BEGIN")
        try:
            for migration in pending:
                migration.apply(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_migrations "
                    "(version, description, applied_at) VALUES (?, ?, ?)",
                    (migration.version, migration.description, utc_now_iso()),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return _current_version(conn)


def get_schema_version(db_path: str | None = None) -> int:
    with _connect(db_path) as conn:
        return _current_version(conn)


def initialize_database(db_path: str | None = None) -> None:
    apply_migrations(db_path)


# ---------------------------------------------------------------------------
# Inventory counts
# ---------------------------------------------------------------------------


def insert_inventory_count(record: dict[str, Any], db_path: str | None = None) -> int:
    initialize_database(db_path)
    payload = dict(record)
    if not payload.get("created_at"):
        payload["created_at"] = utc_now_iso()

    values = [payload.get(col) for col in INVENTORY_COLUMNS]
    placeholders = ", ".join("?" for _ in INVENTORY_COLUMNS)
    col_sql = ", ".join(INVENTORY_COLUMNS)
    sql = f"INSERT INTO inventory_counts ({col_sql}) VALUES ({placeholders})"

    try:
        with _connect(db_path) as conn:
            cur = conn.execute(sql, values)
            return int(cur.lastrowid)
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to save inventory count.") from exc


def get_inventory_history(
    limit: int = 200,
    db_path: str | None = None,
    *,
    user_id: int | None = None,
    include_legacy: bool = False,
) -> list[dict[str, Any]]:
    """Return history rows, newest first.

    ``user_id`` restricts results to one owner. ``include_legacy`` additionally
    returns pre-authentication rows that have no owner recorded.
    """
    initialize_database(db_path)
    sql = "SELECT * FROM inventory_counts"
    params: list[Any] = []
    if user_id is not None:
        if include_legacy:
            sql += " WHERE user_id = ? OR user_id IS NULL"
        else:
            sql += " WHERE user_id = ?"
        params.append(int(user_id))
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    try:
        with _connect(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to load inventory history.") from exc


def count_inventory_rows(
    db_path: str | None = None, *, user_id: int | None = None
) -> int:
    initialize_database(db_path)
    sql = "SELECT COUNT(*) FROM inventory_counts"
    params: list[Any] = []
    if user_id is not None:
        sql += " WHERE user_id = ?"
        params.append(int(user_id))
    with _connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
        return int(row[0] if row else 0)


def compute_percentage_error(ai_count: int, reviewed_count: int) -> float | None:
    """Absolute percentage error; None when reviewed_count is zero."""
    try:
        ai = int(ai_count)
        reviewed = int(reviewed_count)
    except (TypeError, ValueError):
        return None
    if reviewed == 0:
        return None
    return abs(ai - reviewed) / reviewed * 100.0


def compute_reviewed_count(
    ai_count: int,
    false_positive_adjustment: int = 0,
    missed_item_adjustment: int = 0,
    direct_reviewed_count: int | None = None,
) -> int:
    if direct_reviewed_count is not None:
        return max(0, int(direct_reviewed_count))
    return max(0, int(ai_count) - int(false_positive_adjustment) + int(missed_item_adjustment))
