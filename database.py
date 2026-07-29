"""SQLite persistence for reviewed inventory counts."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from config import DB_PATH, ensure_data_dir


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


class DatabaseError(Exception):
    """Raised when a database operation fails."""


@contextmanager
def _connect(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    ensure_data_dir()
    path = db_path or str(DB_PATH)
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Database error: {exc}") from exc
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass


def initialize_database(db_path: str | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)


def insert_inventory_count(record: dict[str, Any], db_path: str | None = None) -> int:
    initialize_database(db_path)
    columns = [
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
    ]
    payload = dict(record)
    if not payload.get("created_at"):
        payload["created_at"] = datetime.now(timezone.utc).isoformat()

    values = [payload.get(col) for col in columns]
    placeholders = ", ".join("?" for _ in columns)
    col_sql = ", ".join(columns)
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
) -> list[dict[str, Any]]:
    initialize_database(db_path)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM inventory_counts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
    except DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError("Failed to load inventory history.") from exc


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
