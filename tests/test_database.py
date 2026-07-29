"""Tests for SQLite inventory history."""

from __future__ import annotations

from database import (
    compute_percentage_error,
    compute_reviewed_count,
    get_inventory_history,
    initialize_database,
    insert_inventory_count,
)


def test_initialize_insert_retrieve(tmp_path):
    db = str(tmp_path / "test.db")
    initialize_database(db)
    row_id = insert_inventory_count(
        {
            "yard": "LA Yard",
            "inventory_type": "Fence Panel",
            "photo_relationship": "Separate inventory areas",
            "number_of_photos": 1,
            "selected_mode": "Single model",
            "accepted_model": "Demo Fence Detector",
            "selected_prompt": "individual temporary metal fence panel",
            "inference_mode": "Whole-image inference",
            "tile_size": 800,
            "tile_overlap": 0.25,
            "deduplication_strategy": "Conservative",
            "confidence_threshold": 0.4,
            "iou_threshold": 0.5,
            "raw_ai_count": 12,
            "ai_count": 10,
            "reviewed_count": 9,
            "false_positive_adjustment": 1,
            "missed_item_adjustment": 0,
            "average_confidence": 0.82,
            "suspected_overlap_count": 2,
            "suspected_occlusion_count": 1,
            "processing_time_seconds": 1.23,
            "percentage_error": compute_percentage_error(10, 9),
            "notes": "spot-checked",
        },
        db_path=db,
    )
    assert row_id >= 1
    rows = get_inventory_history(db_path=db)
    assert len(rows) == 1
    assert rows[0]["yard"] == "LA Yard"
    assert rows[0]["reviewed_count"] == 9
    assert rows[0]["ai_count"] == 10


def test_reviewed_count_helpers():
    assert compute_reviewed_count(10, false_positive_adjustment=2, missed_item_adjustment=1) == 9
    assert compute_reviewed_count(10, direct_reviewed_count=7) == 7
    assert compute_reviewed_count(1, false_positive_adjustment=5) == 0
    assert compute_percentage_error(10, 8) == 25.0
    assert compute_percentage_error(5, 0) is None
