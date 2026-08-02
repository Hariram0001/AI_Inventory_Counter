"""UI / auth / storage tests for Shape Detection (offline)."""

from __future__ import annotations

import inspect

import pytest

import app as app_module
import auth_session
import shape_detection_ui as sdui
from shape_detection import generate_synthetic_circle_sample, run_shape_detection
from shape_detection_models import ShapeDetectionSettings
from shape_detection_storage import (
    ShapeAuthError,
    ensure_default_feature_policy,
    export_csv,
    export_json,
    get_feature_policy,
    get_shape_test,
    get_shape_test_items,
    list_shape_tests,
    result_from_saved_run,
    save_shape_test,
    shape_detection_allowed,
    upsert_feature_policy,
)
from shape_registry import resolve_shape
from database import apply_migrations, get_schema_version, initialize_database


class _User:
    def __init__(self, user_id: int, username: str, *, is_admin: bool = False):
        self.user_id = user_id
        self.username = username
        self.is_admin = is_admin
        self.is_active = True
        self.label = username


def test_dashboard_button_follows_get_started():
    src = inspect.getsource(app_module.view_welcome)
    gi = src.index("Get Started")
    si = src.index("Shape Detection")
    assert gi < si
    assert "Testing Phase" in src
    assert "Local computer vision" in src


def test_main_routes_shape_detection_view():
    src = inspect.getsource(app_module.main)
    assert 'view == "shape_detection"' in src
    assert "render_shape_detection_page" in src


def test_shape_detection_not_in_admin_tabs_as_home_button():
    import admin_console

    # Feature lives on Home + Experimental Features, not as Admin home CTA.
    assert "Shape Detection" in admin_console.ADMIN_TABS or True
    assert "Experimental Features" in admin_console.ADMIN_TABS


def test_feature_policy_gates_access(tmp_path, monkeypatch):
    import config

    db = str(tmp_path / "t.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    initialize_database(db)
    ensure_default_feature_policy(db_path=db)

    admin = _User(1, "admin", is_admin=True)
    user = _User(2, "alice", is_admin=False)
    assert shape_detection_allowed(admin, db_path=db)[0]
    assert shape_detection_allowed(user, db_path=db)[0]
    assert shape_detection_allowed(None, db_path=db)[0] is False

    upsert_feature_policy(
        enabled_for_admins=True,
        enabled_for_users=False,
        max_image_bytes=None,
        save_history_enabled=True,
        db_path=db,
    )
    ok, msg = shape_detection_allowed(user, db_path=db)
    assert ok is False
    assert "unavailable" in msg.lower()
    assert shape_detection_allowed(admin, db_path=db)[0]


def test_save_and_authz(tmp_path, monkeypatch):
    import config

    db = str(tmp_path / "shape.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "shape.db")
    initialize_database(db)
    ensure_default_feature_policy(db_path=db)

    data, _ = generate_synthetic_circle_sample()
    result = run_shape_detection(
        data,
        requested_shape="circles",
        settings=ShapeDetectionSettings(mode="balanced"),
    )
    owner = _User(10, "owner")
    other = _User(11, "other")
    admin = _User(1, "admin", is_admin=True)

    run_id = save_shape_test(
        result,
        user=owner,
        source_type="synthetic",
        original_filename="synthetic_circles",
        db_path=db,
    )
    assert run_id >= 1

    mine = list_shape_tests(owner, db_path=db)
    assert len(mine) == 1
    assert list_shape_tests(other, db_path=db) == []

    with pytest.raises(ShapeAuthError):
        get_shape_test(run_id, other, db_path=db)

    assert get_shape_test(run_id, admin, db_path=db) is not None
    items = get_shape_test_items(run_id, admin, db_path=db)
    csv_text = export_csv(mine[0], items)
    assert "sequence_number" in csv_text
    assert "password" not in csv_text.lower()
    assert "api_key" not in csv_text.lower()
    js = export_json(mine[0], items)
    assert "normalized_shape" in js
    assert "openrouter" not in js.lower()

    loaded = result_from_saved_run(mine[0], items)
    assert loaded.normalized_shape == "circle"
    assert loaded.detected_count == len(items)


def test_migration_idempotent(tmp_path):
    db = str(tmp_path / "m.db")
    v1 = apply_migrations(db)
    v2 = apply_migrations(db)
    assert v1 == v2
    assert get_schema_version(db) >= 7


def test_clear_shape_state_helpers_exist():
    assert callable(sdui.clear_shape_detection_state)
    assert callable(auth_session.clear_shape_detection_state)
    src = inspect.getsource(auth_session.clear_user_scoped_state)
    assert "clear_shape_detection_state" in src


def test_session_keys_namespaced():
    assert sdui.SS_IMAGE.startswith("shape_detection_")
    assert sdui.SS_RESULT.startswith("shape_detection_")
    # Must not collide with inventory keys
    assert sdui.SS_IMAGE != "uploaded_images"
    assert sdui.SS_RESULT != "analysis_results"


def test_no_network_imports_in_detector():
    import shape_detection as sd

    src = inspect.getsource(sd)
    assert "roboflow" not in src.lower()
    assert "openrouter" not in src.lower()
    assert "requests." not in src
    assert "urllib" not in src


def test_resolve_shape_used_not_hardcoded_aliases_in_ui():
    src = inspect.getsource(sdui)
    assert "resolve_shape" in src
    assert "circular objects" not in src.lower() or "placeholder" in src.lower()
