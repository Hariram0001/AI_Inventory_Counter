"""Offline tests for passwords, redaction, users, sessions and bootstrap."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import auth
import database
import security
import user_store
from auth import (
    AuthenticatedUser,
    LocalPasswordProvider,
    bootstrap_admin_if_needed,
    evaluate_session_expiry,
    to_authenticated_user,
)
from security import (
    MIN_PASSWORD_LENGTH,
    generate_temporary_password,
    hash_password,
    mask_secret,
    redact_secrets,
    redact_text,
    validate_email,
    validate_password_policy,
    validate_username,
    verify_password,
)

STRONG = "Str0ng!Passphrase42"


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "auth.db")
    database.apply_migrations(path)
    return path


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_migrations_are_idempotent_and_versioned(tmp_path):
    path = str(tmp_path / "m.db")
    assert database.apply_migrations(path) == database.SCHEMA_VERSION
    assert database.apply_migrations(path) == database.SCHEMA_VERSION
    assert database.get_schema_version(path) == database.SCHEMA_VERSION


def test_migration_adds_ownership_to_existing_database(tmp_path):
    """An existing single-table database upgrades without losing rows."""
    import sqlite3

    path = str(tmp_path / "legacy.db")
    with sqlite3.connect(path) as conn:
        conn.execute(database.CREATE_TABLE_SQL)
        conn.execute(
            "INSERT INTO inventory_counts (created_at, yard, inventory_type) "
            "VALUES ('2026-01-01T00:00:00+00:00', 'LA Yard', 'Fence Panel')"
        )

    database.apply_migrations(path)

    rows = database.get_inventory_history(db_path=path)
    assert len(rows) == 1
    assert rows[0]["yard"] == "LA Yard"
    assert rows[0]["user_id"] is None


def test_migration_backs_up_existing_database(tmp_path):
    import sqlite3

    path = tmp_path / "backup_me.db"
    with sqlite3.connect(str(path)) as conn:
        conn.execute(database.CREATE_TABLE_SQL)
        conn.execute(
            "INSERT INTO inventory_counts (created_at, yard, inventory_type) "
            "VALUES ('2026-01-01T00:00:00+00:00', 'Yard', 'Fence Panel')"
        )
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
            "description TEXT, applied_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO schema_migrations VALUES (1, 'inventory', '2026-01-01')"
        )

    database.apply_migrations(str(path))
    assert list(tmp_path.glob("backup_me.db.bak.*"))


# ---------------------------------------------------------------------------
# Password policy and hashing
# ---------------------------------------------------------------------------


def test_argon2_hash_roundtrip_and_uniqueness():
    first = hash_password(STRONG)
    second = hash_password(STRONG)
    assert first.startswith("$argon2id$")
    assert first != second  # per-password salt
    assert verify_password(first, STRONG)
    assert not verify_password(first, STRONG + "x")
    assert not verify_password("", STRONG)
    assert not verify_password("not-a-hash", STRONG)


def test_password_policy_rejects_short_and_placeholder():
    assert any("12 characters" in p for p in validate_password_policy("Ab1!x"))
    assert any("commonly used" in p for p in validate_password_policy("Password12345!"))
    assert any("commonly used" in p for p in validate_password_policy("ChangeMe1234!"))
    assert validate_password_policy(STRONG) == []


def test_password_policy_rejects_username_and_email_reuse():
    problems = validate_password_policy("alice!Alice12345", username="alice")
    assert any("username" in p for p in problems)
    problems = validate_password_policy(
        "Zx!bob@site.com99", email="bob@site.com"
    )
    assert any("email" in p for p in problems)


def test_password_policy_requires_character_variety():
    problems = validate_password_policy("abcdefghijklmnop")
    assert any("three of" in p for p in problems)


def test_generated_temporary_password_satisfies_policy():
    for _ in range(20):
        candidate = generate_temporary_password()
        assert len(candidate) >= MIN_PASSWORD_LENGTH
        assert validate_password_policy(candidate) == []


def test_username_and_email_validation():
    assert validate_username("  Admin.One  ") == "admin.one"
    with pytest.raises(ValueError):
        validate_username("ab")
    with pytest.raises(ValueError):
        validate_username("-starts-with-hyphen")
    assert validate_email("  USER@Example.COM ") == "user@example.com"
    assert validate_email("") == ""
    with pytest.raises(ValueError):
        validate_email("not-an-email")


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redaction_is_recursive_across_containers():
    payload = {
        "outer": {
            "api_key": "abc123secretvalue",
            "list": [{"model_api_key": "sk-or-v1-deadbeefcafe"}],
            "tuple": ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9",),
        },
        "safe": "hello",
    }
    cleaned = redact_secrets(payload)
    blob = repr(cleaned)
    assert "abc123secretvalue" not in blob
    assert "sk-or-v1-deadbeefcafe" not in blob
    assert "eyJhbGciOiJIUzI1NiJ9" not in blob
    assert cleaned["safe"] == "hello"


def test_redaction_keeps_metadata_field_values():
    """Field names that describe a secret are not themselves secrets."""
    payload = {
        "requires_user_api_key": True,
        "api_key_parameter_name": "model_api_key",
        "roboflow_key_configured": True,
    }
    assert redact_secrets(payload) == payload


def test_redact_text_handles_urls_and_headers():
    assert "zzz" not in redact_text("https://api.example.com/x?api_key=zzz9999&b=1")
    assert "eyJ" not in redact_text("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")
    assert "sk-or-v1-abcdef123456" not in redact_text("key sk-or-v1-abcdef123456 used")


def test_redaction_terminates_on_self_reference():
    node: dict = {"name": "root"}
    node["child"] = node
    assert redact_secrets(node) is not None  # bounded by max depth


def test_mask_secret_never_reveals_full_value():
    masked = mask_secret("sk-or-v1-abcdef1234567890")
    assert "abcdef1234567890" not in masked
    assert masked.startswith("sk-o")
    assert mask_secret("short") == "*****"
    assert mask_secret("") == ""


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------


def test_create_user_normalizes_and_rejects_duplicates(db):
    created = user_store.create_user(
        username="  Alice.B  ", password=STRONG, role="user", db_path=db
    )
    assert created.username == "alice.b"
    assert created.force_password_change is True
    with pytest.raises(user_store.UserStoreError):
        user_store.create_user(username="ALICE.B", password=STRONG, db_path=db)


def test_password_hash_is_never_returned_in_public_views(db):
    user = user_store.create_user(username="carol", password=STRONG, db_path=db)
    public = user.to_public_dict()
    assert "password_hash" not in public
    assert STRONG not in repr(user)


def test_lockout_after_five_failures_then_admin_unlock(db):
    user = user_store.create_user(username="dave", password=STRONG, db_path=db)
    for _ in range(user_store.MAX_FAILED_ATTEMPTS - 1):
        assert user_store.verify_credentials("dave", "wrong", db_path=db).status == "invalid"

    locked = user_store.verify_credentials("dave", "wrong", db_path=db)
    assert locked.status == "locked"
    assert locked.lock_seconds == user_store.LOCKOUT_MINUTES * 60

    # Correct credentials are still refused while the lock holds.
    assert user_store.verify_credentials("dave", STRONG, db_path=db).status == "locked"

    user_store.unlock_user(user.id, db_path=db)
    assert user_store.verify_credentials("dave", STRONG, db_path=db).status == "authenticated"


def test_deactivated_user_cannot_authenticate(db):
    user = user_store.create_user(username="erin", password=STRONG, db_path=db)
    user_store.create_user(username="root", password=STRONG, role="admin", db_path=db)
    user_store.set_user_active(user.id, False, db_path=db)
    assert user_store.verify_credentials("erin", STRONG, db_path=db).status == "disabled"


def test_password_change_bumps_session_version(db):
    user = user_store.create_user(username="frank", password=STRONG, db_path=db)
    updated = user_store.set_password(user.id, "An0ther!Passphrase42", db_path=db)
    assert updated.session_version == user.session_version + 1
    assert updated.force_password_change is False
    assert user_store.verify_credentials(
        "frank", "An0ther!Passphrase42", db_path=db
    ).status == "authenticated"


def test_last_active_administrator_is_protected(db):
    admin = user_store.create_user(
        username="onlyadmin", password=STRONG, role="admin", db_path=db
    )
    with pytest.raises(user_store.UserStoreError):
        user_store.set_user_active(admin.id, False, db_path=db)
    with pytest.raises(user_store.UserStoreError):
        user_store.set_user_role(admin.id, "user", db_path=db)
    with pytest.raises(user_store.UserStoreError):
        user_store.delete_user(admin.id, db_path=db)

    # A second administrator lifts the restriction.
    user_store.create_user(username="admin2", password=STRONG, role="admin", db_path=db)
    assert user_store.set_user_active(admin.id, False, db_path=db).is_active is False


def test_invalid_username_does_not_leak_existence(db):
    user_store.create_user(username="grace", password=STRONG, db_path=db)
    missing = user_store.verify_credentials("nobody", STRONG, db_path=db)
    wrong = user_store.verify_credentials("grace", "wrong", db_path=db)
    assert missing.message == wrong.message


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_audit_events_are_redacted_before_storage(db):
    user_store.record_audit_event(
        "test.event",
        actor_username="alice",
        detail={"api_key": "sk-or-v1-supersecretvalue", "note": "ok"},
        db_path=db,
    )
    events = user_store.get_audit_events(db_path=db)
    assert len(events) == 1
    assert "supersecretvalue" not in str(events[0]["detail"])
    assert "ok" in str(events[0]["detail"])


def test_audit_logging_never_raises_on_bad_input(db):
    class Unserializable:
        def __repr__(self):
            raise RuntimeError("boom")

    assert user_store.record_audit_event(
        "test.event", detail={"x": Unserializable()}, db_path=db
    ) >= 0


def test_audit_filters(db):
    user_store.record_audit_event("a.b", actor_username="x", db_path=db)
    user_store.record_audit_event("c.d", actor_username="y", outcome="failure", db_path=db)
    assert len(user_store.get_audit_events(event_type="a.b", db_path=db)) == 1
    assert len(user_store.get_audit_events(outcome="failure", db_path=db)) == 1
    assert set(user_store.list_audit_event_types(db_path=db)) == {"a.b", "c.d"}


# ---------------------------------------------------------------------------
# Provider and sessions
# ---------------------------------------------------------------------------


def test_local_provider_outcomes(db):
    user_store.create_user(username="henry", password=STRONG, db_path=db)
    provider = LocalPasswordProvider(db)

    assert provider.authenticate("", "").status == "invalid"
    assert provider.authenticate("henry", "wrong").status == "invalid"

    outcome = provider.authenticate("henry", STRONG)
    assert outcome.ok
    assert outcome.user.username == "henry"
    assert outcome.user.auth_provider == "local"


def test_provider_revalidate_rejects_stale_session_version(db):
    record = user_store.create_user(username="ivy", password=STRONG, db_path=db)
    provider = LocalPasswordProvider(db)
    live = to_authenticated_user(record)

    assert provider.revalidate(live) is not None

    user_store.set_password(record.id, "An0ther!Passphrase42", db_path=db)
    assert provider.revalidate(live) is None


def test_provider_revalidate_rejects_deactivated_user(db):
    record = user_store.create_user(username="jack", password=STRONG, db_path=db)
    user_store.create_user(username="root2", password=STRONG, role="admin", db_path=db)
    provider = LocalPasswordProvider(db)
    live = to_authenticated_user(record)

    user_store.set_user_active(record.id, False, db_path=db)
    assert provider.revalidate(live) is None


def test_session_expiry_idle_and_absolute():
    now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)

    fresh = evaluate_session_expiry(
        authenticated_at=now - timedelta(minutes=5),
        last_activity_at=now - timedelta(minutes=1),
        now=now,
        idle_minutes=30,
        absolute_hours=12,
    )
    assert fresh.expired is False

    idle = evaluate_session_expiry(
        authenticated_at=now - timedelta(hours=1),
        last_activity_at=now - timedelta(minutes=31),
        now=now,
        idle_minutes=30,
        absolute_hours=12,
    )
    assert idle.expired and idle.reason == "idle"
    assert "inactivity" in idle.message

    absolute = evaluate_session_expiry(
        authenticated_at=now - timedelta(hours=13),
        last_activity_at=now,
        now=now,
        idle_minutes=30,
        absolute_hours=12,
    )
    assert absolute.expired and absolute.reason == "absolute"

    missing = evaluate_session_expiry(
        authenticated_at=None, last_activity_at=None, now=now
    )
    assert missing.expired and missing.reason == "revoked"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_creates_first_admin_then_skips(db, monkeypatch):
    import config

    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_USERNAME", "bootadmin")
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_PASSWORD", STRONG)
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_EMAIL", "boot@example.com")

    result = bootstrap_admin_if_needed(db)
    assert result.created
    created = user_store.get_user_by_username("bootadmin", db)
    assert created.is_admin
    assert created.force_password_change is True

    assert bootstrap_admin_if_needed(db).status == "skipped"


def test_bootstrap_reports_missing_configuration(db, monkeypatch):
    import config

    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_USERNAME", "")
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_PASSWORD", "")
    monkeypatch.delenv("BOOTSTRAP_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD", raising=False)

    result = bootstrap_admin_if_needed(db)
    assert result.status == "misconfigured"
    assert "BOOTSTRAP_ADMIN_USERNAME" in result.message


def test_bootstrap_rejects_weak_password(db, monkeypatch):
    import config

    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_USERNAME", "bootadmin")
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_PASSWORD", "changeme")
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_EMAIL", "")

    result = bootstrap_admin_if_needed(db)
    assert result.status == "misconfigured"
    assert "password policy" in result.message
    assert user_store.count_users(db) == 0


def test_bootstrap_password_is_not_written_to_audit(db, monkeypatch):
    import config

    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_USERNAME", "bootadmin")
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_PASSWORD", STRONG)
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_EMAIL", "")

    bootstrap_admin_if_needed(db)
    events = user_store.get_audit_events(db_path=db)
    assert events
    assert STRONG not in str(events)


# ---------------------------------------------------------------------------
# Per-user history ownership
# ---------------------------------------------------------------------------


def test_history_is_scoped_by_owner(db):
    alice = user_store.create_user(username="alice", password=STRONG, db_path=db)
    bob = user_store.create_user(username="bob", password=STRONG, db_path=db)

    database.insert_inventory_count(
        {"yard": "Y", "inventory_type": "Fence Panel", "user_id": alice.id, "username": "alice"},
        db_path=db,
    )
    database.insert_inventory_count(
        {"yard": "Y", "inventory_type": "Fence Panel", "user_id": bob.id, "username": "bob"},
        db_path=db,
    )
    database.insert_inventory_count(
        {"yard": "Y", "inventory_type": "Fence Panel"}, db_path=db
    )

    alice_rows = database.get_inventory_history(db_path=db, user_id=alice.id)
    assert [r["username"] for r in alice_rows] == ["alice"]

    with_legacy = database.get_inventory_history(
        db_path=db, user_id=alice.id, include_legacy=True
    )
    assert len(with_legacy) == 2

    assert len(database.get_inventory_history(db_path=db)) == 3
