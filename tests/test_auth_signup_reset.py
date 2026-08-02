"""Self-signup approval and admin-authorized password reset requests."""

from __future__ import annotations

import database
from auth import LocalPasswordProvider
from user_store import (
    ACCOUNT_STATUS_PENDING,
    RESET_STATUS_FULFILLED,
    RESET_STATUS_PENDING,
    RESET_STATUS_REJECTED,
    approve_pending_signup,
    create_pending_signup,
    create_user,
    get_user_by_username,
    list_password_reset_requests,
    list_pending_signups,
    reject_pending_signup,
    request_password_reset,
    resolve_password_reset_request,
    set_password,
    verify_credentials,
)


def test_migration_adds_account_status_and_reset_table(tmp_path):
    db = tmp_path / "auth.db"
    database.initialize_database(str(db))
    assert database.get_schema_version(str(db)) >= 8
    with database._connect(str(db)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        assert "account_status" in cols
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "password_reset_requests" in tables


def test_pending_signup_cannot_sign_in_until_approved(tmp_path):
    db = str(tmp_path / "s.db")
    database.initialize_database(db)
    created = create_pending_signup(
        username="yardtech",
        password="secret1",
        display_name="Yard Tech",
        db_path=db,
    )
    assert created.account_status == ACCOUNT_STATUS_PENDING
    assert not created.is_active
    assert list_pending_signups(db_path=db)[0].username == "yardtech"

    result = verify_credentials("yardtech", "secret1", db_path=db)
    assert result.status == "pending"

    outcome = LocalPasswordProvider(db_path=db).authenticate("yardtech", "secret1")
    assert not outcome.ok
    assert "approve" in (outcome.message or "").lower()

    approved = approve_pending_signup(created.id, db_path=db)
    assert approved is not None
    assert approved.is_active
    assert approved.account_status == "active"
    assert verify_credentials("yardtech", "secret1", db_path=db).status == "authenticated"


def test_reject_pending_signup_deletes_account(tmp_path):
    db = str(tmp_path / "r.db")
    database.initialize_database(db)
    created = create_pending_signup(
        username="rejectme", password="x", db_path=db
    )
    reject_pending_signup(created.id, db_path=db)
    assert get_user_by_username("rejectme", db_path=db) is None
    assert list_pending_signups(db_path=db) == []


def test_password_reset_request_is_generic_and_admin_resolvable(tmp_path):
    db = str(tmp_path / "p.db")
    database.initialize_database(db)
    create_user(
        username="worker",
        password="oldpass",
        force_password_change=False,
        db_path=db,
    )

    ok, msg = request_password_reset("worker", db_path=db)
    assert ok
    assert "administrator" in msg.lower()

    # Unknown usernames get the same style of success (no enumeration).
    ok2, msg2 = request_password_reset("nosuchuser", db_path=db)
    assert ok2
    assert msg2 == msg

    pending = list_password_reset_requests(db_path=db)
    assert len(pending) == 1
    assert pending[0].username == "worker"
    assert pending[0].status == RESET_STATUS_PENDING

    # Duplicate request while pending does not create a second row.
    request_password_reset("worker", db_path=db)
    assert len(list_password_reset_requests(db_path=db)) == 1

    set_password(pending[0].user_id, "temp-pass", force_password_change=True, db_path=db)
    resolve_password_reset_request(
        pending[0].id,
        status=RESET_STATUS_FULFILLED,
        reviewed_by="admin",
        db_path=db,
    )
    assert list_password_reset_requests(db_path=db) == []
    user = get_user_by_username("worker", db_path=db)
    assert user is not None
    assert user.force_password_change
    assert verify_credentials("worker", "temp-pass", db_path=db).status == "authenticated"


def test_reject_password_reset_leaves_password(tmp_path):
    db = str(tmp_path / "q.db")
    database.initialize_database(db)
    create_user(
        username="keeper",
        password="keepme",
        force_password_change=False,
        db_path=db,
    )
    request_password_reset("keeper", db_path=db)
    req = list_password_reset_requests(db_path=db)[0]
    resolve_password_reset_request(
        req.id,
        status=RESET_STATUS_REJECTED,
        reviewed_by="admin",
        db_path=db,
    )
    assert verify_credentials("keeper", "keepme", db_path=db).status == "authenticated"
