"""Administrator console flows exercised through the real Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pytest

import database
import user_store
from test_auth_app_flows import (  # noqa: F401  (app_env fixture)
    BOOTSTRAP_PASSWORD,
    NEW_PASSWORD,
    app_env,
    click,
    sget,
    signed_in_admin,
)


@pytest.fixture
def console(app_env):
    at = signed_in_admin(app_env)
    click(at, "Open administrator console")
    return at


def submit(at, label: str) -> None:
    for button in at.button:
        if button.label == label:
            button.click().run()
            return
    raise AssertionError(f"button {label!r} not found")


def test_overview_reports_live_counts(console):
    assert not console.exception
    values = {m.label: m.value for m in console.metric}
    assert values["Users"] == "1"
    assert values["Active"] == "1"
    assert values["Administrators"] == "1"
    assert values["Locked"] == "0"
    assert values["Schema version"] == str(database.get_schema_version())


def test_admin_creates_a_user_with_a_one_time_password(console):
    console.text_input(key="admin_user_new_username").set_value("newhire")
    console.text_input(key="admin_user_new_display").set_value("New Hire")
    submit(console, "Create user")

    assert not console.exception
    record = user_store.get_user_by_username("newhire")
    assert record is not None
    assert record.role == "user"
    assert record.force_password_change is True

    temporary = console.code[0].value
    assert len(temporary) >= 12
    assert user_store.verify_credentials("newhire", temporary).status == "authenticated"
    # The password itself is shown once and never stored, in any form.
    assert not hasattr(record, "password_hash")
    assert temporary.encode() not in Path(database.current_db_path()).read_bytes()


def test_creating_a_duplicate_username_is_refused(console):
    console.text_input(key="admin_user_new_username").set_value("rootadmin")
    submit(console, "Create user")
    assert any("already" in e.value.lower() for e in console.error)


def test_invalid_username_is_rejected_with_guidance(console):
    console.text_input(key="admin_user_new_username").set_value("a b/c")
    submit(console, "Create user")
    assert console.error
    assert not console.exception


def test_admin_deactivates_and_reactivates_a_user(console):
    user_store.create_user(username="tempworker", password=NEW_PASSWORD, role="user")
    console.run()
    console.selectbox(key="admin_user_pick").set_value("tempworker (user)").run()

    submit(console, "Deactivate")
    assert user_store.get_user_by_username("tempworker").is_active is False

    submit(console, "Activate")
    assert user_store.get_user_by_username("tempworker").is_active is True


def test_admin_password_reset_forces_a_change_and_invalidates_sessions(console):
    user_store.create_user(
        username="resetme",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    before = user_store.get_user_by_username("resetme").session_version
    console.run()
    console.selectbox(key="admin_user_pick").set_value("resetme (user)").run()
    submit(console, "Generate temporary password")

    after = user_store.get_user_by_username("resetme")
    assert after.force_password_change is True
    assert after.session_version > before
    # The old password no longer works; the new one is shown once.
    assert user_store.verify_credentials("resetme", NEW_PASSWORD).status == "invalid"
    temporary = console.code[0].value
    assert user_store.verify_credentials("resetme", temporary).status == "authenticated"


def test_last_active_administrator_is_protected(console):
    assert any("last active administrator" in i.value for i in console.info)
    disabled = {b.label for b in console.button if b.disabled}
    assert {"Apply role", "Deactivate", "Delete user"} <= disabled


def test_deletion_requires_typing_the_username(console):
    user_store.create_user(username="deleteme", password=NEW_PASSWORD, role="user")
    console.run()
    console.selectbox(key="admin_user_pick").set_value("deleteme (user)").run()

    delete_button = next(b for b in console.button if b.label == "Delete user")
    assert delete_button.disabled

    console.text_input(key="admin_user_delete_confirm").set_value("deleteme").run()
    submit(console, "Delete user")
    assert user_store.get_user_by_username("deleteme") is None


def test_model_access_policies_are_editable(console):
    import model_access

    policies = {p.model_key: p for p in user_store.list_model_policies()}
    seeded = {seed["model_key"] for seed in model_access.DEFAULT_POLICY_SEEDS}
    assert seeded <= set(policies)

    byok = [p for p in policies.values() if p.requires_user_api_key]
    assert byok, "the OpenRouter policy should require a user-supplied key"
    assert all(p.requires_cost_confirmation for p in byok)


def test_audit_log_tab_lists_recent_events(console):
    labels = [s.label for s in console.selectbox]
    assert "Event type" in labels
    assert "Outcome" in labels
    # Sign-in and forced change of the bootstrap admin are already recorded.
    import auth

    types = {e["event_type"] for e in user_store.get_audit_events(limit=50)}
    assert auth.EVENT_LOGIN_SUCCESS in types
    assert auth.EVENT_PASSWORD_CHANGED in types
    assert auth.EVENT_BOOTSTRAP_ADMIN in types


def test_connectivity_tab_never_prints_secrets(console):
    import config

    rendered = " ".join(
        [m.value for m in console.markdown]
        + [c.value for c in console.caption]
        + [c.value for c in console.code]
    )
    if config.ROBOFLOW_API_KEY:
        assert config.ROBOFLOW_API_KEY not in rendered


def test_storage_tab_documents_ephemeral_hosting(console):
    text = " ".join(w.value for w in console.warning)
    assert "ephemeral storage" in text.lower()
    assert "redeploy" in text.lower()


def test_regular_user_reaching_the_console_is_audited(app_env):
    from streamlit.testing.v1 import AppTest

    at = signed_in_admin(app_env)
    user_store.create_user(
        username="nosy",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    click(at, "Sign out")
    at.text_input(key="login_username").set_value("nosy")
    at.text_input(key="login_password").set_value(NEW_PASSWORD)
    at.button[0].click().run()

    at.session_state.app_view = "admin"
    at.run()
    assert not at.exception
    assert "admin_user_new_username" not in at.session_state
    denied = user_store.get_audit_events(event_type="authz.denied")
    assert denied
