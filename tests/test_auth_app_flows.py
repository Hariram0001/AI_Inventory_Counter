"""End-to-end Streamlit flows: login gate, forced change, roles, isolation."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import auth_session
import database
import user_store
from auth import to_authenticated_user

BOOTSTRAP_PASSWORD = "Str0ng!Bootstrap#2026"
NEW_PASSWORD = "An0ther!Passphrase#2026"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Run the real app against a throwaway data directory."""
    import config

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BOOTSTRAP_ADMIN_USERNAME", "rootadmin")
    monkeypatch.setenv("BOOTSTRAP_ADMIN_PASSWORD", BOOTSTRAP_PASSWORD)
    monkeypatch.setenv("BOOTSTRAP_ADMIN_EMAIL", "root@example.com")
    config.reload_settings()
    yield config
    config.reload_settings()


def run_app() -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    return at


def sget(at: AppTest, key: str, default=None):
    """AppTest session state has no .get(), so emulate it."""
    return at.session_state[key] if key in at.session_state else default


def click(at: AppTest, label: str) -> bool:
    for button in at.button:
        if button.label == label:
            button.click().run()
            return True
    return False


def sign_in(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input(key="login_username").set_value(username)
    at.text_input(key="login_password").set_value(password)
    at.button[0].click().run()
    return at


def change_password(at: AppTest, current: str, new: str) -> AppTest:
    at.text_input(key="pwchange_current").set_value(current)
    at.text_input(key="pwchange_new").set_value(new)
    at.text_input(key="pwchange_confirm").set_value(new)
    at.button[0].click().run()
    return at


def signed_in_admin(app_env) -> AppTest:
    at = run_app()
    sign_in(at, "rootadmin", BOOTSTRAP_PASSWORD)
    change_password(at, BOOTSTRAP_PASSWORD, NEW_PASSWORD)
    return at


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------


def test_app_opens_on_login_screen(app_env):
    at = run_app()
    assert not at.exception
    assert [i.label for i in at.text_input] == ["Username", "Password"]
    assert [b.label for b in at.button] == ["Sign in"]


def test_no_dashboard_content_leaks_before_sign_in(app_env):
    at = run_app()
    labels = {b.label for b in at.button}
    assert "Get Started" not in labels
    assert "Open administrator console" not in labels
    assert "Sign out" not in labels


def test_bootstrap_creates_first_admin_and_reports_it(app_env):
    at = run_app()
    assert any("rootadmin" in s.value for s in at.success)
    assert user_store.get_user_by_username("rootadmin") is not None


def test_missing_bootstrap_configuration_warns_instead_of_crashing(
    tmp_path, monkeypatch
):
    import config

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BOOTSTRAP_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_USERNAME", "")
    monkeypatch.setattr(config, "BOOTSTRAP_ADMIN_PASSWORD", "")
    config.reload_settings()

    at = run_app()
    assert not at.exception
    assert any("BOOTSTRAP_ADMIN_USERNAME" in w.value for w in at.warning)
    config.reload_settings()


def test_invalid_credentials_show_a_generic_error(app_env):
    at = run_app()
    sign_in(at, "rootadmin", "definitely-wrong")
    assert not at.exception
    assert any("Invalid username or password" in e.value for e in at.error)
    # Still on the login screen.
    assert at.text_input(key="login_username") is not None


def test_password_is_not_retained_after_a_failed_attempt(app_env):
    at = run_app()
    sign_in(at, "rootadmin", "definitely-wrong")
    assert sget(at, "login_password", "") == ""


def test_repeated_failures_lock_the_account(app_env):
    at = run_app()
    for _ in range(user_store.MAX_FAILED_ATTEMPTS):
        sign_in(at, "rootadmin", "definitely-wrong")
    assert any("Too many failed attempts" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# Forced password change
# ---------------------------------------------------------------------------


def test_bootstrap_admin_must_change_password_before_using_the_app(app_env):
    at = run_app()
    sign_in(at, "rootadmin", BOOTSTRAP_PASSWORD)

    assert not at.exception
    assert [i.label for i in at.text_input] == [
        "Current password",
        "New password",
        "Confirm new password",
    ]
    assert "Get Started" not in {b.label for b in at.button}


def test_force_change_rejects_reuse_and_weak_passwords(app_env):
    at = run_app()
    sign_in(at, "rootadmin", BOOTSTRAP_PASSWORD)

    change_password(at, BOOTSTRAP_PASSWORD, BOOTSTRAP_PASSWORD)
    assert any("different from the current one" in e.value for e in at.error)

    change_password(at, BOOTSTRAP_PASSWORD, "short1!A")
    assert any("at least 12 characters" in e.value for e in at.error)


def test_force_change_rejects_wrong_current_password(app_env):
    at = run_app()
    sign_in(at, "rootadmin", BOOTSTRAP_PASSWORD)
    at.text_input(key="pwchange_current").set_value("wrong-password")
    at.text_input(key="pwchange_new").set_value(NEW_PASSWORD)
    at.text_input(key="pwchange_confirm").set_value(NEW_PASSWORD)
    at.button[0].click().run()
    assert any("current password is incorrect" in e.value for e in at.error)


def test_successful_change_unlocks_the_dashboard(app_env):
    at = signed_in_admin(app_env)
    assert not at.exception
    labels = {b.label for b in at.button}
    assert "Get Started" in labels
    assert "Sign out" in labels
    record = user_store.get_user_by_username("rootadmin")
    assert record.force_password_change is False


# ---------------------------------------------------------------------------
# Role-based surfaces
# ---------------------------------------------------------------------------


def test_administrator_sees_the_admin_console(app_env):
    at = signed_in_admin(app_env)
    assert "Open administrator console" in {b.label for b in at.button}

    assert click(at, "Open administrator console")
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Users",
        "Samples",
        "Model Access",
        "Connectivity",
        "Audit Log",
        "Storage and System",
    ]


def test_regular_user_cannot_reach_the_admin_console(app_env):
    at = signed_in_admin(app_env)
    user_store.create_user(
        username="worker",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )

    click(at, "Sign out")
    sign_in(at, "worker", NEW_PASSWORD)
    assert not at.exception
    assert "Open administrator console" not in {b.label for b in at.button}

    # Forcing the view directly is refused and audited.
    at.session_state.app_view = "admin"
    at.run()
    assert any("do not have permission" in e.value for e in at.error)
    denied = user_store.get_audit_events(event_type="authz.denied")
    assert denied and denied[0]["actor_username"] == "worker"


def test_regular_user_history_is_scoped_to_their_own_rows(app_env):
    at = signed_in_admin(app_env)
    admin = user_store.get_user_by_username("rootadmin")
    worker = user_store.create_user(
        username="worker2",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    database.insert_inventory_count(
        {
            "yard": "Admin Yard",
            "inventory_type": "Fence Panel",
            "user_id": admin.id,
            "username": admin.username,
        }
    )
    database.insert_inventory_count(
        {
            "yard": "Worker Yard",
            "inventory_type": "Fence Panel",
            "user_id": worker.id,
            "username": worker.username,
        }
    )

    import app as app_module

    admin_rows = app_module._visible_history_rows(to_authenticated_user(admin))
    worker_rows = app_module._visible_history_rows(to_authenticated_user(worker))
    assert {r["yard"] for r in admin_rows} == {"Admin Yard", "Worker Yard"}
    assert {r["yard"] for r in worker_rows} == {"Worker Yard"}


# ---------------------------------------------------------------------------
# Session lifecycle and state isolation
# ---------------------------------------------------------------------------


def test_sign_out_clears_identity_and_returns_to_login(app_env):
    at = signed_in_admin(app_env)
    assert click(at, "Sign out")
    assert not at.exception
    assert sget(at, "auth_user") is None
    assert [i.label for i in at.text_input] == ["Username", "Password"]
    assert any("signed out" in i.value for i in at.info)


def test_sign_out_clears_the_session_openrouter_key(app_env):
    at = signed_in_admin(app_env)
    at.session_state.openrouter_api_key = "sk-or-v1-" + "h" * 32
    at.session_state.openrouter_key_status = {"verified": True}
    at.session_state.openrouter_cost_ack = "2026-07-31T00:00:00+00:00"

    click(at, "Sign out")
    assert "openrouter_api_key" not in at.session_state
    assert "openrouter_key_status" not in at.session_state
    assert "openrouter_cost_ack" not in at.session_state


def test_sign_out_clears_wizard_and_analysis_state(app_env):
    at = signed_in_admin(app_env)
    at.session_state.analysis_results = ["leftover"]
    at.session_state.uploaded_images = [{"name": "secret.jpg"}]
    at.session_state.inference_cache = {"k": "v"}

    click(at, "Sign out")
    # Defaults are re-seeded on the next rerun, so assert the data itself is gone.
    for key in ("analysis_results", "uploaded_images", "inference_cache"):
        assert not sget(at, key, None)


def test_expired_session_returns_to_login_with_a_notice(app_env):
    at = signed_in_admin(app_env)
    at.session_state.auth_last_activity = "2020-01-01T00:00:00+00:00"
    at.run()

    assert sget(at, "auth_user") is None
    assert any("inactivity" in i.value for i in at.info)


def test_deactivating_a_user_revokes_their_live_session(app_env):
    at = signed_in_admin(app_env)
    user_store.create_user(
        username="worker3",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    click(at, "Sign out")
    sign_in(at, "worker3", NEW_PASSWORD)
    assert sget(at, "auth_user") is not None

    worker = user_store.get_user_by_username("worker3")
    user_store.set_user_active(worker.id, False)
    at.run()

    assert sget(at, "auth_user") is None
    assert any("no longer valid" in i.value for i in at.info)


# ---------------------------------------------------------------------------
# API connections page
# ---------------------------------------------------------------------------


def test_api_connections_page_offers_key_entry_and_cost_notice(app_env):
    at = signed_in_admin(app_env)
    assert click(at, "API Keys")
    assert not at.exception
    assert "OpenRouter API key" in [i.label for i in at.text_input]
    assert any("billed to you" in w.value for w in at.warning)
    assert any(
        "billed to my own account" in c.label for c in at.checkbox
    )


def test_verified_key_is_held_in_session_only(app_env, monkeypatch):
    at = signed_in_admin(app_env)
    click(at, "API Keys")

    key = "sk-or-v1-" + "j" * 32
    at.session_state.openrouter_api_key = key
    at.session_state.openrouter_key_status = {"verified": True, "masked": "sk-o…jjjj"}
    at.run()

    # Nothing on disk may contain the key.
    import config

    for path in config.DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix in {".db", ".json"}:
            assert key not in path.read_bytes().decode("utf-8", errors="ignore")


def test_cost_notice_state_helpers(app_env):
    at = signed_in_admin(app_env)
    assert sget(at, "openrouter_cost_ack") is None
    at.session_state.openrouter_cost_ack = "2026-07-31T00:00:00+00:00"
    at.run()
    assert sget(at, "openrouter_cost_ack")
