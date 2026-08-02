"""End-to-end Streamlit flows: login gate, forced change, roles, isolation."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import auth_session
import database
import user_store
from auth import to_authenticated_user

# Deliberately trivial: this POC enforces no password complexity.
BOOTSTRAP_PASSWORD = "admin"
NEW_PASSWORD = "changed"


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


def click_key(at: AppTest, key: str) -> bool:
    for button in at.button:
        if getattr(button, "key", None) == key:
            button.click().run()
            return True
    return False


def button_keys(at: AppTest) -> set[str]:
    return {getattr(b, "key", None) for b in at.button if getattr(b, "key", None)}


def sign_in(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input(key="login_username").set_value(username)
    at.text_input(key="login_password").set_value(password)
    # AppTest may retain a prior Sign in submitter; use the latest one.
    sign_ins = [b for b in at.button if b.label == "Sign in"]
    if not sign_ins:
        raise AssertionError("Sign in button not found")
    sign_ins[-1].click().run()
    return at


def change_password(at: AppTest, current: str, new: str) -> AppTest:
    at.text_input(key="pwchange_current").set_value(current)
    at.text_input(key="pwchange_new").set_value(new)
    at.text_input(key="pwchange_confirm").set_value(new)
    for button in at.button:
        if button.label == "Update password":
            button.click().run()
            return at
    raise AssertionError("Update password button not found")


def signed_in_admin(app_env) -> AppTest:
    """Bootstrap admin reaches the dashboard directly — no forced change."""
    at = run_app()
    sign_in(at, "rootadmin", BOOTSTRAP_PASSWORD)
    return at


def signed_in_user_needing_change(app_env, username: str = "newhire") -> AppTest:
    """Sign in as a user the admin created, who must set a password first."""
    at = run_app()
    user_store.create_user(
        username=username, password=BOOTSTRAP_PASSWORD, force_password_change=True
    )
    sign_in(at, username, BOOTSTRAP_PASSWORD)
    return at


# ---------------------------------------------------------------------------
# Login gate
# ---------------------------------------------------------------------------


def test_app_opens_on_login_screen(app_env):
    at = run_app()
    assert not at.exception
    # Login fields lead; signup/reset live in expanders (extra inputs may appear).
    assert "Username" in [i.label for i in at.text_input]
    assert "Password" in [i.label for i in at.text_input]
    assert "Sign in" in [b.label for b in at.button]
    assert any("Create an account" in (e.label or "") for e in at.expander)
    assert any("Forgot password?" in (e.label or "") for e in at.expander)


def test_no_dashboard_content_leaks_before_sign_in(app_env):
    at = run_app()
    labels = {b.label for b in at.button}
    keys = button_keys(at)
    assert "Get Started" not in labels
    assert "Open administrator console" not in labels
    assert "menu_signout" not in keys
    assert "nav_home" not in keys


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


def test_bootstrap_admin_reaches_the_dashboard_directly(app_env):
    at = signed_in_admin(app_env)
    assert not at.exception
    keys = button_keys(at)
    # Admins land on Administration; Get Started is Home-only (not in the left panel).
    assert "nav_admin" in keys
    assert "nav_home" in keys
    assert "nav_history" in keys
    assert "nav_ai_configuration" in keys
    assert "nav_diagnostics" in keys
    assert "nav_api_keys" in keys
    assert "menu_signout" in keys
    assert "nav_theme_toggle" in keys
    assert "nav_get_started" not in keys
    assert [t.label for t in at.tabs][0] == "Overview"
    record = user_store.get_user_by_username("rootadmin")
    assert record.force_password_change is False


def test_trivial_admin_password_is_accepted(app_env):
    # admin/admin is a supported configuration for this POC.
    at = run_app()
    user_store.create_user(username="admin", password="admin", role="admin",
                           force_password_change=False)
    sign_in(at, "admin", "admin")
    assert not at.exception
    assert sget(at, "auth_user") is not None
    assert "nav_admin" in button_keys(at)


def test_admin_created_user_must_set_a_password_first(app_env):
    at = signed_in_user_needing_change(app_env)
    assert not at.exception
    assert [i.label for i in at.text_input] == [
        "Current password",
        "New password",
        "Confirm new password",
    ]
    assert "Get Started" not in {b.label for b in at.button}


def test_force_change_still_requires_matching_confirmation(app_env):
    at = signed_in_user_needing_change(app_env)
    at.text_input(key="pwchange_current").set_value(BOOTSTRAP_PASSWORD)
    at.text_input(key="pwchange_new").set_value("one")
    at.text_input(key="pwchange_confirm").set_value("two")
    assert click(at, "Update password")
    assert any("do not match" in e.value for e in at.error)


def test_force_change_rejects_wrong_current_password(app_env):
    at = signed_in_user_needing_change(app_env)
    at.text_input(key="pwchange_current").set_value("wrong-password")
    at.text_input(key="pwchange_new").set_value(NEW_PASSWORD)
    at.text_input(key="pwchange_confirm").set_value(NEW_PASSWORD)
    assert click(at, "Update password")
    assert any("current password is incorrect" in e.value for e in at.error)


def test_force_change_accepts_any_non_empty_password(app_env):
    at = signed_in_user_needing_change(app_env)
    change_password(at, BOOTSTRAP_PASSWORD, "x")

    assert not at.exception
    assert "Get Started" in {b.label for b in at.button}
    record = user_store.get_user_by_username("newhire")
    assert record.force_password_change is False
    assert user_store.verify_credentials("newhire", "x").status == "authenticated"


# ---------------------------------------------------------------------------
# Role-based surfaces
# ---------------------------------------------------------------------------


def test_administrator_sees_the_admin_console(app_env):
    at = signed_in_admin(app_env)
    assert "nav_admin" in button_keys(at)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Users",
        "Samples",
        "Model Access",
        "Experimental Features",
        "Connectivity",
        "Audit Log",
        "Storage and System",
    ]


def test_regular_user_cannot_reach_the_admin_console(app_env):
    run_app()  # bootstrap admin + migrate
    user_store.create_user(
        username="worker",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    # Fresh AppTest avoids logout/re-login widget residue on the login forms.
    at = run_app()
    sign_in(at, "worker", NEW_PASSWORD)
    assert not at.exception
    keys = button_keys(at)
    assert "nav_admin" not in keys
    assert "nav_api_keys" not in keys
    assert "nav_home" in keys
    assert "nav_history" in keys
    assert "nav_ai_configuration" in keys
    assert "nav_diagnostics" in keys
    assert "nav_profile" in keys
    assert "nav_theme_toggle" in keys
    assert "Get Started" in {b.label for b in at.button}

    # Forcing the view directly is refused and audited.
    at.session_state.app_view = "admin"
    at.run()
    assert any("do not have permission" in e.value for e in at.error)
    denied = user_store.get_audit_events(event_type="authz.denied")
    assert denied and denied[0]["actor_username"] == "worker"


def test_history_is_private_per_user_including_admins(app_env):
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
    # Unowned / pre-auth rows must never appear in anyone's private history.
    database.insert_inventory_count(
        {"yard": "Legacy Shared Yard", "inventory_type": "Fence Panel"}
    )

    import app as app_module

    admin_rows = app_module._visible_history_rows(to_authenticated_user(admin))
    worker_rows = app_module._visible_history_rows(to_authenticated_user(worker))
    assert {r["yard"] for r in admin_rows} == {"Admin Yard"}
    assert {r["yard"] for r in worker_rows} == {"Worker Yard"}
    assert "Legacy Shared Yard" not in {r["yard"] for r in admin_rows}
    assert "Legacy Shared Yard" not in {r["yard"] for r in worker_rows}


# ---------------------------------------------------------------------------
# Session lifecycle and state isolation
# ---------------------------------------------------------------------------


def test_sign_out_clears_identity_and_returns_to_login(app_env):
    at = signed_in_admin(app_env)
    assert click_key(at, "menu_signout")
    assert not at.exception
    assert sget(at, "auth_user") is None
    labels = [i.label for i in at.text_input]
    assert "Username" in labels
    assert "Password" in labels
    assert "Sign in" in [b.label for b in at.button]
    assert any("signed out" in i.value for i in at.info)


def test_sign_out_clears_the_session_openrouter_key(app_env):
    at = signed_in_admin(app_env)
    at.session_state.openrouter_api_key = "sk-or-v1-" + "h" * 32
    at.session_state.openrouter_key_status = {"verified": True}
    at.session_state.openrouter_cost_ack = "2026-07-31T00:00:00+00:00"

    click_key(at, "menu_signout")
    assert "openrouter_api_key" not in at.session_state
    assert "openrouter_key_status" not in at.session_state
    assert "openrouter_cost_ack" not in at.session_state


def test_sign_out_clears_wizard_and_analysis_state(app_env):
    at = signed_in_admin(app_env)
    at.session_state.analysis_results = ["leftover"]
    at.session_state.uploaded_images = [{"name": "secret.jpg"}]
    at.session_state.inference_cache = {"k": "v"}

    click_key(at, "menu_signout")
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
    run_app()
    user_store.create_user(
        username="worker3",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    at = run_app()
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


def test_admin_api_keys_page_offers_deployment_key_entry(app_env):
    at = signed_in_admin(app_env)
    assert "nav_api_keys" in button_keys(at)
    assert click_key(at, "nav_api_keys")
    assert not at.exception
    assert "OpenRouter API key" in [i.label for i in at.text_input]
    assert not any(
        "billed to the administrator" in (w.value or "").lower() for w in at.warning
    )


def test_regular_user_never_sees_api_keys_or_openrouter_key_ui(app_env):
    run_app()
    user_store.create_user(
        username="norights",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    at = run_app()
    sign_in(at, "norights", NEW_PASSWORD)
    assert not at.exception
    keys = button_keys(at)
    assert "nav_api_keys" not in keys
    assert "OpenRouter API key" not in [i.label for i in at.text_input]
    assert "Add OpenRouter key" not in {b.label for b in at.button}
    assert "Configure OpenRouter key" not in {b.label for b in at.button}

    assert click_key(at, "nav_ai_configuration")
    assert "OpenRouter API key" not in [i.label for i in at.text_input]

    # Forcing the legacy view is refused — same as other admin-only pages.
    at.session_state.app_view = "api_connections"
    at.run()
    assert any("do not have permission" in e.value for e in at.error)
    assert "OpenRouter API key" not in [i.label for i in at.text_input]


def test_admin_deployment_key_unlocks_openrouter_for_users(app_env):
    import model_access
    import openrouter_store

    at = signed_in_admin(app_env)
    key = "sk-or-v1-" + "j" * 32
    openrouter_store.save_deployment_key(
        key,
        verification={"verified": True, "masked": "sk-o…jjjj"},
        updated_by="rootadmin",
    )
    model_access.ensure_default_policies()
    user_store.upsert_model_policy(
        "workflow:hariram-s-mzhvc/playground-gpt-5-6-luna-od",
        is_enabled=True,
        requires_user_api_key=False,
    )

    worker = user_store.create_user(
        username="runner",
        password=NEW_PASSWORD,
        role="user",
        force_password_change=False,
    )
    from auth import to_authenticated_user
    from schemas import ModelConfig

    model = ModelConfig(
        name="OpenRouter VLM Detector",
        kind="workflow",
        enabled=True,
        workspace_name="hariram-s-mzhvc",
        workflow_id="playground-gpt-5-6-luna-od",
        key="workflow:hariram-s-mzhvc/playground-gpt-5-6-luna-od",
        provider="openrouter",
        requires_user_api_key=True,
    )
    decision = model_access.evaluate_model_access(
        model,
        to_authenticated_user(worker),
        has_verified_key=openrouter_store.has_verified_deployment_key(),
    )
    assert decision.allowed
    # Audit / status never expose the plaintext key.
    status = openrouter_store.get_deployment_key_status()
    assert status.masked
    assert key not in status.masked
    assert key not in str(user_store.get_audit_events(limit=50))
