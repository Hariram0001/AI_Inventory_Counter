"""Login, forced password change, account and left-panel navigation."""

from __future__ import annotations

import html
from dataclasses import replace

import streamlit as st

import auth_session
from auth import (
    EVENT_ACCESS_DENIED,
    EVENT_LOGIN_FAILURE,
    EVENT_LOGIN_LOCKED,
    EVENT_LOGIN_SUCCESS,
    EVENT_PASSWORD_CHANGED,
    AuthenticatedUser,
    bootstrap_admin_if_needed,
    get_auth_provider,
    to_authenticated_user,
)
from security import (
    MIN_PASSWORD_LENGTH,
    validate_email,
    validate_password_policy,
    verify_password,
)
from user_store import (
    get_user_by_id,
    record_audit_event,
    set_password,
    update_user_profile,
)
from database import _connect

PASSWORD_HELP = (
    f"At least {MIN_PASSWORD_LENGTH} characters, mixing at least three of "
    "lowercase, uppercase, digits and symbols. Common or placeholder phrases "
    "are rejected."
)


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def render_login_page() -> None:
    """Full-page sign-in. There is no public self-registration."""
    from ui_helpers import render_page_hero

    render_page_hero(
        "AI Inventory Counter",
        "Sign in to count inventory from photos.",
    )

    bootstrap = bootstrap_admin_if_needed()
    if bootstrap.status == "created":
        st.success(bootstrap.message)
    elif bootstrap.status == "misconfigured":
        st.warning(bootstrap.message)
    elif bootstrap.status == "error":
        st.error("The user database could not be prepared. Check the server logs.")

    notice = st.session_state.pop("auth_logout_notice", "")
    if notice:
        st.info(notice)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input(
            "Username", key="login_username", autocomplete="username"
        )
        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            autocomplete="current-password",
        )
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

    if submitted:
        _handle_login_submit(username, password)

    error = st.session_state.get("auth_login_error")
    if error:
        st.error(error)

    st.caption(
        "Accounts are created by an administrator. "
        + auth_session.session_timeout_summary()
    )


def _handle_login_submit(username: str, password: str) -> None:
    outcome = get_auth_provider().authenticate(username, password)

    if outcome.ok and outcome.user is not None:
        st.session_state.pop("auth_login_error", None)
        auth_session.start_session(outcome.user)
        record_audit_event(
            EVENT_LOGIN_SUCCESS,
            actor_user_id=outcome.user.user_id,
            actor_username=outcome.user.username,
            target_type="session",
            target_id=outcome.user.username,
            detail={"role": outcome.user.role},
        )
        st.session_state.pop("login_password", None)
        st.rerun()
        return

    st.session_state.auth_login_error = outcome.message or "Sign-in failed."
    event = EVENT_LOGIN_LOCKED if outcome.status == "locked" else EVENT_LOGIN_FAILURE
    record_audit_event(
        event,
        actor_username=str(username or "").strip().lower() or None,
        target_type="session",
        target_id=str(username or "").strip().lower() or None,
        outcome="failure",
        detail={"status": outcome.status},
    )
    st.session_state.pop("login_password", None)


# ---------------------------------------------------------------------------
# Forced password change
# ---------------------------------------------------------------------------


def render_force_password_change(user: AuthenticatedUser) -> None:
    """Blocking screen shown until the user sets their own password."""
    from ui_helpers import render_page_hero

    render_page_hero(
        "Set a new password",
        "Your password must be changed before you can continue.",
    )
    st.info(
        "You are signed in as "
        f"**{user.label}**. Choose a new password to unlock the application."
    )
    st.caption(PASSWORD_HELP)

    with st.form("force_password_change_form", clear_on_submit=False):
        current = st.text_input(
            "Current password", type="password", key="pwchange_current"
        )
        new_password = st.text_input(
            "New password", type="password", key="pwchange_new"
        )
        confirm = st.text_input(
            "Confirm new password", type="password", key="pwchange_confirm"
        )
        submitted = st.form_submit_button(
            "Update password", type="primary", width="stretch"
        )

    if submitted:
        problems = change_own_password(user, current, new_password, confirm)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            st.session_state.app_view = "admin" if user.is_admin else "welcome"
            st.success("Password updated. Loading your workspace…")
            st.rerun()

    st.divider()
    if st.button("Sign out", key="pwchange_signout", width="stretch"):
        auth_session.end_session(reason="logout", notice="You have been signed out.")
        st.rerun()


def change_own_password(
    user: AuthenticatedUser,
    current_password: str,
    new_password: str,
    confirm_password: str,
) -> list[str]:
    """Validate and apply a self-service password change.

    Returns user-safe problem messages; an empty list means the change applied.
    """
    problems: list[str] = []

    if not current_password:
        problems.append("Enter your current password.")
    if new_password != confirm_password:
        problems.append("The new passwords do not match.")
    if current_password and new_password and current_password == new_password:
        problems.append("The new password must be different from the current one.")

    problems.extend(
        validate_password_policy(
            new_password or "", username=user.username, email=user.email
        )
    )

    if problems:
        return problems

    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (int(user.user_id),)
        ).fetchone()
    stored_hash = str(row["password_hash"]) if row else ""
    if not verify_password(stored_hash, current_password):
        record_audit_event(
            EVENT_PASSWORD_CHANGED,
            actor_user_id=user.user_id,
            actor_username=user.username,
            target_type="user",
            target_id=user.username,
            outcome="failure",
            detail={"reason": "current_password_mismatch"},
        )
        return ["Your current password is incorrect."]

    updated = set_password(
        user.user_id,
        new_password,
        force_password_change=False,
        invalidate_sessions=True,
    )
    record_audit_event(
        EVENT_PASSWORD_CHANGED,
        actor_user_id=user.user_id,
        actor_username=user.username,
        target_type="user",
        target_id=user.username,
        detail={"self_service": True},
    )

    if updated is not None:
        st.session_state.auth_user = replace(
            to_authenticated_user(updated), authenticated_at=user.authenticated_at
        )

    for key in ("pwchange_current", "pwchange_new", "pwchange_confirm"):
        st.session_state.pop(key, None)
    return []


# ---------------------------------------------------------------------------
# Left navigation + account page
# ---------------------------------------------------------------------------


def role_badge_label(user: AuthenticatedUser) -> str:
    return "Admin" if user.is_admin else "User"


def render_app_sidebar(user: AuthenticatedUser) -> None:
    """Icon-only left panel using outline Material icons; hover for labels."""
    from ui_helpers import navigate_to, normalize_view

    view = normalize_view(st.session_state.get("app_view") or "welcome")
    role = role_badge_label(user)

    def _icon_nav(
        material_icon: str,
        label: str,
        key: str,
        *,
        active: bool,
        target: str,
    ) -> None:
        # Zero-width label keeps the skeletal Material glyph only; help = hover text.
        if st.button(
            "\u200b",
            key=key,
            help=label,
            icon=f":material/{material_icon}:",
            type="primary" if active else "secondary",
            width="stretch",
        ):
            navigate_to(target)

    with st.sidebar:
        st.markdown(
            '<div class="aic-side-brand" title="AI Inventory Counter">'
            '<div class="aic-rgb-bar"></div></div>',
            unsafe_allow_html=True,
        )

        _icon_nav(
            "home", "Home", "nav_home", active=view == "welcome", target="welcome"
        )

        if user.is_admin:
            _icon_nav(
                "admin_panel_settings",
                "Administration",
                "nav_admin",
                active=view == "admin",
                target="admin",
            )

        _icon_nav(
            "history",
            "Inventory History",
            "nav_history",
            active=view == "history",
            target="history",
        )
        _icon_nav(
            "psychology",
            "AI Configuration",
            "nav_ai_configuration",
            active=view == "ai_configuration",
            target="ai_configuration",
        )
        _icon_nav(
            "monitor_heart",
            "Diagnostics",
            "nav_diagnostics",
            active=view == "diagnostics",
            target="diagnostics",
        )
        if user.is_admin:
            _icon_nav(
                "key",
                "API Keys",
                "nav_api_keys",
                active=view == "api_keys",
                target="api_keys",
            )

        st.markdown('<div class="aic-side-spacer"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            f"""
            <div class="aic-side-profile" title="{html.escape(user.label)} · {html.escape(role)}">
              <span class="aic-role-badge aic-role-{'admin' if user.is_admin else 'user'}">{html.escape(role)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _icon_nav(
            "person",
            f"Profile · {user.label}",
            "nav_profile",
            active=view == "account",
            target="account",
        )
        if st.button(
            "\u200b",
            key="menu_signout",
            help="Sign out",
            icon=":material/logout:",
            width="stretch",
        ):
            auth_session.end_session(
                reason="logout", notice="You have been signed out."
            )
            st.rerun()


def render_user_menu(user: AuthenticatedUser, *, on_navigate=None) -> None:
    """Deprecated top chrome — identity now lives in the left sidebar."""
    del user, on_navigate


def render_account_page(user: AuthenticatedUser) -> None:
    record = get_user_by_id(user.user_id)
    if record is None:
        st.error("Your account could not be loaded.")
        return

    left, right = st.columns(2)
    with left:
        st.metric("Username", record.username)
        st.caption(f"Role: {'Administrator' if record.is_admin else 'User'}")
    with right:
        st.metric("Last sign-in", (record.last_login_at or "—")[:19].replace("T", " "))
        st.caption(auth_session.session_timeout_summary())

    st.divider()
    st.markdown("#### Profile")
    with st.form("account_profile_form"):
        display_name = st.text_input("Display name", value=record.display_name)
        email = st.text_input("Email", value=record.email)
        saved = st.form_submit_button("Save profile")
    if saved:
        try:
            clean_email = validate_email(email)
        except ValueError as exc:
            st.error(str(exc))
        else:
            update_user_profile(
                record.id, email=clean_email, display_name=display_name
            )
            st.success("Profile updated.")
            st.rerun()

    st.divider()
    st.markdown("#### Change password")
    st.caption(PASSWORD_HELP)
    with st.form("account_password_form"):
        current = st.text_input(
            "Current password", type="password", key="pwchange_current"
        )
        new_password = st.text_input("New password", type="password", key="pwchange_new")
        confirm = st.text_input(
            "Confirm new password", type="password", key="pwchange_confirm"
        )
        submitted = st.form_submit_button("Update password", type="primary")
    if submitted:
        problems = change_own_password(user, current, new_password, confirm)
        if problems:
            for problem in problems:
                st.error(problem)
        else:
            st.success("Password updated.")


def deny_access(reason: str, *, user: AuthenticatedUser | None = None) -> None:
    """Render a uniform authorization failure and record it."""
    record_audit_event(
        EVENT_ACCESS_DENIED,
        actor_user_id=user.user_id if user else None,
        actor_username=user.username if user else None,
        target_type="page",
        target_id=reason,
        outcome="failure",
        detail={"reason": reason},
    )
    st.error("You do not have permission to view this page.")
