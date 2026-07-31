"""Login, forced password change, account and user-menu surfaces."""

from __future__ import annotations

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
    st.markdown(
        """
        <div class="aic-dash-hero">
          <div class="aic-rgb-bar"></div>
          <h1>AI Inventory Counter</h1>
          <p>Sign in to count inventory from photos.</p>
        </div>
        """,
        unsafe_allow_html=True,
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
        # Password field must not persist across the rerun.
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
    st.markdown(
        """
        <div class="aic-dash-hero">
          <div class="aic-rgb-bar"></div>
          <h1>Set a new password</h1>
          <p>Your password must be changed before you can continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
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
            st.success("Password updated. Loading your dashboard…")
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

    # The stored session_version just advanced; refresh the live session so the
    # user is not immediately signed out by their own change.
    if updated is not None:
        st.session_state.auth_user = replace(
            to_authenticated_user(updated), authenticated_at=user.authenticated_at
        )

    for key in ("pwchange_current", "pwchange_new", "pwchange_confirm"):
        st.session_state.pop(key, None)
    return []


# ---------------------------------------------------------------------------
# User menu and account page
# ---------------------------------------------------------------------------


def render_user_menu(user: AuthenticatedUser, *, on_navigate=None) -> None:
    """Always-visible identity strip with role and sign-out."""
    role_label = "Administrator" if user.is_admin else "User"
    cols = st.columns([3, 1, 1, 1], gap="small")
    with cols[0]:
        st.markdown(
            f"**{user.label}** · {role_label}"
            + (
                " · OpenRouter key active"
                if auth_session.has_verified_openrouter_key()
                else ""
            )
        )
    with cols[1]:
        if st.button("Account", key="menu_account", width="stretch"):
            if on_navigate:
                on_navigate("account")
    with cols[2]:
        if st.button("API Keys", key="menu_api_keys", width="stretch"):
            if on_navigate:
                on_navigate("api_connections")
    with cols[3]:
        if st.button("Sign out", key="menu_signout", width="stretch"):
            auth_session.end_session(
                reason="logout", notice="You have been signed out."
            )
            st.rerun()


def render_account_page(user: AuthenticatedUser) -> None:
    st.markdown("### Your account")
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
