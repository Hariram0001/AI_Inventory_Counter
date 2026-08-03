"""Administrator console: users, samples, model access, connectivity, audit."""

from __future__ import annotations

import html
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import admin_samples
import config
import model_access
from auth import (
    EVENT_PASSWORD_RESET,
    EVENT_POLICY_UPDATED,
    EVENT_SAMPLE_DELETED,
    EVENT_SAMPLE_UPDATED,
    EVENT_SAMPLE_UPLOADED,
    EVENT_SIGNUP_APPROVED,
    EVENT_SIGNUP_REJECTED,
    EVENT_USER_ACTIVATED,
    EVENT_USER_CREATED,
    EVENT_USER_DEACTIVATED,
    EVENT_USER_DELETED,
    EVENT_USER_UNLOCKED,
    EVENT_USER_UPDATED,
    ROLE_ADMIN,
    ROLE_USER,
    AuthenticatedUser,
)
from database import count_inventory_rows, get_schema_version
from security import (
    generate_temporary_password,
    redact_secrets,
    validate_email,
    validate_password_policy,
    validate_username,
)
from user_store import (
    RESET_STATUS_FULFILLED,
    RESET_STATUS_REJECTED,
    ROLES,
    UserStoreError,
    approve_pending_signup,
    count_active_admins,
    delete_user,
    get_audit_events,
    get_password_reset_request,
    get_usage_summary,
    get_user_by_id,
    list_audit_event_types,
    list_model_policies,
    list_password_reset_requests,
    list_pending_signups,
    list_users,
    lock_remaining_seconds,
    record_audit_event,
    reject_pending_signup,
    resolve_password_reset_request,
    set_password,
    set_user_active,
    set_user_role,
    unlock_user,
    upsert_model_policy,
)
from user_store import create_user as store_create_user

ADMIN_TABS = (
    "Overview",
    "Users",
    "Samples",
    "Model Access",
    "Experimental Features",
    "Connectivity",
    "Audit Log",
    "Storage and System",
)


def _admin_section(title: str, caption: str = "") -> None:
    caption_html = f"<p>{html.escape(caption)}</p>" if caption else ""
    st.markdown(
        f'<div class="aic-admin-section"><h4>{html.escape(title)}</h4>{caption_html}</div>',
        unsafe_allow_html=True,
    )


def _admin_metric_cards(items: list[tuple[str, Any]], *, columns: int = 4) -> None:
    cls = "aic-admin-metrics" + ("" if columns == 4 else f" aic-admin-metrics-{columns}")
    cells = "".join(
        f'<div class="aic-admin-metric"><span class="val">{html.escape(str(value))}</span>'
        f'<span class="lbl">{html.escape(label)}</span></div>'
        for label, value in items
    )
    st.markdown(f'<div class="{cls}">{cells}</div>', unsafe_allow_html=True)


def render_admin_console(user: AuthenticatedUser) -> None:
    """Entry point. Callers must already have verified the administrator role."""
    if not user.is_admin:
        st.error("You do not have permission to view this page.")
        return

    from ui_helpers import render_page_hero

    render_page_hero(
        "Administration",
        "Manage accounts, model access, demo samples, and connectivity for this deployment.",
    )

    notice = st.session_state.pop("admin_action_notice", "")
    if notice:
        st.success(notice)
    error = st.session_state.pop("admin_action_error", "")
    if error:
        st.error(error)

    tabs = st.tabs(list(ADMIN_TABS))
    with tabs[0]:
        _render_overview(user)
    with tabs[1]:
        _render_users(user)
    with tabs[2]:
        _render_samples(user)
    with tabs[3]:
        _render_model_access(user)
    with tabs[4]:
        _render_experimental_features(user)
    with tabs[5]:
        _render_connectivity(user)
    with tabs[6]:
        _render_audit_log(user)
    with tabs[7]:
        _render_storage_and_system(user)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _render_overview(user: AuthenticatedUser) -> None:
    del user  # Overview is deployment-wide; actor is unused here.
    users = list_users()
    active = [u for u in users if u.is_active]
    locked = [u for u in users if u.is_locked()]

    _admin_section("Deployment snapshot", "Live counts for this environment.")
    _admin_metric_cards(
        [
            ("Users", len(users)),
            ("Active", len(active)),
            ("Administrators", sum(1 for u in active if u.is_admin)),
            ("Locked", len(locked)),
        ]
    )
    try:
        saved_counts: Any = count_inventory_rows()
    except Exception:  # noqa: BLE001
        saved_counts = "—"
    _admin_metric_cards(
        [
            ("Saved counts", saved_counts),
            ("Samples", len(admin_samples.list_samples())),
            ("Schema version", get_schema_version()),
        ],
        columns=3,
    )

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "OpenRouter usage",
        "Runs recorded in the last 7 days (no key material).",
    )
    usage = get_usage_summary(days=7)
    if usage:
        st.dataframe(pd.DataFrame(usage), width="stretch", hide_index=True)
    else:
        st.caption("No model runs have been recorded in the last 7 days.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("Recent activity", "Latest audit events across the deployment.")
    events = get_audit_events(limit=10)
    if events:
        rows_html: list[str] = ['<div class="aic-admin-activity">']
        for event in events:
            when = html.escape(str(event.get("created_at") or "")[:19].replace("T", " "))
            actor = html.escape(str(event.get("actor_username") or "system"))
            outcome = str(event.get("outcome") or "success")
            outcome_cls = "ok" if outcome.lower() in {"success", "ok"} else "bad"
            event_type = html.escape(str(event.get("event_type") or "—"))
            rows_html.append(
                "<div class='aic-admin-activity-row'>"
                f"<span class='when'>{when}</span>"
                f"<span class='actor'>{actor}</span>"
                f"<span class='event'>{event_type}</span>"
                f"<span class='outcome {outcome_cls}'>{html.escape(outcome)}</span>"
                "</div>"
            )
        rows_html.append("</div>")
        st.markdown("".join(rows_html), unsafe_allow_html=True)
    else:
        st.caption("No audit events recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def _render_users(admin: AuthenticatedUser) -> None:
    _render_pending_signups(admin)
    _render_password_reset_requests(admin)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Create a user",
        "A temporary password is generated once; the user must change it at first sign-in.",
    )
    with st.form("admin_create_user_form", clear_on_submit=True):
        cols = st.columns(2)
        with cols[0]:
            username = st.text_input("Username", key="admin_user_new_username")
            role = st.selectbox("Role", ROLES, index=1, key="admin_user_new_role")
        with cols[1]:
            display_name = st.text_input("Display name", key="admin_user_new_display")
            email = st.text_input("Email (optional)", key="admin_user_new_email")
        submitted = st.form_submit_button("Create user", type="primary")

    if submitted:
        _handle_create_user(admin, username, role, display_name, email)

    temp = st.session_state.pop("admin_temp_password", None)
    if temp:
        st.success(f"User '{temp['username']}' created.")
        st.warning(
            "Copy this temporary password now — it is not stored and cannot be "
            "shown again."
        )
        st.code(temp["password"], language=None)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("Manage users", "Select a row below, then apply role or account actions.")
    users = list_users()
    if not users:
        st.caption("No users exist yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows = []
    for record in users:
        remaining = lock_remaining_seconds(record.locked_until)
        rows.append(
            {
                "Username": record.username,
                "Name": record.display_name,
                "Role": record.role,
                "Status": record.account_status,
                "Active": record.is_active,
                "Locked": f"{remaining // 60} min" if remaining else "—",
                "Must change password": record.force_password_change,
                "Last sign-in": (record.last_login_at or "—")[:19].replace("T", " "),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    labels = {f"{u.username} ({u.role})": u.id for u in users}
    choice = st.selectbox("Select a user", list(labels), key="admin_user_pick")
    target = get_user_by_id(labels[choice])
    if target is None:
        return

    is_self = target.id == admin.user_id
    last_admin = target.is_admin and count_active_admins(exclude_user_id=target.id) == 0
    if last_admin:
        st.info(
            "This is the last active administrator. Role, activation and deletion "
            "controls are disabled to prevent lockout."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        new_role = st.selectbox(
            "Role",
            ROLES,
            index=ROLES.index(target.role) if target.role in ROLES else 1,
            key="admin_user_role_edit",
            disabled=last_admin,
        )
        if st.button("Apply role", key="admin_user_role_apply", disabled=last_admin):
            _guarded(
                lambda: set_user_role(target.id, new_role),
                admin,
                EVENT_USER_UPDATED,
                target.username,
                f"Role for '{target.username}' set to {new_role}.",
                detail={"role": new_role},
            )
    with c2:
        if target.is_active:
            if st.button(
                "Deactivate", key="admin_user_deactivate", disabled=last_admin or is_self
            ):
                _guarded(
                    lambda: set_user_active(target.id, False),
                    admin,
                    EVENT_USER_DEACTIVATED,
                    target.username,
                    f"'{target.username}' deactivated.",
                )
        else:
            if st.button("Activate", key="admin_user_activate"):
                _guarded(
                    lambda: set_user_active(target.id, True),
                    admin,
                    EVENT_USER_ACTIVATED,
                    target.username,
                    f"'{target.username}' activated.",
                )
    with c3:
        locked = lock_remaining_seconds(target.locked_until) > 0
        if st.button("Unlock", key="admin_user_unlock", disabled=not locked):
            _guarded(
                lambda: unlock_user(target.id),
                admin,
                EVENT_USER_UNLOCKED,
                target.username,
                f"'{target.username}' unlocked.",
            )

    st.markdown("##### Reset password")
    st.caption(
        "Generates a temporary password and forces a change at next sign-in. "
        "Existing sessions for that user are invalidated."
    )
    if st.button("Generate temporary password", key="admin_user_reset"):
        _handle_password_reset(admin, target.id, target.username)

    st.markdown("##### Delete user")
    st.caption(
        "Deleting a user does not delete their saved inventory history, which "
        "stays attributed to their username for audit purposes."
    )
    confirm = st.text_input(
        "Type the username to confirm deletion", key="admin_user_delete_confirm"
    )
    if st.button(
        "Delete user",
        key="admin_user_delete",
        disabled=last_admin or is_self or confirm.strip() != target.username,
    ):
        _guarded(
            lambda: delete_user(target.id),
            admin,
            EVENT_USER_DELETED,
            target.username,
            f"'{target.username}' deleted.",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_pending_signups(admin: AuthenticatedUser) -> None:
    pending = list_pending_signups()
    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Pending sign-ups",
        "Users who registered themselves. Approve to let them sign in with the "
        "password they chose, or reject to delete the request.",
    )
    if not pending:
        st.caption("No sign-up requests waiting.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for record in pending:
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.markdown(
                f"**{record.username}**"
                + (f" · {record.display_name}" if record.display_name else "")
                + (f" · {record.email}" if record.email else "")
            )
            st.caption(f"Requested {(record.created_at or '')[:19].replace('T', ' ')}")
        with cols[1]:
            if st.button(
                "Approve",
                key=f"admin_signup_approve_{record.id}",
                type="primary",
                width="stretch",
            ):
                _guarded(
                    lambda rid=record.id: approve_pending_signup(rid),
                    admin,
                    EVENT_SIGNUP_APPROVED,
                    record.username,
                    f"Approved '{record.username}'. They can sign in now.",
                    detail={"account_status": "active"},
                )
        with cols[2]:
            if st.button(
                "Reject",
                key=f"admin_signup_reject_{record.id}",
                width="stretch",
            ):
                _guarded(
                    lambda rid=record.id: reject_pending_signup(rid),
                    admin,
                    EVENT_SIGNUP_REJECTED,
                    record.username,
                    f"Rejected and removed '{record.username}'.",
                )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_password_reset_requests(admin: AuthenticatedUser) -> None:
    requests = list_password_reset_requests()
    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Password reset requests",
        "Authorize a request to generate a one-time temporary password. Deliver "
        "it to the user out of band — it is shown only once here.",
    )
    if not requests:
        st.caption("No pending password reset requests.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for req in requests:
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.markdown(f"**{req.username}**")
            st.caption(
                f"Requested {(req.requested_at or '')[:19].replace('T', ' ')}"
            )
        with cols[1]:
            if st.button(
                "Authorize reset",
                key=f"admin_reset_ok_{req.id}",
                type="primary",
                width="stretch",
            ):
                _handle_authorize_reset_request(admin, req.id)
        with cols[2]:
            if st.button(
                "Reject",
                key=f"admin_reset_reject_{req.id}",
                width="stretch",
            ):
                _guarded(
                    lambda rid=req.id: resolve_password_reset_request(
                        rid,
                        status=RESET_STATUS_REJECTED,
                        reviewed_by=admin.username,
                    ),
                    admin,
                    EVENT_PASSWORD_RESET,
                    req.username,
                    f"Rejected password reset for '{req.username}'.",
                    detail={"request_id": req.id, "status": "rejected"},
                )
    st.markdown("</div>", unsafe_allow_html=True)


def _handle_authorize_reset_request(admin: AuthenticatedUser, request_id: int) -> None:
    req = get_password_reset_request(request_id)
    if req is None or req.status != "pending":
        st.session_state.admin_action_error = "That reset request is no longer pending."
        st.rerun()
        return
    user = get_user_by_id(req.user_id) if req.user_id else None
    if user is None:
        user = next(
            (u for u in list_users() if u.username == req.username),
            None,
        )
    if user is None:
        st.session_state.admin_action_error = (
            f"No account found for '{req.username}'. Reject the request instead."
        )
        st.rerun()
        return

    temporary = generate_temporary_password()
    try:
        set_password(
            user.id,
            temporary,
            force_password_change=True,
            invalidate_sessions=True,
        )
        resolve_password_reset_request(
            request_id,
            status=RESET_STATUS_FULFILLED,
            reviewed_by=admin.username,
        )
    except (UserStoreError, ValueError) as exc:
        st.session_state.admin_action_error = str(exc)
        st.rerun()
        return

    record_audit_event(
        EVENT_PASSWORD_RESET,
        actor_user_id=admin.user_id,
        actor_username=admin.username,
        target_type="user",
        target_id=user.username,
        detail={
            "force_password_change": True,
            "request_id": request_id,
            "status": "fulfilled",
        },
    )
    st.session_state.admin_temp_password = {
        "username": user.username,
        "password": temporary,
    }
    st.session_state.admin_action_notice = (
        f"Authorized reset for '{user.username}'. Copy the temporary password below."
    )
    st.rerun()


def _handle_create_user(
    admin: AuthenticatedUser,
    username: str,
    role: str,
    display_name: str,
    email: str,
) -> None:
    try:
        clean_username = validate_username(username)
        clean_email = validate_email(email)
    except ValueError as exc:
        st.session_state.admin_action_error = str(exc)
        st.rerun()
        return

    temporary = generate_temporary_password()
    problems = validate_password_policy(
        temporary, username=clean_username, email=clean_email
    )
    if problems:
        # Regenerate once; the generator already screens against the policy.
        temporary = generate_temporary_password(20)

    try:
        store_create_user(
            username=clean_username,
            password=temporary,
            role=role if role in ROLES else ROLE_USER,
            email=clean_email,
            display_name=display_name,
            force_password_change=True,
            created_by=admin.username,
        )
    except (UserStoreError, ValueError) as exc:
        st.session_state.admin_action_error = str(exc)
        st.rerun()
        return

    record_audit_event(
        EVENT_USER_CREATED,
        actor_user_id=admin.user_id,
        actor_username=admin.username,
        target_type="user",
        target_id=clean_username,
        detail={"role": role, "force_password_change": True},
    )
    st.session_state.admin_temp_password = {
        "username": clean_username,
        "password": temporary,
    }
    st.rerun()


def _handle_password_reset(admin: AuthenticatedUser, user_id: int, username: str) -> None:
    temporary = generate_temporary_password()
    try:
        set_password(
            user_id,
            temporary,
            force_password_change=True,
            invalidate_sessions=True,
        )
    except (UserStoreError, ValueError) as exc:
        st.session_state.admin_action_error = str(exc)
        st.rerun()
        return

    record_audit_event(
        EVENT_PASSWORD_RESET,
        actor_user_id=admin.user_id,
        actor_username=admin.username,
        target_type="user",
        target_id=username,
        detail={"force_password_change": True},
    )
    st.session_state.admin_temp_password = {
        "username": username,
        "password": temporary,
    }
    st.rerun()


def _guarded(
    action,
    admin: AuthenticatedUser,
    event_type: str,
    target: str,
    success_message: str,
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    """Run an admin mutation, audit it, and surface a safe message."""
    try:
        action()
    except (UserStoreError, ValueError) as exc:
        st.session_state.admin_action_error = str(exc)
        record_audit_event(
            event_type,
            actor_user_id=admin.user_id,
            actor_username=admin.username,
            target_type="user",
            target_id=target,
            outcome="failure",
            detail={"error": str(exc)},
        )
        st.rerun()
        return

    record_audit_event(
        event_type,
        actor_user_id=admin.user_id,
        actor_username=admin.username,
        target_type="user",
        target_id=target,
        detail=detail or {},
    )
    st.session_state.admin_action_notice = success_message
    st.rerun()


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


def _render_samples(admin: AuthenticatedUser) -> None:
    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Upload a sample image",
        "Files are validated by decoding them. Enabled samples appear on Add Photos → "
        "Sample Images for the matching inventory type.",
    )

    inventory_options = ["Fence Panel"] + [
        key for key in getattr(config, "INVENTORY_TYPES", []) if key != "Fence Panel"
    ]

    with st.form("admin_sample_upload_form", clear_on_submit=True):
        upload = st.file_uploader(
            "Image file", type=["jpg", "jpeg", "png"], key="admin_sample_file"
        )
        sample_kind = st.selectbox(
            "Sample classification",
            ["Inventory sample", "Shape Detection sample"],
            key="admin_sample_kind",
        )
        cols = st.columns(2)
        with cols[0]:
            title = st.text_input("Title", key="admin_sample_title")
            inventory_type = st.selectbox(
                "Inventory type",
                inventory_options,
                key="admin_sample_inventory",
                disabled=sample_kind.startswith("Shape"),
            )
            expected_shape = st.text_input(
                "Expected shape (shape samples)",
                value="circle",
                key="admin_sample_shape",
            )
        with cols[1]:
            expected = st.number_input(
                "Expected count (optional)",
                min_value=0,
                value=0,
                step=1,
                key="admin_sample_expected",
            )
            verified = st.number_input(
                "Verified count (optional ground truth)",
                min_value=0,
                value=0,
                step=1,
                key="admin_sample_verified",
            )
            difficulty = st.text_input("Difficulty", key="admin_sample_difficulty")
            enabled = st.checkbox("Enabled", value=True, key="admin_sample_enabled")
        description = st.text_area("Description", key="admin_sample_description")
        submitted = st.form_submit_button("Upload sample", type="primary")

    if submitted:
        if upload is None:
            st.session_state.admin_action_error = "Choose an image file to upload."
            st.rerun()
        kind = (
            "shape_detection"
            if str(sample_kind).startswith("Shape")
            else "inventory"
        )
        try:
            sample = admin_samples.add_sample(
                data=upload.getvalue(),
                title=title,
                inventory_type="" if kind == "shape_detection" else inventory_type,
                description=description,
                expected_count=int(expected) if expected else None,
                uploaded_by=admin.username,
                is_enabled=bool(enabled),
                sample_kind=kind,
                expected_shape=expected_shape if kind == "shape_detection" else "",
                verified_count=int(verified) if verified else None,
                difficulty=difficulty,
            )
        except admin_samples.SampleValidationError as exc:
            st.session_state.admin_action_error = str(exc)
            st.rerun()
        else:
            record_audit_event(
                EVENT_SAMPLE_UPLOADED,
                actor_user_id=admin.user_id,
                actor_username=admin.username,
                target_type="sample",
                target_id=sample.sample_id,
                detail={
                    "inventory_type": sample.inventory_type,
                    "width": sample.width,
                    "height": sample.height,
                    "size_bytes": sample.size_bytes,
                },
            )
            st.session_state.admin_action_notice = f"Sample '{sample.title}' uploaded."
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("Manage samples", "Enable, disable, or remove uploaded demo images.")
    samples = admin_samples.list_samples()
    if not samples:
        st.caption("No administrator samples have been uploaded yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for sample in samples:
        with st.expander(
            f"{sample.title} · {sample.inventory_type or 'unassigned'}"
            + ("" if sample.is_enabled else " · disabled"),
            expanded=False,
        ):
            cols = st.columns([1, 2])
            with cols[0]:
                if sample.exists:
                    st.image(str(sample.path), width="stretch")
                else:
                    st.warning("The image file is missing from storage.")
            with cols[1]:
                st.caption(
                    f"{sample.width}×{sample.height} · "
                    f"{sample.size_bytes / 1024:.0f} KB · {sample.mime_type}"
                )
                st.caption(f"Uploaded by {sample.uploaded_by or 'unknown'}")
                if sample.description:
                    st.write(sample.description)
                if sample.expected_count is not None:
                    st.caption(f"Expected count: {sample.expected_count}")

                toggle_label = "Disable" if sample.is_enabled else "Enable"
                b1, b2 = st.columns(2)
                with b1:
                    if st.button(toggle_label, key=f"admin_sample_toggle_{sample.id}"):
                        admin_samples.update_sample(
                            sample.id, is_enabled=not sample.is_enabled
                        )
                        record_audit_event(
                            EVENT_SAMPLE_UPDATED,
                            actor_user_id=admin.user_id,
                            actor_username=admin.username,
                            target_type="sample",
                            target_id=sample.sample_id,
                            detail={"is_enabled": not sample.is_enabled},
                        )
                        st.rerun()
                with b2:
                    if st.button("Delete", key=f"admin_sample_delete_{sample.id}"):
                        deleted = admin_samples.delete_sample(sample.id)
                        record_audit_event(
                            EVENT_SAMPLE_DELETED,
                            actor_user_id=admin.user_id,
                            actor_username=admin.username,
                            target_type="sample",
                            target_id=deleted or str(sample.id),
                        )
                        st.session_state.admin_action_notice = "Sample deleted."
                        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------


def _render_experimental_features(admin: AuthenticatedUser) -> None:
    """Local experimental tools — no model quotas or API keys."""
    from shape_detection_storage import (
        ensure_default_feature_policy,
        get_feature_policy,
        upsert_feature_policy,
    )

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Shape Detection",
        "Free local OpenCV circle detection. No Roboflow, OpenRouter, or paid inference.",
    )
    ensure_default_feature_policy()
    policy = get_feature_policy()
    enabled_admins = st.checkbox(
        "Enabled for administrators",
        value=bool(policy.get("enabled_for_admins", True)),
        key="exp_shape_admins",
    )
    enabled_users = st.checkbox(
        "Enabled for regular users",
        value=bool(policy.get("enabled_for_users", True)),
        key="exp_shape_users",
    )
    save_history = st.checkbox(
        "Save history enabled",
        value=bool(policy.get("save_history_enabled", True)),
        key="exp_shape_history",
    )
    max_mb = st.number_input(
        "Maximum image size (MB, 0 = use app default)",
        min_value=0,
        max_value=100,
        value=int((policy.get("max_image_bytes") or 0) // (1024 * 1024)),
        key="exp_shape_max_mb",
    )
    notes = st.text_area(
        "Notes",
        value=str(policy.get("notes") or ""),
        key="exp_shape_notes",
    )
    if st.button("Save Shape Detection policy", type="primary", key="exp_shape_save"):
        upsert_feature_policy(
            enabled_for_admins=bool(enabled_admins),
            enabled_for_users=bool(enabled_users),
            max_image_bytes=(int(max_mb) * 1024 * 1024) if max_mb else None,
            save_history_enabled=bool(save_history),
            notes=notes,
            updated_by=admin.username,
        )
        record_audit_event(
            EVENT_POLICY_UPDATED,
            actor_user_id=admin.user_id,
            actor_username=admin.username,
            target_type="feature_policy",
            target_id="shape_detection",
            detail={
                "enabled_for_admins": enabled_admins,
                "enabled_for_users": enabled_users,
            },
        )
        st.session_state.admin_action_notice = "Shape Detection policy saved."
        st.rerun()
    st.caption(
        "Shape Detection is available to all signed-in users from the left sidebar "
        "(Work in progress). Role toggles below affect save/history policy notes; "
        "they no longer hide the page. "
        "This feature does not use model-access quota counters."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_model_access(admin: AuthenticatedUser) -> None:
    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Model access policies",
        "Choose which roles may select each model and set daily run limits. "
        "OpenRouter models use the admin deployment key — users never see it.",
    )

    model_access.ensure_default_policies()
    policies = list_model_policies()
    if not policies:
        st.caption("No policies are defined.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    from openrouter_store import has_verified_deployment_key

    globally_enabled = model_access.openrouter_globally_enabled()
    if not globally_enabled:
        st.info(
            "OPENROUTER_MODELS_ENABLED is false for this deployment, so OpenRouter "
            "models stay unavailable regardless of the policies below."
        )
    elif not has_verified_deployment_key():
        st.warning(
            "No OpenRouter deployment key is configured yet. Add one under "
            "Connectivity or API Keys before enabling OpenRouter models for users."
        )

    for policy in policies:
        label = policy.display_name or policy.model_key
        with st.expander(label, expanded=False):
            st.caption(f"Model key: `{policy.model_key}`")
            with st.form(f"admin_policy_{policy.model_key}"):
                cols = st.columns(2)
                with cols[0]:
                    enabled = st.checkbox("Enabled for selected roles", value=policy.is_enabled)
                    roles = st.multiselect(
                        "Allowed roles",
                        list(ROLES),
                        default=[r for r in policy.allowed_roles if r in ROLES]
                        or [ROLE_ADMIN],
                    )
                with cols[1]:
                    limit = st.number_input(
                        "Maximum runs per user per day (0 = unlimited)",
                        min_value=0,
                        value=int(policy.maximum_runs_per_user_per_day or 0),
                        step=1,
                    )
                notes = st.text_area("Notes", value=policy.notes)
                saved = st.form_submit_button("Save policy", type="primary")

            if saved:
                if not roles:
                    st.session_state.admin_action_error = (
                        "Select at least one role, or disable the model instead."
                    )
                    st.rerun()
                upsert_model_policy(
                    policy.model_key,
                    display_name=policy.display_name or label,
                    is_enabled=enabled,
                    allowed_roles=tuple(roles),
                    requires_user_api_key=False,
                    requires_cost_confirmation=False,
                    maximum_runs_per_user_per_day=int(limit) if limit else 0,
                    notes=notes,
                    updated_by=admin.username,
                )
                record_audit_event(
                    EVENT_POLICY_UPDATED,
                    actor_user_id=admin.user_id,
                    actor_username=admin.username,
                    target_type="model_policy",
                    target_id=policy.model_key,
                    detail={
                        "is_enabled": enabled,
                        "allowed_roles": roles,
                        "maximum_runs_per_user_per_day": int(limit) or None,
                    },
                )
                st.session_state.admin_action_notice = f"Policy for {label} saved."
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("Usage against quotas", "Runs recorded in the last 30 days.")
    usage = get_usage_summary(days=30)
    if usage:
        st.dataframe(pd.DataFrame(usage), width="stretch", hide_index=True)
    else:
        st.caption("No usage recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def _render_connectivity(admin: AuthenticatedUser) -> None:
    from poc_ux import (
        connection_status_payload,
        render_connection_light_html,
        resolve_connection_label,
    )

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Roboflow",
        "Checked automatically when you open this tab. Retest after key changes.",
    )
    # Auto-check when this tab opens if nothing fresh is cached.
    from roboflow_status import ensure_roboflow_probe

    probe = ensure_roboflow_probe(force=False)
    api_ok = bool(config.api_key_configured() or config.DEMO_MODE)
    label = resolve_connection_label(api_configured=api_ok, last_probe=probe)
    st.markdown(
        render_connection_light_html(
            label,
            auth_ok=probe.get("auth_ok") if isinstance(probe, dict) else None,
            detail=str((probe or {}).get("message") or "")[:140],
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        f"API URL: `{config.ROBOFLOW_API_URL}` · "
        f"Workspace: `{config.ROBOFLOW_WORKSPACE}` · "
        f"Workflow: `{config.ROBOFLOW_WORKFLOW_ID}`"
    )

    if st.button("Retest Roboflow", key="admin_conn_roboflow"):
        ensure_roboflow_probe(force=True)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("OpenRouter", "Deployment key status for vision models.")
    from openrouter_store import get_deployment_key_status

    or_status = get_deployment_key_status()
    if or_status.configured and or_status.verified:
        st.success(
            f"Deployment key configured ({or_status.masked or 'masked'}). "
            "Users never see this key."
        )
    else:
        st.warning(
            "No OpenRouter key configured. Add one on the API Keys page "
            "(administrators only), then enable models under Model Access."
        )
    if st.button("Open API Keys", key="admin_conn_open_api_keys"):
        st.session_state.app_view = "api_keys"
        st.rerun()
    st.caption(
        "Models globally enabled: "
        + ("yes" if model_access.openrouter_globally_enabled() else "no")
    )
    st.caption(f"Workflow: {getattr(config, 'OPENROUTER_WORKFLOW_ID', '—')}")
    st.caption(f"Key verification endpoint: {getattr(config, 'OPENROUTER_KEY_VERIFY_URL', '—')}")

    # Compact technical snapshot from the auto probe (no secrets).
    if isinstance(probe, dict) and probe:
        with st.expander("Last Roboflow check details", expanded=False):
            st.json(
                redact_secrets(
                    connection_status_payload(
                        api_configured=api_ok,
                        workspace=config.ROBOFLOW_WORKSPACE,
                        workflow_available=bool(config.ROBOFLOW_WORKFLOW_ID),
                        validated_model_count=0,
                        last_probe=probe,
                    )
                )
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _render_audit_log(admin: AuthenticatedUser) -> None:
    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section(
        "Audit log",
        "Every entry is redacted before storage — passwords and API keys are never written here.",
    )

    types = ["(all)"] + list_audit_event_types()
    cols = st.columns(3)
    with cols[0]:
        event_type = st.selectbox("Event type", types, key="admin_audit_filter_type")
    with cols[1]:
        outcome = st.selectbox(
            "Outcome", ["(all)", "success", "failure"], key="admin_audit_filter_outcome"
        )
    with cols[2]:
        limit = st.number_input(
            "Rows", min_value=25, max_value=1000, value=200, step=25, key="admin_audit_limit"
        )

    events = get_audit_events(
        limit=int(limit),
        event_type=None if event_type == "(all)" else event_type,
        outcome=None if outcome == "(all)" else outcome,
    )
    if not events:
        st.caption("No audit events match these filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    frame = pd.DataFrame(
        [
            {
                "When": str(e.get("created_at") or "")[:19].replace("T", " "),
                "Actor": e.get("actor_username") or "system",
                "Event": e.get("event_type"),
                "Target": e.get("target_id") or "",
                "Outcome": e.get("outcome"),
                "Detail": e.get("detail") or "",
            }
            for e in events
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        "Download audit log (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="audit_log.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Storage and system
# ---------------------------------------------------------------------------


def _render_storage_and_system(admin: AuthenticatedUser) -> None:
    del admin
    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("Storage", "Paths used by this deployment (no secrets).")
    st.warning(
        "**Streamlit Community Cloud uses ephemeral storage.** The SQLite "
        "database, uploaded samples and audit log live on the container's local "
        "disk and are lost whenever the app is redeployed, restarted or put to "
        "sleep. Treat this deployment as a proof of concept: export anything you "
        "need to keep, and expect to re-run the administrator bootstrap after a "
        "redeploy."
    )

    data_dir = Path(config.DATA_DIR)
    rows = []
    for path in (
        config.DB_PATH,
        data_dir / "admin_samples",
        data_dir / "benchmarks.json",
        data_dir / "model_catalog.json",
    ):
        try:
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.glob("*") if f.is_file())
                rows.append({"Path": path.name, "Type": "directory", "Size (KB)": size // 1024})
            elif path.exists():
                rows.append(
                    {
                        "Path": path.name,
                        "Type": "file",
                        "Size (KB)": path.stat().st_size // 1024,
                    }
                )
            else:
                rows.append({"Path": path.name, "Type": "missing", "Size (KB)": 0})
        except OSError:
            rows.append({"Path": path.name, "Type": "unreadable", "Size (KB)": 0})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"Data directory: `{data_dir}`")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("System", "Runtime and session policy.")
    s1, s2 = st.columns(2)
    with s1:
        st.caption(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
        st.caption(f"Streamlit {st.__version__}")
        st.caption(f"Database schema version: {get_schema_version()}")
    with s2:
        st.caption(f"Demo mode: {'on' if config.DEMO_MODE else 'off'}")
        st.caption(
            "Session policy: "
            f"{config.SESSION_IDLE_TIMEOUT_MINUTES} min idle / "
            f"{config.SESSION_ABSOLUTE_TIMEOUT_HOURS} h absolute"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="aic-admin-panel">', unsafe_allow_html=True)
    _admin_section("Configuration snapshot", "Sanitized deployment settings.")
    snapshot = {
        "data_dir": str(config.DATA_DIR),
        "roboflow_api_url": config.ROBOFLOW_API_URL,
        "roboflow_workspace": config.ROBOFLOW_WORKSPACE,
        "roboflow_key_configured": config.api_key_configured(),
        "openrouter_models_enabled": model_access.openrouter_globally_enabled(),
        "openrouter_workflow_id": getattr(config, "OPENROUTER_WORKFLOW_ID", ""),
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "session_idle_timeout_minutes": config.SESSION_IDLE_TIMEOUT_MINUTES,
        "session_absolute_timeout_hours": config.SESSION_ABSOLUTE_TIMEOUT_HOURS,
    }
    st.code(json.dumps(redact_secrets(snapshot), indent=2), language="json")
    st.markdown("</div>", unsafe_allow_html=True)
