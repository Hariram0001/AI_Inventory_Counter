"""Administrator console: users, samples, model access, connectivity, audit."""

from __future__ import annotations

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
    ROLES,
    UserStoreError,
    count_active_admins,
    delete_user,
    get_audit_events,
    get_usage_summary,
    get_user_by_id,
    list_audit_event_types,
    list_model_policies,
    list_users,
    lock_remaining_seconds,
    record_audit_event,
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
    "Connectivity",
    "Audit Log",
    "Storage and System",
)


def render_admin_console(user: AuthenticatedUser) -> None:
    """Entry point. Callers must already have verified the administrator role."""
    if not user.is_admin:
        st.error("You do not have permission to view this page.")
        return

    st.markdown("### Administrator console")
    st.caption(
        "Manage accounts, model access policies, demo samples and connectivity "
        "for this deployment."
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
        _render_connectivity(user)
    with tabs[5]:
        _render_audit_log(user)
    with tabs[6]:
        _render_storage_and_system(user)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _render_overview(user: AuthenticatedUser) -> None:
    users = list_users()
    active = [u for u in users if u.is_active]
    locked = [u for u in users if u.is_locked()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", len(users))
    c2.metric("Active", len(active))
    c3.metric("Administrators", sum(1 for u in active if u.is_admin))
    c4.metric("Locked", len(locked))

    d1, d2, d3 = st.columns(3)
    try:
        d1.metric("Saved counts", count_inventory_rows())
    except Exception:  # noqa: BLE001
        d1.metric("Saved counts", "—")
    d2.metric("Samples", len(admin_samples.list_samples()))
    d3.metric("Schema version", get_schema_version())

    st.divider()
    st.markdown("#### OpenRouter usage (last 7 days)")
    usage = get_usage_summary(days=7)
    if usage:
        st.dataframe(pd.DataFrame(usage), width="stretch", hide_index=True)
    else:
        st.caption("No model runs have been recorded in the last 7 days.")

    st.divider()
    st.markdown("#### Recent activity")
    events = get_audit_events(limit=10)
    if events:
        for event in events:
            when = str(event.get("created_at") or "")[:19].replace("T", " ")
            actor = event.get("actor_username") or "system"
            outcome = event.get("outcome") or "success"
            st.caption(f"{when} · {actor} · {event.get('event_type')} · {outcome}")
    else:
        st.caption("No audit events recorded yet.")


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def _render_users(admin: AuthenticatedUser) -> None:
    st.markdown("#### Create a user")
    with st.form("admin_create_user_form", clear_on_submit=True):
        cols = st.columns(2)
        with cols[0]:
            username = st.text_input("Username", key="admin_user_new_username")
            role = st.selectbox("Role", ROLES, index=1, key="admin_user_new_role")
        with cols[1]:
            display_name = st.text_input("Display name", key="admin_user_new_display")
            email = st.text_input("Email (optional)", key="admin_user_new_email")
        st.caption(
            "A temporary password is generated and shown once. The user must "
            "change it at first sign-in."
        )
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

    st.divider()
    st.markdown("#### Manage users")
    users = list_users()
    if not users:
        st.caption("No users exist yet.")
        return

    rows = []
    for record in users:
        remaining = lock_remaining_seconds(record.locked_until)
        rows.append(
            {
                "Username": record.username,
                "Name": record.display_name,
                "Role": record.role,
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
    st.markdown("#### Upload a sample image")
    st.caption(
        "Uploaded samples appear on the user dashboard for one-click demos. "
        "Files are validated by decoding them, not by their declared type."
    )

    inventory_options = ["Fence Panel"] + [
        key for key in getattr(config, "INVENTORY_TYPES", []) if key != "Fence Panel"
    ]

    with st.form("admin_sample_upload_form", clear_on_submit=True):
        upload = st.file_uploader(
            "Image file", type=["jpg", "jpeg", "png"], key="admin_sample_file"
        )
        cols = st.columns(2)
        with cols[0]:
            title = st.text_input("Title", key="admin_sample_title")
            inventory_type = st.selectbox(
                "Inventory type", inventory_options, key="admin_sample_inventory"
            )
        with cols[1]:
            expected = st.number_input(
                "Expected count (optional)",
                min_value=0,
                value=0,
                step=1,
                key="admin_sample_expected",
            )
            enabled = st.checkbox("Enabled", value=True, key="admin_sample_enabled")
        description = st.text_area("Description", key="admin_sample_description")
        submitted = st.form_submit_button("Upload sample", type="primary")

    if submitted:
        if upload is None:
            st.session_state.admin_action_error = "Choose an image file to upload."
            st.rerun()
        try:
            sample = admin_samples.add_sample(
                data=upload.getvalue(),
                title=title,
                inventory_type=inventory_type,
                description=description,
                expected_count=int(expected) if expected else None,
                uploaded_by=admin.username,
                is_enabled=bool(enabled),
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

    st.divider()
    st.markdown("#### Manage samples")
    samples = admin_samples.list_samples()
    if not samples:
        st.caption("No administrator samples have been uploaded yet.")
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


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------


def _render_model_access(admin: AuthenticatedUser) -> None:
    st.markdown("#### Model access policies")
    st.caption(
        "Policies decide which roles may select a model, whether the user must "
        "supply their own API key, and how many runs each user gets per day."
    )

    model_access.ensure_default_policies()
    policies = list_model_policies()
    if not policies:
        st.caption("No policies are defined.")
        return

    globally_enabled = model_access.openrouter_globally_enabled()
    if not globally_enabled:
        st.info(
            "OPENROUTER_MODELS_ENABLED is false for this deployment, so OpenRouter "
            "models stay unavailable regardless of the policies below."
        )

    for policy in policies:
        label = policy.display_name or policy.model_key
        with st.expander(label, expanded=False):
            st.caption(f"Model key: `{policy.model_key}`")
            with st.form(f"admin_policy_{policy.model_key}"):
                cols = st.columns(2)
                with cols[0]:
                    enabled = st.checkbox("Enabled", value=policy.is_enabled)
                    roles = st.multiselect(
                        "Allowed roles",
                        list(ROLES),
                        default=[r for r in policy.allowed_roles if r in ROLES]
                        or [ROLE_ADMIN],
                    )
                    requires_key = st.checkbox(
                        "Requires the user's own API key",
                        value=policy.requires_user_api_key,
                    )
                with cols[1]:
                    requires_cost = st.checkbox(
                        "Requires cost confirmation",
                        value=policy.requires_cost_confirmation,
                    )
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
                    requires_user_api_key=requires_key,
                    requires_cost_confirmation=requires_cost,
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
                        "requires_user_api_key": requires_key,
                        "requires_cost_confirmation": requires_cost,
                        "maximum_runs_per_user_per_day": int(limit) or None,
                    },
                )
                st.session_state.admin_action_notice = f"Policy for {label} saved."
                st.rerun()

    st.divider()
    st.markdown("#### Usage against quotas (last 30 days)")
    usage = get_usage_summary(days=30)
    if usage:
        st.dataframe(pd.DataFrame(usage), width="stretch", hide_index=True)
    else:
        st.caption("No usage recorded yet.")


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


def _render_connectivity(admin: AuthenticatedUser) -> None:
    st.markdown("#### Roboflow")
    if config.api_key_configured():
        st.success("A Roboflow API key is configured for this deployment.")
    else:
        st.error(
            "No Roboflow API key is configured. Set ROBOFLOW_API_KEY in the "
            "environment or Streamlit secrets."
        )
    st.caption(f"API URL: {config.ROBOFLOW_API_URL}")
    st.caption(f"Workspace: {config.ROBOFLOW_WORKSPACE}")
    st.caption(f"Default workflow: {config.ROBOFLOW_WORKFLOW_ID}")

    if st.button("Run Roboflow connectivity test", key="admin_conn_roboflow"):
        st.session_state.admin_connectivity_result = _test_roboflow()
        st.rerun()

    st.divider()
    st.markdown("#### OpenRouter")
    st.caption(
        "OpenRouter runs use each user's own key, so this deployment holds no "
        "OpenRouter credentials. There is nothing to test here without a user key."
    )
    st.caption(
        "Models enabled: "
        + ("yes" if model_access.openrouter_globally_enabled() else "no")
    )
    st.caption(f"Workflow: {getattr(config, 'OPENROUTER_WORKFLOW_ID', '—')}")
    st.caption(f"Key verification endpoint: {getattr(config, 'OPENROUTER_KEY_VERIFY_URL', '—')}")

    result = st.session_state.get("admin_connectivity_result")
    if result:
        st.divider()
        st.markdown("#### Last test result")
        if result.get("ok"):
            st.success(result.get("message", "Connected."))
        else:
            st.error(result.get("message", "Connection failed."))
        with st.expander("Technical details", expanded=False):
            st.json(redact_secrets(result))


def _test_roboflow() -> dict[str, Any]:
    """Live connectivity probe. Errors are redacted before display."""
    try:
        from detector import RoboflowDetector

        detector = RoboflowDetector()
        ok, message = detector.test_connectivity()
        return {
            "ok": bool(ok),
            "message": message,
            "api_url": config.ROBOFLOW_API_URL,
            "workspace": config.ROBOFLOW_WORKSPACE,
        }
    except Exception as exc:  # noqa: BLE001
        from detector import sanitize_exception_text

        return {
            "ok": False,
            "message": "Roboflow connectivity test failed.",
            "error": sanitize_exception_text(f"{type(exc).__name__}: {exc}"),
        }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def _render_audit_log(admin: AuthenticatedUser) -> None:
    st.markdown("#### Audit log")
    st.caption(
        "Every entry is redacted before storage — passwords and API keys are "
        "never written here."
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


# ---------------------------------------------------------------------------
# Storage and system
# ---------------------------------------------------------------------------


def _render_storage_and_system(admin: AuthenticatedUser) -> None:
    st.markdown("#### Storage")
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
    st.caption(f"Data directory: {data_dir}")

    st.divider()
    st.markdown("#### System")
    st.caption(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}")
    st.caption(f"Streamlit {st.__version__}")
    st.caption(f"Database schema version: {get_schema_version()}")
    st.caption(f"Demo mode: {'on' if config.DEMO_MODE else 'off'}")
    st.caption(
        "Session policy: "
        f"{config.SESSION_IDLE_TIMEOUT_MINUTES} minutes idle, "
        f"{config.SESSION_ABSOLUTE_TIMEOUT_HOURS} hours absolute."
    )

    st.divider()
    st.markdown("#### Configuration snapshot")
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
