"""Administrator OpenRouter key management (never shown to regular users)."""

from __future__ import annotations

import streamlit as st

import config
from auth import (
    EVENT_KEY_REMOVED,
    EVENT_KEY_VERIFIED,
    EVENT_KEY_VERIFY_FAILED,
    AuthenticatedUser,
)
from openrouter import verify_api_key
from openrouter_store import (
    clear_deployment_key,
    get_deployment_key_status,
    save_deployment_key,
)
from user_store import record_audit_event


def render_api_connections_page(user: AuthenticatedUser) -> None:
    if not user.is_admin:
        st.error("Only administrators can manage OpenRouter credentials.")
        return

    _render_roboflow_status()
    st.divider()
    _render_openrouter_section(user)


def _render_roboflow_status() -> None:
    from poc_ux import render_connection_light_html, resolve_connection_label
    from roboflow_status import ensure_roboflow_probe

    st.markdown("#### Roboflow")
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
    if not api_ok:
        st.caption(
            "Set ROBOFLOW_API_KEY in the environment or Streamlit secrets."
        )


def _render_openrouter_section(user: AuthenticatedUser) -> None:
    st.markdown("#### OpenRouter")

    if not getattr(config, "OPENROUTER_MODELS_ENABLED", True):
        st.info(
            "OPENROUTER_MODELS_ENABLED is false. Adding a key will not make "
            "OpenRouter models available until that setting is turned on."
        )

    status = get_deployment_key_status()
    if status.configured and status.verified:
        st.success(
            f"Deployment key **{status.masked or '—'}** is configured."
        )
        if status.label:
            st.caption(f"OpenRouter key label: {status.label}")
        if status.updated_by:
            when = (status.updated_at or "")[:19].replace("T", " ")
            st.caption(f"Last updated by {status.updated_by} at {when or '—'}")
        if st.button("Remove OpenRouter key", key="admin_or_remove", width="stretch"):
            clear_deployment_key(updated_by=user.username)
            record_audit_event(
                EVENT_KEY_REMOVED,
                actor_user_id=user.user_id,
                actor_username=user.username,
                target_type="api_key",
                target_id="openrouter",
                detail={"scope": "deployment", "masked": status.masked},
            )
            st.rerun()
        return

    st.caption(
        "Paste an OpenRouter inference key (`sk-or-…`). Verification is free; "
        "enable models afterward under Admin Console → Model Access."
    )
    with st.form("admin_openrouter_key_form", clear_on_submit=True):
        api_key = st.text_input(
            "OpenRouter API key",
            type="password",
            key="admin_or_key_input",
            placeholder="sk-or-v1-…",
            autocomplete="off",
        )
        submitted = st.form_submit_button(
            "Verify and save key", type="primary", width="stretch"
        )

    if submitted:
        _handle_verify(user, api_key)

    error = st.session_state.pop("admin_or_verify_error", "")
    if error:
        st.error(error)


def _handle_verify(user: AuthenticatedUser, api_key: str) -> None:
    verification = verify_api_key(api_key)

    if verification.verified:
        save_deployment_key(
            api_key,
            verification=verification.to_public_dict(),
            updated_by=user.username,
        )
        from openrouter_runtime import clear_stale_credential_test_state

        # Re-verification clears stale credential failures → Ready to test.
        # Does not mark the model live-validated.
        clear_stale_credential_test_state()
        record_audit_event(
            EVENT_KEY_VERIFIED,
            actor_user_id=user.user_id,
            actor_username=user.username,
            target_type="api_key",
            target_id="openrouter",
            detail={
                "scope": "deployment",
                "masked": verification.masked_key,
                "label": verification.label,
                "is_free_tier": verification.is_free_tier,
            },
        )
        st.rerun()
        return

    st.session_state.admin_or_verify_error = verification.message
    record_audit_event(
        EVENT_KEY_VERIFY_FAILED,
        actor_user_id=user.user_id,
        actor_username=user.username,
        target_type="api_key",
        target_id="openrouter",
        outcome="failure",
        detail={"scope": "deployment", "status": verification.status},
    )
