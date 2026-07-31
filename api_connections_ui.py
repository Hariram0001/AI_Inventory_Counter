"""User API Connections page — session-only OpenRouter key management."""

from __future__ import annotations

import streamlit as st

import auth_session
import config
from auth import (
    EVENT_COST_ACKNOWLEDGED,
    EVENT_KEY_REMOVED,
    EVENT_KEY_VERIFIED,
    EVENT_KEY_VERIFY_FAILED,
    AuthenticatedUser,
)
from openrouter import (
    COST_NOTICE_BODY,
    COST_NOTICE_POINTS,
    COST_NOTICE_TITLE,
    verify_api_key,
)
from user_store import record_audit_event


def render_api_connections_page(user: AuthenticatedUser) -> None:
    st.markdown("### API Connections")
    st.caption(
        "Connect your own OpenRouter account to use vision language models. "
        "Your key is held for this browser session only and is never saved to "
        "disk, logs or your inventory history."
    )

    _render_roboflow_status()
    st.divider()
    _render_openrouter_section(user)
    st.divider()
    _render_cost_notice(user)


def _render_roboflow_status() -> None:
    st.markdown("#### Roboflow (managed by your administrator)")
    if config.api_key_configured():
        st.success(
            "A shared Roboflow key is configured for this deployment. "
            "You do not need to supply one."
        )
    else:
        st.warning(
            "No Roboflow key is configured for this deployment. Roboflow-backed "
            "models are unavailable until an administrator adds one."
        )


def _render_openrouter_section(user: AuthenticatedUser) -> None:
    st.markdown("#### OpenRouter (your own key)")

    if not getattr(config, "OPENROUTER_MODELS_ENABLED", True):
        st.info(
            "OpenRouter models are disabled for this deployment. Adding a key "
            "will not make them available."
        )

    status = auth_session.get_openrouter_key_status()
    if auth_session.has_verified_openrouter_key():
        st.success(
            f"Verified key **{status.get('masked') or '—'}** is active for this session."
        )
        label = status.get("label")
        if label:
            st.caption(f"OpenRouter key label: {label}")
        limit = status.get("credit_limit")
        usage = status.get("usage")
        if limit is not None:
            remaining = max(0.0, float(limit) - float(usage or 0.0))
            st.caption(f"Approximate remaining credit: ${remaining:.2f}")
        elif usage is not None:
            st.caption(f"Usage reported by OpenRouter: ${float(usage):.2f}")
        st.caption(auth_session.session_timeout_summary())

        if st.button("Remove key", key="byok_remove", width="stretch"):
            auth_session.remove_openrouter_key()
            record_audit_event(
                EVENT_KEY_REMOVED,
                actor_user_id=user.user_id,
                actor_username=user.username,
                target_type="api_key",
                target_id="openrouter",
                detail={"scope": "session"},
            )
            st.rerun()
        return

    st.caption(
        "Create a key at openrouter.ai/keys. Verification uses OpenRouter's free "
        "key endpoint and does not incur any charge."
    )
    with st.form("openrouter_key_form", clear_on_submit=True):
        api_key = st.text_input(
            "OpenRouter API key",
            type="password",
            key="byok_key_input",
            placeholder="sk-or-v1-…",
            autocomplete="off",
        )
        submitted = st.form_submit_button(
            "Verify key", type="primary", width="stretch"
        )

    if submitted:
        _handle_verify(user, api_key)

    error = st.session_state.pop("byok_verify_error", "")
    if error:
        st.error(error)


def _handle_verify(user: AuthenticatedUser, api_key: str) -> None:
    verification = verify_api_key(api_key)

    if verification.verified:
        auth_session.set_openrouter_key(api_key, verification.to_public_dict())
        record_audit_event(
            EVENT_KEY_VERIFIED,
            actor_user_id=user.user_id,
            actor_username=user.username,
            target_type="api_key",
            target_id="openrouter",
            # Only the sanitized summary is recorded; never the key itself.
            detail={
                "masked": verification.masked_key,
                "label": verification.label,
                "is_free_tier": verification.is_free_tier,
            },
        )
        st.rerun()
        return

    st.session_state.byok_verify_error = verification.message
    record_audit_event(
        EVENT_KEY_VERIFY_FAILED,
        actor_user_id=user.user_id,
        actor_username=user.username,
        target_type="api_key",
        target_id="openrouter",
        outcome="failure",
        detail={"status": verification.status},
    )


def _render_cost_notice(user: AuthenticatedUser) -> None:
    st.markdown(f"#### {COST_NOTICE_TITLE}")
    st.warning(COST_NOTICE_BODY)
    for point in COST_NOTICE_POINTS:
        st.markdown(f"- {point}")

    if auth_session.has_accepted_cost_notice():
        st.success("You accepted this notice for the current session.")
        return

    accepted = st.checkbox(
        "I understand that OpenRouter runs are billed to my own account.",
        key="byok_cost_ack_checkbox",
    )
    if st.button(
        "Accept and enable OpenRouter models",
        key="byok_cost_ack",
        type="primary",
        disabled=not accepted,
        width="stretch",
    ):
        auth_session.accept_cost_notice()
        record_audit_event(
            EVENT_COST_ACKNOWLEDGED,
            actor_user_id=user.user_id,
            actor_username=user.username,
            target_type="policy",
            target_id="openrouter_cost_notice",
        )
        st.rerun()


def render_inline_cost_gate(user: AuthenticatedUser) -> bool:
    """Blocking cost confirmation shown before the first paid run.

    Returns True when the user has accepted and the run may proceed.
    """
    if auth_session.has_accepted_cost_notice():
        return True

    st.warning(f"**{COST_NOTICE_TITLE}**\n\n{COST_NOTICE_BODY}")
    accepted = st.checkbox(
        "I understand that this run is billed to my own OpenRouter account.",
        key="byok_inline_cost_ack_checkbox",
    )
    if st.button(
        "Accept and continue",
        key="byok_inline_cost_ack",
        type="primary",
        disabled=not accepted,
    ):
        auth_session.accept_cost_notice()
        record_audit_event(
            EVENT_COST_ACKNOWLEDGED,
            actor_user_id=user.user_id,
            actor_username=user.username,
            target_type="policy",
            target_id="openrouter_cost_notice",
            detail={"surface": "inline_run_gate"},
        )
        st.rerun()
    return False
