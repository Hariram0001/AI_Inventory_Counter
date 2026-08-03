"""Streamlit session wiring for authentication, state isolation and BYOK keys.

Session state is grouped into named namespaces so logout, timeout and user
switches can clear exactly the right slice without disturbing the rest.

Each browser Streamlit session is independent — many users can be signed in
at once. Identity never lives in the URL or in process-global mutable state.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Iterable

import config
from auth import (
    EVENT_LOGOUT,
    EVENT_SESSION_INVALIDATED,
    EVENT_SESSION_TIMEOUT,
    AuthenticatedUser,
    evaluate_session_expiry,
    get_auth_provider,
)
from security import mask_secret
from user_store import record_audit_event, touch_last_activity

# --- session state key namespaces -----------------------------------------

AUTH_KEYS: tuple[str, ...] = (
    "auth_user",
    "auth_last_activity",
    "auth_logout_notice",
    "auth_login_error",
    "auth_pending_username",
    "auth_signup_notice",
    "auth_reset_notice",
)

BYOK_KEYS: tuple[str, ...] = (
    "openrouter_api_key",
    "openrouter_key_status",
    "openrouter_cost_ack",
)

WIZARD_KEYS: tuple[str, ...] = (
    "form",
    "uploaded_images",
    "pending_camera",
    "analysis_status",
    "analysis_results",
    "analysis_failures",
    "analysis_meta",
    "analysis_run_id",
    "analyze_running",
    "_analysis_executing",
    "consensus_result",
    "comparison_summaries",
    "accepted_result_key",
    "review_state",
    "review_edits",
    "save_status",
    "saved_record",
    "pending_review_payload",
    "selected_photo_index",
    "selected_detection_id",
    "review_active_image",
    "review_active_model",
    "inference_cache",
    "model_test_results",
    "model_trial_rows",
    "model_trial_suggestion",
    "session_models",
    "run_context",
    "sample_selected_ids",
    "sample_preview_id",
    "sample_gallery_page",
    "selected_photos_page",
    "compare_side_by_side",
)

BENCHMARK_KEYS: tuple[str, ...] = (
    "benchmark_outcomes",
    "benchmark_active_idx",
    "benchmark_meta",
    "benchmark_promote_choice",
    "benchmark_mode",
    "benchmark_expected_count",
    "benchmark_expected_prefill",
    "batch_image_bytes",
    "batch_annotated",
    "batch_session",
    "batch_progress",
    "batch_cancel",
    "batch_run_cache",
    "batch_force_rerun",
    "batch_gt_edits",
    "batch_specs",
)

ADMIN_KEYS: tuple[str, ...] = (
    "admin_tab",
    "admin_selected_user_id",
    "admin_action_notice",
    "admin_action_error",
    "admin_temp_password",
    "admin_connectivity_result",
    "admin_sample_notice",
    "admin_policy_notice",
    "admin_audit_filter_type",
    "admin_audit_filter_outcome",
)

TRANSIENT_UI_KEYS: tuple[str, ...] = (
    "connection_probe",
    "connection_probe_flash",
    "ai_config_test_result",
    "ai_config_test_image_bytes",
    "ai_config_test_image_name",
    "last_diag_error",
    "diag_dyn_report",
    "diag_dyn_annotated",
    "demo_sample_id",
    "catalog_test_result",
    "catalog_pending_test",
    "catalog_last_sync",
    "sample_clear_pending",
)

SHAPE_DETECTION_PREFIX = "shape_detection"

# Widget key prefixes whose values must not survive a user switch.
_TRANSIENT_PREFIXES: tuple[str, ...] = (
    "sample_sel_",
    "prompt_",
    "benchmark_image_",
    "login_",  # never retain credentials across logout / account switch
    "signup_",
    "reset_username",
    "pwchange_",
    "admin_user_",
    "shape_detection",
)

LOGIN_CREDENTIAL_KEYS: tuple[str, ...] = (
    "login_username",
    "login_password",
)

# Extra session keys that must never survive logout / account switch.
_EXTRA_USER_SCOPED_KEYS: tuple[str, ...] = (
    "openrouter_session_key_rejected",
    "openrouter_user_model_test_state",
    "catalog_selected_key",
    "catalog_pending_test",
    "catalog_test_result",
    "catalog_last_sync",
    "annotation_style",
    "photo_source_mode",
    "uploader_nonce",
    "settings_section",
    "previous_page",
    "previous_wizard_step",
    "rev_show_all_markers",
    "open_advanced_settings",
    "auth_session_id",
    "_auth_activity_written",
)

# Keys that may survive an identity change (preferences only — never auth).
_IDENTITY_SURVIVORS: frozenset[str] = frozenset({"ui_theme"})

AUTH_SESSION_ID_KEY = "auth_session_id"
AUTH_SESSION_QUERY_KEY = "sid"


def _st():
    import streamlit as st

    return st


def _drop(keys: Iterable[str]) -> None:
    st = _st()
    for key in keys:
        st.session_state.pop(key, None)


def _drop_prefixed(prefixes: Iterable[str]) -> None:
    st = _st()
    for key in list(st.session_state.keys()):
        text = str(key)
        if any(text.startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def clear_wizard_state() -> None:
    _drop(WIZARD_KEYS)


def clear_benchmark_state() -> None:
    _drop(BENCHMARK_KEYS)


def clear_admin_state() -> None:
    _drop(ADMIN_KEYS)


def clear_transient_ui_state() -> None:
    _drop(TRANSIENT_UI_KEYS)


def clear_byok_state() -> None:
    """Remove the session-only OpenRouter key and its derived flags."""
    _drop(BYOK_KEYS)


def clear_auth_state() -> None:
    _drop(AUTH_KEYS)


def clear_login_credentials() -> None:
    """Drop username/password widget values so they cannot leak across accounts."""
    _drop(LOGIN_CREDENTIAL_KEYS)


def clear_shape_detection_state() -> None:
    """Clear Shape Detection session keys without touching auth or OpenRouter."""
    _drop_prefixed((SHAPE_DETECTION_PREFIX,))


def clear_user_scoped_state() -> None:
    """Everything that belongs to one signed-in user."""
    clear_wizard_state()
    clear_benchmark_state()
    clear_admin_state()
    clear_transient_ui_state()
    clear_byok_state()
    clear_shape_detection_state()
    _drop_prefixed(_TRANSIENT_PREFIXES)
    _drop(_EXTRA_USER_SCOPED_KEYS)


def wipe_session_for_identity_change() -> None:
    """Clear nearly all session_state so no prior account data can leak.

    Survives only intentional preferences (currently ``ui_theme``). Streamlit
    creates a separate server session per browser connection, so many users can
    be signed in concurrently — this wipe only affects the current connection.
    """
    st = _st()
    for key in list(st.session_state.keys()):
        if key in _IDENTITY_SURVIVORS:
            continue
        st.session_state.pop(key, None)
    clear_login_credentials()
    _clear_session_binding_from_url()


def _set_session_binding(session_id: str) -> None:
    st = _st()
    st.session_state[AUTH_SESSION_ID_KEY] = session_id
    try:
        st.query_params[AUTH_SESSION_QUERY_KEY] = session_id
    except Exception:  # noqa: BLE001
        pass


def _clear_session_binding_from_url() -> None:
    st = _st()
    st.session_state.pop(AUTH_SESSION_ID_KEY, None)
    try:
        params = st.query_params
        if AUTH_SESSION_QUERY_KEY in params:
            del params[AUTH_SESSION_QUERY_KEY]
    except Exception:  # noqa: BLE001
        pass


def _query_session_id() -> str | None:
    st = _st()
    try:
        raw = st.query_params.get(AUTH_SESSION_QUERY_KEY)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    text = str(raw or "").strip()
    return text or None


def verify_session_binding() -> bool:
    """Reject a foreign ``?sid=`` pasted into an already-authenticated tab.

    Missing ``sid`` is re-stamped (Streamlit can drop query params). A different
    ``sid`` means the URL belongs to another session — force sign-out.
    """
    st = _st()
    expected = str(st.session_state.get(AUTH_SESSION_ID_KEY) or "").strip()
    if not expected:
        return True
    url_sid = _query_session_id()
    if not url_sid:
        _set_session_binding(expected)
        return True
    return url_sid == expected


# --- identity --------------------------------------------------------------


def current_user() -> AuthenticatedUser | None:
    st = _st()
    user = st.session_state.get("auth_user")
    return user if isinstance(user, AuthenticatedUser) else None


def is_authenticated() -> bool:
    return current_user() is not None


def is_admin() -> bool:
    user = current_user()
    return bool(user and user.is_admin)


def start_session(user: AuthenticatedUser) -> None:
    """Install a freshly authenticated identity, clearing any prior user state."""
    st = _st()
    wipe_session_for_identity_change()
    st.session_state.auth_user = user
    st.session_state.auth_last_activity = datetime.now(timezone.utc).isoformat()
    session_id = secrets.token_urlsafe(24)
    _set_session_binding(session_id)
    # Everyone lands on Home; admins open the console from the sidebar.
    st.session_state.app_view = "welcome"
    st.session_state.wizard_stage = "setup"


def end_session(
    *,
    reason: str = "logout",
    notice: str = "",
    audit: bool = True,
) -> None:
    """Clear all user state. Sensitive material never survives this call."""
    st = _st()
    user = current_user()
    if audit and user is not None:
        event = {
            "logout": EVENT_LOGOUT,
            "timeout": EVENT_SESSION_TIMEOUT,
            "revoked": EVENT_SESSION_INVALIDATED,
        }.get(reason, EVENT_LOGOUT)
        record_audit_event(
            event,
            actor_user_id=user.user_id,
            actor_username=user.username,
            target_type="session",
            target_id=user.username,
            detail={"reason": reason},
        )
    wipe_session_for_identity_change()
    st.session_state.app_view = "welcome"
    st.session_state.wizard_stage = "setup"
    if notice:
        st.session_state.auth_logout_notice = notice


def enforce_session() -> AuthenticatedUser | None:
    """Validate the session on every rerun.

    Applies idle and absolute timeouts, then re-reads the user so deactivation,
    role changes and password changes revoke live sessions immediately.
    """
    st = _st()
    user = current_user()
    if user is None:
        return None

    if not verify_session_binding():
        end_session(
            reason="revoked",
            notice=(
                "This link belongs to a different browser session. "
                "Please sign in again."
            ),
        )
        return None

    expiry = evaluate_session_expiry(
        authenticated_at=user.authenticated_at,
        last_activity_at=st.session_state.get("auth_last_activity"),
    )
    if expiry.expired:
        end_session(reason="timeout", notice=expiry.message)
        return None

    refreshed = get_auth_provider().revalidate(user)
    if refreshed is None:
        end_session(
            reason="revoked",
            notice="Your session is no longer valid. Please sign in again.",
        )
        return None

    st.session_state.auth_user = refreshed
    now = datetime.now(timezone.utc)
    st.session_state.auth_last_activity = now.isoformat()
    _throttled_activity_write(refreshed.user_id, now)
    return refreshed


def _throttled_activity_write(user_id: int, now: datetime) -> None:
    """Persist ``last_activity_at`` at most once a minute to limit DB writes."""
    st = _st()
    last_written = st.session_state.get("_auth_activity_written")
    if last_written:
        try:
            previous = datetime.fromisoformat(str(last_written))
            if (now - previous).total_seconds() < 60:
                return
        except (TypeError, ValueError):
            pass
    st.session_state._auth_activity_written = now.isoformat()
    touch_last_activity(user_id, when=now.isoformat())


# --- session-only OpenRouter key ------------------------------------------


def set_openrouter_key(key: str, verification: dict[str, Any] | None = None) -> None:
    """Hold a verified key for this browser session only. Never persisted."""
    st = _st()
    st.session_state.openrouter_api_key = str(key or "")
    st.session_state.openrouter_key_status = {
        "verified": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "masked": mask_secret(key),
        **(verification or {}),
    }


def get_openrouter_key() -> str:
    st = _st()
    return str(st.session_state.get("openrouter_api_key") or "")


def has_verified_openrouter_key() -> bool:
    status = _st().session_state.get("openrouter_key_status") or {}
    return bool(get_openrouter_key()) and bool(status.get("verified"))


def get_openrouter_key_status() -> dict[str, Any]:
    """Sanitized verification summary — contains no key material."""
    status = _st().session_state.get("openrouter_key_status") or {}
    return {k: v for k, v in dict(status).items() if k != "api_key"}


def remove_openrouter_key() -> None:
    clear_byok_state()


def has_accepted_cost_notice() -> bool:
    return bool(_st().session_state.get("openrouter_cost_ack"))


def accept_cost_notice() -> None:
    _st().session_state.openrouter_cost_ack = datetime.now(timezone.utc).isoformat()


def session_timeout_summary() -> str:
    idle = getattr(config, "SESSION_IDLE_TIMEOUT_MINUTES", 30)
    absolute = getattr(config, "SESSION_ABSOLUTE_TIMEOUT_HOURS", 12)
    return f"Signed out after {idle} minutes idle or {absolute} hours total."
