"""Authentication abstraction, first-admin bootstrap and session policy.

The rest of the application depends only on :class:`AuthenticatedUser` and
:class:`AuthenticationProvider`. Swapping the local password provider for an
OIDC provider later means implementing the same protocol and changing
:func:`get_auth_provider`, with no changes to callers.

This module deliberately does not import Streamlit so it stays testable
offline; Streamlit session wiring lives in ``auth_session.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

import config
from security import validate_password_policy, validate_username
from user_store import (
    ROLE_ADMIN,
    ROLE_USER,
    CredentialResult,
    UserRecord,
    UserStoreError,
    count_users,
    create_user,
    get_user_by_id,
    get_user_by_username,
    record_audit_event,
    verify_credentials,
)

AUTH_PROVIDER_LOCAL = "local"

# Audit event names used across the app.
EVENT_LOGIN_SUCCESS = "auth.login.success"
EVENT_LOGIN_FAILURE = "auth.login.failure"
EVENT_LOGIN_LOCKED = "auth.login.locked"
EVENT_LOGOUT = "auth.logout"
EVENT_SESSION_TIMEOUT = "auth.session.timeout"
EVENT_SESSION_INVALIDATED = "auth.session.invalidated"
EVENT_PASSWORD_CHANGED = "auth.password.changed"
EVENT_PASSWORD_RESET = "admin.password.reset"
EVENT_PASSWORD_RESET_REQUESTED = "auth.password.reset_requested"
EVENT_SIGNUP_REQUESTED = "auth.signup.requested"
EVENT_SIGNUP_APPROVED = "admin.signup.approved"
EVENT_SIGNUP_REJECTED = "admin.signup.rejected"
EVENT_BOOTSTRAP_ADMIN = "admin.bootstrap"
EVENT_USER_CREATED = "admin.user.created"
EVENT_USER_UPDATED = "admin.user.updated"
EVENT_USER_DEACTIVATED = "admin.user.deactivated"
EVENT_USER_ACTIVATED = "admin.user.activated"
EVENT_USER_UNLOCKED = "admin.user.unlocked"
EVENT_USER_DELETED = "admin.user.deleted"
EVENT_POLICY_UPDATED = "admin.policy.updated"
EVENT_SAMPLE_UPLOADED = "admin.sample.uploaded"
EVENT_SAMPLE_UPDATED = "admin.sample.updated"
EVENT_SAMPLE_DELETED = "admin.sample.deleted"
EVENT_KEY_VERIFIED = "byok.key.verified"
EVENT_KEY_VERIFY_FAILED = "byok.key.verify_failed"
EVENT_KEY_REMOVED = "byok.key.removed"
EVENT_COST_ACKNOWLEDGED = "byok.cost.acknowledged"
EVENT_QUOTA_BLOCKED = "policy.quota.blocked"
EVENT_INFERENCE_RUN = "inference.run"
EVENT_ACCESS_DENIED = "authz.denied"


@dataclass(frozen=True)
class AuthenticatedUser:
    """Canonical identity handed to every protected surface."""

    user_id: int
    username: str
    display_name: str
    email: str
    role: str
    is_active: bool
    force_password_change: bool
    session_version: int
    auth_provider: str = AUTH_PROVIDER_LOCAL
    authenticated_at: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def label(self) -> str:
        return self.display_name or self.username

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "force_password_change": self.force_password_change,
            "session_version": self.session_version,
            "auth_provider": self.auth_provider,
            "authenticated_at": self.authenticated_at,
        }


def to_authenticated_user(record: UserRecord) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=record.id,
        username=record.username,
        display_name=record.display_name or record.username,
        email=record.email,
        role=record.role,
        is_active=record.is_active,
        force_password_change=record.force_password_change,
        session_version=record.session_version,
        auth_provider=record.auth_provider or AUTH_PROVIDER_LOCAL,
        authenticated_at=datetime.now(timezone.utc).isoformat(),
    )


@dataclass(frozen=True)
class AuthOutcome:
    """Result of an authentication attempt, safe to show to the end user."""

    status: str  # authenticated | invalid | locked | disabled | error
    user: AuthenticatedUser | None = None
    message: str = ""
    lock_seconds: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "authenticated" and self.user is not None


@runtime_checkable
class AuthenticationProvider(Protocol):
    """Replaceable authentication backend."""

    name: str

    def authenticate(self, username: str, password: str) -> AuthOutcome: ...

    def get_user(self, user_id: int) -> AuthenticatedUser | None: ...

    def revalidate(self, user: AuthenticatedUser) -> AuthenticatedUser | None: ...


class LocalPasswordProvider:
    """Username + Argon2id password verification against the local users table."""

    name = AUTH_PROVIDER_LOCAL

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path

    def authenticate(self, username: str, password: str) -> AuthOutcome:
        if not str(username or "").strip() or not password:
            return AuthOutcome(
                status="invalid", message="Enter both a username and a password."
            )
        try:
            result: CredentialResult = verify_credentials(
                username, password, db_path=self.db_path
            )
        except Exception:  # noqa: BLE001 — never surface storage internals at login
            return AuthOutcome(
                status="error",
                message="Sign-in is temporarily unavailable. Try again shortly.",
            )

        if result.status == "authenticated" and result.user is not None:
            return AuthOutcome(status="authenticated", user=to_authenticated_user(result.user))
        return AuthOutcome(
            status=result.status,
            message=result.message or "Invalid username or password.",
            lock_seconds=result.lock_seconds,
        )

    def get_user(self, user_id: int) -> AuthenticatedUser | None:
        record = get_user_by_id(user_id, self.db_path)
        return to_authenticated_user(record) if record else None

    def revalidate(self, user: AuthenticatedUser) -> AuthenticatedUser | None:
        """Re-read the stored user and reject stale or revoked sessions."""
        record = get_user_by_id(user.user_id, self.db_path)
        if record is None or not record.is_active:
            return None
        if int(record.session_version) != int(user.session_version):
            return None
        # Preserve the original sign-in time so absolute timeouts stay honest.
        return replace(
            to_authenticated_user(record),
            authenticated_at=user.authenticated_at,
        )


_PROVIDER: AuthenticationProvider | None = None


def get_auth_provider(db_path: str | None = None) -> AuthenticationProvider:
    """Return the active provider. Replace this to adopt OIDC later."""
    global _PROVIDER
    if _PROVIDER is None or db_path is not None:
        _PROVIDER = LocalPasswordProvider(db_path)
    return _PROVIDER


def reset_auth_provider() -> None:
    """Test hook — drop the cached provider."""
    global _PROVIDER
    _PROVIDER = None


# ---------------------------------------------------------------------------
# Session policy (pure functions so they can be tested offline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionExpiry:
    expired: bool
    reason: str = ""

    @property
    def message(self) -> str:
        if self.reason == "idle":
            return "You were signed out after a period of inactivity."
        if self.reason == "absolute":
            return "Your session reached its maximum length. Please sign in again."
        if self.reason == "revoked":
            return "Your session is no longer valid. Please sign in again."
        return ""


def evaluate_session_expiry(
    *,
    authenticated_at: str | datetime | None,
    last_activity_at: str | datetime | None,
    now: datetime | None = None,
    idle_minutes: int | None = None,
    absolute_hours: int | None = None,
) -> SessionExpiry:
    """Decide whether a session exceeded the idle or absolute lifetime."""
    reference = now or datetime.now(timezone.utc)
    idle_limit = timedelta(
        minutes=int(
            idle_minutes
            if idle_minutes is not None
            else getattr(config, "SESSION_IDLE_TIMEOUT_MINUTES", 30)
        )
    )
    absolute_limit = timedelta(
        hours=int(
            absolute_hours
            if absolute_hours is not None
            else getattr(config, "SESSION_ABSOLUTE_TIMEOUT_HOURS", 12)
        )
    )

    started = _coerce_datetime(authenticated_at)
    seen = _coerce_datetime(last_activity_at) or started
    if started is None:
        return SessionExpiry(expired=True, reason="revoked")
    if reference - started >= absolute_limit:
        return SessionExpiry(expired=True, reason="absolute")
    if seen is not None and reference - seen >= idle_limit:
        return SessionExpiry(expired=True, reason="idle")
    return SessionExpiry(expired=False)


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# First administrator bootstrap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    status: str  # created | skipped | misconfigured | error
    username: str = ""
    message: str = ""

    @property
    def created(self) -> bool:
        return self.status == "created"


def bootstrap_admin_if_needed(db_path: str | None = None) -> BootstrapResult:
    """Create the first administrator from environment / Streamlit secrets.

    Runs only when the users table is empty. The bootstrap password is never
    stored, echoed or logged. The account is usable immediately: this POC does
    not force a password change, since passwords are unconstrained anyway.
    """
    try:
        if count_users(db_path) > 0:
            return BootstrapResult(status="skipped", message="Users already exist.")
    except Exception:  # noqa: BLE001
        return BootstrapResult(
            status="error", message="Could not read the user database."
        )

    username_raw = getattr(config, "BOOTSTRAP_ADMIN_USERNAME", "") or os.getenv(
        "BOOTSTRAP_ADMIN_USERNAME", ""
    )
    password = getattr(config, "BOOTSTRAP_ADMIN_PASSWORD", "") or os.getenv(
        "BOOTSTRAP_ADMIN_PASSWORD", ""
    )
    email = getattr(config, "BOOTSTRAP_ADMIN_EMAIL", "") or os.getenv(
        "BOOTSTRAP_ADMIN_EMAIL", ""
    )

    if not username_raw or not password:
        return BootstrapResult(
            status="misconfigured",
            message=(
                "No administrator exists yet. Set BOOTSTRAP_ADMIN_USERNAME and "
                "BOOTSTRAP_ADMIN_PASSWORD in your environment or Streamlit "
                "secrets, then reload this page."
            ),
        )

    try:
        username = validate_username(username_raw)
    except ValueError as exc:
        return BootstrapResult(status="misconfigured", message=str(exc))

    problems = validate_password_policy(password, username=username, email=email)
    if problems:
        return BootstrapResult(
            status="misconfigured",
            message=(
                "The configured bootstrap password does not meet the password "
                "policy: " + " ".join(problems)
            ),
        )

    if get_user_by_username(username, db_path) is not None:
        return BootstrapResult(status="skipped", username=username)

    try:
        create_user(
            username=username,
            password=password,
            role=ROLE_ADMIN,
            email=email,
            display_name="Administrator",
            force_password_change=False,
            created_by="bootstrap",
            db_path=db_path,
        )
    except (UserStoreError, ValueError) as exc:
        return BootstrapResult(status="error", message=str(exc))

    record_audit_event(
        EVENT_BOOTSTRAP_ADMIN,
        actor_username="bootstrap",
        target_type="user",
        target_id=username,
        detail={"role": ROLE_ADMIN, "source": "environment"},
        db_path=db_path,
    )
    return BootstrapResult(
        status="created",
        username=username,
        message=(
            f"Administrator '{username}' was created. Sign in with the "
            "configured bootstrap password."
        ),
    )


__all__ = [
    "AUTH_PROVIDER_LOCAL",
    "AuthOutcome",
    "AuthenticatedUser",
    "AuthenticationProvider",
    "BootstrapResult",
    "LocalPasswordProvider",
    "ROLE_ADMIN",
    "ROLE_USER",
    "SessionExpiry",
    "bootstrap_admin_if_needed",
    "evaluate_session_expiry",
    "get_auth_provider",
    "reset_auth_provider",
    "to_authenticated_user",
]
