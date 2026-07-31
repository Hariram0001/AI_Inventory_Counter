"""User, audit, model-policy and usage persistence.

All password material is hashed before it reaches this layer's storage calls,
and every audit detail payload is passed through :func:`security.redact_secrets`
so secrets can never be written to the audit table.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from database import DatabaseError, _connect, initialize_database, utc_now_iso
from security import (
    hash_password,
    needs_rehash,
    normalize_email,
    normalize_username,
    redact_secrets,
    verify_password,
)

ROLE_ADMIN = "admin"
ROLE_USER = "user"
ROLES = (ROLE_ADMIN, ROLE_USER)

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


class UserStoreError(Exception):
    """Raised for user-management failures that are safe to show to operators."""


@dataclass(frozen=True)
class UserRecord:
    id: int
    username: str
    email: str = ""
    display_name: str = ""
    role: str = ROLE_USER
    is_active: bool = True
    force_password_change: bool = False
    failed_login_count: int = 0
    locked_until: str | None = None
    last_login_at: str | None = None
    last_activity_at: str | None = None
    password_changed_at: str | None = None
    session_version: int = 1
    auth_provider: str = "local"
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    @property
    def label(self) -> str:
        return self.display_name or self.username

    def is_locked(self, *, now: datetime | None = None) -> bool:
        return lock_remaining_seconds(self.locked_until, now=now) > 0

    def to_public_dict(self) -> dict[str, Any]:
        """Serializable view with no password material."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role,
            "is_active": self.is_active,
            "force_password_change": self.force_password_change,
            "failed_login_count": self.failed_login_count,
            "locked_until": self.locked_until,
            "last_login_at": self.last_login_at,
            "last_activity_at": self.last_activity_at,
            "session_version": self.session_version,
            "auth_provider": self.auth_provider,
            "created_at": self.created_at,
        }


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def lock_remaining_seconds(locked_until: str | None, *, now: datetime | None = None) -> int:
    """Seconds remaining on a lockout; 0 when unlocked or unparseable."""
    deadline = _parse_iso(locked_until)
    if deadline is None:
        return 0
    reference = now or datetime.now(timezone.utc)
    remaining = (deadline - reference).total_seconds()
    return int(remaining) if remaining > 0 else 0


def _row_to_user(row: sqlite3.Row | dict[str, Any] | None) -> UserRecord | None:
    if row is None:
        return None
    data = dict(row)
    return UserRecord(
        id=int(data["id"]),
        username=str(data.get("username") or ""),
        email=str(data.get("email") or ""),
        display_name=str(data.get("display_name") or ""),
        role=str(data.get("role") or ROLE_USER),
        is_active=bool(data.get("is_active", 1)),
        force_password_change=bool(data.get("force_password_change", 0)),
        failed_login_count=int(data.get("failed_login_count") or 0),
        locked_until=data.get("locked_until"),
        last_login_at=data.get("last_login_at"),
        last_activity_at=data.get("last_activity_at"),
        password_changed_at=data.get("password_changed_at"),
        session_version=int(data.get("session_version") or 1),
        auth_provider=str(data.get("auth_provider") or "local"),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
        created_by=str(data.get("created_by") or ""),
    )


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def record_audit_event(
    event_type: str,
    *,
    actor_user_id: int | None = None,
    actor_username: str | None = None,
    target_type: str | None = None,
    target_id: str | int | None = None,
    outcome: str = "success",
    detail: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> int:
    """Append a redacted audit entry. Never raises into the caller's flow."""
    payload: str | None = None
    if detail:
        try:
            payload = json.dumps(redact_secrets(detail), default=str)[:4000]
        except Exception:  # noqa: BLE001
            payload = None
    try:
        initialize_database(db_path)
        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO audit_events
                    (created_at, actor_user_id, actor_username, event_type,
                     target_type, target_id, outcome, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    actor_user_id,
                    actor_username,
                    str(event_type),
                    target_type,
                    None if target_id is None else str(target_id),
                    str(outcome or "success"),
                    payload,
                ),
            )
            return int(cur.lastrowid)
    except Exception:  # noqa: BLE001 — auditing must never break the request
        return 0


def get_audit_events(
    *,
    limit: int = 200,
    event_type: str | None = None,
    actor_username: str | None = None,
    outcome: str | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    initialize_database(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if event_type:
        clauses.append("event_type = ?")
        params.append(str(event_type))
    if actor_username:
        clauses.append("actor_username = ?")
        params.append(normalize_username(actor_username))
    if outcome:
        clauses.append("outcome = ?")
        params.append(str(outcome))
    sql = "SELECT * FROM audit_events"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with _connect(db_path) as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def list_audit_event_types(db_path: str | None = None) -> list[str]:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT event_type FROM audit_events ORDER BY event_type"
        ).fetchall()
    return [str(row[0]) for row in rows]


# ---------------------------------------------------------------------------
# User lookups
# ---------------------------------------------------------------------------


def get_user_by_username(username: str, db_path: str | None = None) -> UserRecord | None:
    normalized = normalize_username(username)
    if not normalized:
        return None
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (normalized,)
        ).fetchone()
    return _row_to_user(row)


def get_user_by_id(user_id: int, db_path: str | None = None) -> UserRecord | None:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
    return _row_to_user(row)


def list_users(
    db_path: str | None = None, *, include_inactive: bool = True
) -> list[UserRecord]:
    initialize_database(db_path)
    sql = "SELECT * FROM users"
    if not include_inactive:
        sql += " WHERE is_active = 1"
    sql += " ORDER BY role = 'admin' DESC, username ASC"
    with _connect(db_path) as conn:
        rows = conn.execute(sql).fetchall()
    return [user for user in (_row_to_user(row) for row in rows) if user is not None]


def count_users(db_path: str | None = None) -> int:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    return int(row[0] if row else 0)


def count_active_admins(db_path: str | None = None, *, exclude_user_id: int | None = None) -> int:
    initialize_database(db_path)
    sql = "SELECT COUNT(*) FROM users WHERE role = ? AND is_active = 1"
    params: list[Any] = [ROLE_ADMIN]
    if exclude_user_id is not None:
        sql += " AND id != ?"
        params.append(int(exclude_user_id))
    with _connect(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return int(row[0] if row else 0)


# ---------------------------------------------------------------------------
# User mutations
# ---------------------------------------------------------------------------


def create_user(
    *,
    username: str,
    password: str,
    role: str = ROLE_USER,
    email: str = "",
    display_name: str = "",
    force_password_change: bool = True,
    is_active: bool = True,
    created_by: str = "",
    db_path: str | None = None,
) -> UserRecord:
    """Create a user from an already policy-validated password."""
    normalized = normalize_username(username)
    if not normalized:
        raise UserStoreError("Username is required.")
    if role not in ROLES:
        raise UserStoreError("Unknown role.")

    now = utc_now_iso()
    password_hash = hash_password(password)
    initialize_database(db_path)
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users
                    (username, email, display_name, password_hash, role, is_active,
                     force_password_change, failed_login_count, session_version,
                     auth_provider, password_changed_at, created_at, updated_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, 'local', ?, ?, ?, ?)
                """,
                (
                    normalized,
                    normalize_email(email),
                    str(display_name or "").strip(),
                    password_hash,
                    role,
                    1 if is_active else 0,
                    1 if force_password_change else 0,
                    now,
                    now,
                    now,
                    str(created_by or ""),
                ),
            )
    except DatabaseError as exc:
        if "UNIQUE" in str(exc).upper():
            raise UserStoreError("That username is already taken.") from exc
        raise UserStoreError("Could not create the user.") from exc

    created = get_user_by_username(normalized, db_path)
    if created is None:
        raise UserStoreError("Could not create the user.")
    return created


def _update_user(user_id: int, fields: dict[str, Any], db_path: str | None = None) -> None:
    if not fields:
        return
    payload = dict(fields)
    payload["updated_at"] = utc_now_iso()
    assignments = ", ".join(f"{key} = ?" for key in payload)
    values = list(payload.values()) + [int(user_id)]
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", values)


def update_user_profile(
    user_id: int,
    *,
    email: str | None = None,
    display_name: str | None = None,
    db_path: str | None = None,
) -> UserRecord | None:
    fields: dict[str, Any] = {}
    if email is not None:
        fields["email"] = normalize_email(email)
    if display_name is not None:
        fields["display_name"] = str(display_name).strip()
    _update_user(user_id, fields, db_path)
    return get_user_by_id(user_id, db_path)


def set_user_role(
    user_id: int, role: str, *, db_path: str | None = None
) -> UserRecord | None:
    if role not in ROLES:
        raise UserStoreError("Unknown role.")
    user = get_user_by_id(user_id, db_path)
    if user is None:
        raise UserStoreError("User not found.")
    if (
        user.role == ROLE_ADMIN
        and role != ROLE_ADMIN
        and count_active_admins(db_path, exclude_user_id=user_id) == 0
    ):
        raise UserStoreError("The last active administrator cannot be demoted.")
    # Role changes invalidate existing sessions so the new role takes effect now.
    _update_user(
        user_id,
        {"role": role, "session_version": int(user.session_version) + 1},
        db_path,
    )
    return get_user_by_id(user_id, db_path)


def set_user_active(
    user_id: int, is_active: bool, *, db_path: str | None = None
) -> UserRecord | None:
    user = get_user_by_id(user_id, db_path)
    if user is None:
        raise UserStoreError("User not found.")
    if (
        not is_active
        and user.role == ROLE_ADMIN
        and count_active_admins(db_path, exclude_user_id=user_id) == 0
    ):
        raise UserStoreError("The last active administrator cannot be deactivated.")
    fields: dict[str, Any] = {"is_active": 1 if is_active else 0}
    if not is_active:
        fields["session_version"] = int(user.session_version) + 1
    _update_user(user_id, fields, db_path)
    return get_user_by_id(user_id, db_path)


def unlock_user(user_id: int, *, db_path: str | None = None) -> UserRecord | None:
    _update_user(
        user_id, {"failed_login_count": 0, "locked_until": None}, db_path
    )
    return get_user_by_id(user_id, db_path)


def set_password(
    user_id: int,
    new_password: str,
    *,
    force_password_change: bool = False,
    invalidate_sessions: bool = True,
    db_path: str | None = None,
) -> UserRecord | None:
    """Rehash and store a new password; bumps session_version by default."""
    user = get_user_by_id(user_id, db_path)
    if user is None:
        raise UserStoreError("User not found.")
    now = utc_now_iso()
    fields: dict[str, Any] = {
        "password_hash": hash_password(new_password),
        "password_changed_at": now,
        "force_password_change": 1 if force_password_change else 0,
        "failed_login_count": 0,
        "locked_until": None,
    }
    if invalidate_sessions:
        fields["session_version"] = int(user.session_version) + 1
    _update_user(user_id, fields, db_path)
    return get_user_by_id(user_id, db_path)


def touch_last_activity(
    user_id: int, *, db_path: str | None = None, when: str | None = None
) -> None:
    try:
        _update_user(user_id, {"last_activity_at": when or utc_now_iso()}, db_path)
    except Exception:  # noqa: BLE001 — activity tracking must not break a request
        pass


def delete_user(user_id: int, *, db_path: str | None = None) -> None:
    user = get_user_by_id(user_id, db_path)
    if user is None:
        raise UserStoreError("User not found.")
    if user.role == ROLE_ADMIN and count_active_admins(db_path, exclude_user_id=user_id) == 0:
        raise UserStoreError("The last active administrator cannot be deleted.")
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))


# ---------------------------------------------------------------------------
# Credential verification with lockout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CredentialResult:
    status: str  # authenticated | invalid | locked | disabled
    user: UserRecord | None = None
    lock_seconds: int = 0
    message: str = ""


def verify_credentials(
    username: str,
    password: str,
    *,
    db_path: str | None = None,
) -> CredentialResult:
    """Check a password, applying and updating the lockout policy."""
    user = get_user_by_username(username, db_path)
    if user is None:
        # Spend comparable time so absent users are not distinguishable.
        verify_password(
            "$argon2id$v=19$m=65536,t=3,p=2$AAAAAAAAAAAAAAAAAAAAAA$"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            password or "x",
        )
        return CredentialResult(status="invalid", message="Invalid username or password.")

    remaining = lock_remaining_seconds(user.locked_until)
    if remaining > 0:
        return CredentialResult(
            status="locked",
            user=user,
            lock_seconds=remaining,
            message="This account is temporarily locked.",
        )

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user.id,)
        ).fetchone()
    stored_hash = str(row["password_hash"]) if row else ""

    if not verify_password(stored_hash, password):
        failures = int(user.failed_login_count) + 1
        fields: dict[str, Any] = {"failed_login_count": failures}
        lock_seconds = 0
        if failures >= MAX_FAILED_ATTEMPTS:
            until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
            fields["locked_until"] = until.isoformat()
            fields["failed_login_count"] = 0
            lock_seconds = LOCKOUT_MINUTES * 60
        _update_user(user.id, fields, db_path)
        if lock_seconds:
            return CredentialResult(
                status="locked",
                user=user,
                lock_seconds=lock_seconds,
                message=(
                    f"Too many failed attempts. Try again in {LOCKOUT_MINUTES} minutes."
                ),
            )
        return CredentialResult(
            status="invalid", user=user, message="Invalid username or password."
        )

    if not user.is_active:
        return CredentialResult(
            status="disabled",
            user=user,
            message="This account is deactivated. Contact an administrator.",
        )

    now = utc_now_iso()
    fields = {
        "failed_login_count": 0,
        "locked_until": None,
        "last_login_at": now,
        "last_activity_at": now,
    }
    if needs_rehash(stored_hash):
        fields["password_hash"] = hash_password(password)
    _update_user(user.id, fields, db_path)

    return CredentialResult(status="authenticated", user=get_user_by_id(user.id, db_path))


# ---------------------------------------------------------------------------
# Model access policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelAccessPolicy:
    model_key: str
    display_name: str = ""
    is_enabled: bool = True
    allowed_roles: tuple[str, ...] = field(default=(ROLE_ADMIN, ROLE_USER))
    requires_user_api_key: bool = False
    requires_cost_confirmation: bool = False
    maximum_runs_per_user_per_day: int | None = None
    notes: str = ""
    updated_at: str = ""
    updated_by: str = ""

    def allows_role(self, role: str) -> bool:
        return str(role) in self.allowed_roles


def _row_to_policy(row: sqlite3.Row | dict[str, Any]) -> ModelAccessPolicy:
    data = dict(row)
    roles_raw = str(data.get("allowed_roles") or "admin,user")
    roles = tuple(r.strip() for r in roles_raw.split(",") if r.strip())
    limit = data.get("maximum_runs_per_user_per_day")
    return ModelAccessPolicy(
        model_key=str(data.get("model_key") or ""),
        display_name=str(data.get("display_name") or ""),
        is_enabled=bool(data.get("is_enabled", 1)),
        allowed_roles=roles or (ROLE_ADMIN, ROLE_USER),
        requires_user_api_key=bool(data.get("requires_user_api_key", 0)),
        requires_cost_confirmation=bool(data.get("requires_cost_confirmation", 0)),
        maximum_runs_per_user_per_day=None if limit is None else int(limit),
        notes=str(data.get("notes") or ""),
        updated_at=str(data.get("updated_at") or ""),
        updated_by=str(data.get("updated_by") or ""),
    )


def upsert_model_policy(
    model_key: str,
    *,
    display_name: str | None = None,
    is_enabled: bool | None = None,
    allowed_roles: tuple[str, ...] | list[str] | None = None,
    requires_user_api_key: bool | None = None,
    requires_cost_confirmation: bool | None = None,
    maximum_runs_per_user_per_day: int | None = None,
    notes: str | None = None,
    updated_by: str = "",
    db_path: str | None = None,
) -> ModelAccessPolicy:
    key = str(model_key or "").strip()
    if not key:
        raise UserStoreError("Model key is required.")
    existing = get_model_policy(key, db_path=db_path, create_default=False)
    base = existing or ModelAccessPolicy(model_key=key)

    roles = tuple(allowed_roles) if allowed_roles is not None else base.allowed_roles
    merged = ModelAccessPolicy(
        model_key=key,
        display_name=base.display_name if display_name is None else display_name,
        is_enabled=base.is_enabled if is_enabled is None else bool(is_enabled),
        allowed_roles=roles,
        requires_user_api_key=(
            base.requires_user_api_key
            if requires_user_api_key is None
            else bool(requires_user_api_key)
        ),
        requires_cost_confirmation=(
            base.requires_cost_confirmation
            if requires_cost_confirmation is None
            else bool(requires_cost_confirmation)
        ),
        maximum_runs_per_user_per_day=(
            base.maximum_runs_per_user_per_day
            if maximum_runs_per_user_per_day is None
            else (None if int(maximum_runs_per_user_per_day) <= 0 else int(maximum_runs_per_user_per_day))
        ),
        notes=base.notes if notes is None else str(notes),
        updated_at=utc_now_iso(),
        updated_by=str(updated_by or ""),
    )

    initialize_database(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_access_policies
                (model_key, display_name, is_enabled, allowed_roles,
                 requires_user_api_key, requires_cost_confirmation,
                 maximum_runs_per_user_per_day, notes, updated_at, updated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_key) DO UPDATE SET
                display_name = excluded.display_name,
                is_enabled = excluded.is_enabled,
                allowed_roles = excluded.allowed_roles,
                requires_user_api_key = excluded.requires_user_api_key,
                requires_cost_confirmation = excluded.requires_cost_confirmation,
                maximum_runs_per_user_per_day = excluded.maximum_runs_per_user_per_day,
                notes = excluded.notes,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (
                merged.model_key,
                merged.display_name,
                1 if merged.is_enabled else 0,
                ",".join(merged.allowed_roles),
                1 if merged.requires_user_api_key else 0,
                1 if merged.requires_cost_confirmation else 0,
                merged.maximum_runs_per_user_per_day,
                merged.notes,
                merged.updated_at,
                merged.updated_by,
            ),
        )
    return merged


def get_model_policy(
    model_key: str, *, db_path: str | None = None, create_default: bool = True
) -> ModelAccessPolicy | None:
    key = str(model_key or "").strip()
    if not key:
        return None
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM model_access_policies WHERE model_key = ?", (key,)
        ).fetchone()
    if row is not None:
        return _row_to_policy(row)
    if create_default:
        return ModelAccessPolicy(model_key=key)
    return None


def list_model_policies(db_path: str | None = None) -> list[ModelAccessPolicy]:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM model_access_policies ORDER BY model_key"
        ).fetchall()
    return [_row_to_policy(row) for row in rows]


# ---------------------------------------------------------------------------
# Per-user daily usage
# ---------------------------------------------------------------------------


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_usage_count(
    user_id: int, model_key: str, *, usage_date: str | None = None, db_path: str | None = None
) -> int:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT run_count FROM user_model_usage "
            "WHERE user_id = ? AND model_key = ? AND usage_date = ?",
            (int(user_id), str(model_key), usage_date or _today_utc()),
        ).fetchone()
    return int(row[0]) if row else 0


def increment_usage(
    user_id: int,
    model_key: str,
    *,
    amount: int = 1,
    usage_date: str | None = None,
    db_path: str | None = None,
) -> int:
    date_key = usage_date or _today_utc()
    now = utc_now_iso()
    initialize_database(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_model_usage (user_id, model_key, usage_date, run_count, last_run_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, model_key, usage_date) DO UPDATE SET
                run_count = run_count + excluded.run_count,
                last_run_at = excluded.last_run_at
            """,
            (int(user_id), str(model_key), date_key, int(amount), now),
        )
    return get_usage_count(user_id, model_key, usage_date=date_key, db_path=db_path)


def get_usage_summary(
    *, days: int = 7, db_path: str | None = None
) -> list[dict[str, Any]]:
    initialize_database(db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).strftime(
        "%Y-%m-%d"
    )
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT u.username AS username, m.model_key AS model_key,
                   SUM(m.run_count) AS runs, MAX(m.last_run_at) AS last_run_at
            FROM user_model_usage m
            LEFT JOIN users u ON u.id = m.user_id
            WHERE m.usage_date >= ?
            GROUP BY u.username, m.model_key
            ORDER BY runs DESC
            """,
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]
