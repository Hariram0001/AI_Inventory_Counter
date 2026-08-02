"""Administrator-managed OpenRouter API key for the whole deployment.

Only administrators may set or remove the key. Regular users never see it —
they can only run OpenRouter models when an administrator has configured a
key and enabled the model in Model Access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from database import _connect, initialize_database, utc_now_iso
from security import mask_secret, redact_secrets

SECRET_NAME = "openrouter_api_key"


@dataclass(frozen=True)
class DeploymentKeyStatus:
    """Public status only — never includes the plaintext key."""

    configured: bool
    verified: bool
    masked: str = ""
    label: str = ""
    credit_limit: float | None = None
    usage: float | None = None
    is_free_tier: bool | None = None
    updated_at: str = ""
    updated_by: str = ""

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "verified": self.verified,
            "masked": self.masked,
            "label": self.label,
            "credit_limit": self.credit_limit,
            "usage": self.usage,
            "is_free_tier": self.is_free_tier,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


def has_verified_deployment_key(db_path: str | None = None) -> bool:
    return bool(get_deployment_key(db_path))


def get_deployment_key(db_path: str | None = None) -> str:
    """Return the plaintext key for inference only. Empty when unset."""
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT secret_value FROM deployment_secrets WHERE name = ?",
            (SECRET_NAME,),
        ).fetchone()
    if row is None:
        return ""
    return str(row["secret_value"] or "").strip()


def get_deployment_key_status(db_path: str | None = None) -> DeploymentKeyStatus:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT secret_value, metadata_json, updated_at, updated_by "
            "FROM deployment_secrets WHERE name = ?",
            (SECRET_NAME,),
        ).fetchone()
    if row is None:
        return DeploymentKeyStatus(configured=False, verified=False)

    meta: dict[str, Any] = {}
    try:
        meta = json.loads(row["metadata_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        meta = {}
    meta = redact_secrets(meta) if isinstance(meta, dict) else {}
    key = str(row["secret_value"] or "").strip()
    return DeploymentKeyStatus(
        configured=bool(key),
        verified=bool(key) and bool(meta.get("verified", True)),
        masked=str(meta.get("masked") or mask_secret(key)),
        label=str(meta.get("label") or ""),
        credit_limit=meta.get("credit_limit"),
        usage=meta.get("usage"),
        is_free_tier=meta.get("is_free_tier"),
        updated_at=str(row["updated_at"] or ""),
        updated_by=str(row["updated_by"] or ""),
    )


def save_deployment_key(
    api_key: str,
    *,
    verification: dict[str, Any] | None = None,
    updated_by: str = "",
    db_path: str | None = None,
) -> DeploymentKeyStatus:
    """Persist a verified OpenRouter key for the whole deployment."""
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("OpenRouter API key must not be blank.")

    public = dict(verification or {})
    public.pop("key", None)
    public.pop("api_key", None)
    public["verified"] = True
    public.setdefault("masked", mask_secret(key))
    public = redact_secrets(public)

    initialize_database(db_path)
    now = utc_now_iso()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO deployment_secrets
                (name, secret_value, metadata_json, updated_at, updated_by)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                secret_value = excluded.secret_value,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at,
                updated_by = excluded.updated_by
            """,
            (SECRET_NAME, key, json.dumps(public), now, str(updated_by or "")),
        )
    return get_deployment_key_status(db_path)


def clear_deployment_key(
    *, updated_by: str = "", db_path: str | None = None
) -> None:
    initialize_database(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM deployment_secrets WHERE name = ?", (SECRET_NAME,)
        )


__all__ = [
    "DeploymentKeyStatus",
    "SECRET_NAME",
    "clear_deployment_key",
    "get_deployment_key",
    "get_deployment_key_status",
    "has_verified_deployment_key",
    "save_deployment_key",
]
