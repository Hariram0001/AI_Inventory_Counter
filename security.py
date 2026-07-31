"""Secret redaction and password hashing primitives.

This module is deliberately free of Streamlit and database imports so it can be
used from any layer (UI, persistence, adapters, audit logging).
"""

from __future__ import annotations

import re
import secrets
import string
import unicodedata
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

REDACTED = "***REDACTED***"

# Key names whose *values* must never be surfaced, regardless of nesting depth.
SENSITIVE_KEY_PARTS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "api-key",
    "model_api_key",
    "roboflow_api_key",
    "openrouter_api_key",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
    "id_token",
    "bearer",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
    "password_hash",
    "temporary_password",
    "new_password",
    "current_password",
    "session_token",
    "private_key",
    "credential",
)

_QUERY_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password)=([^&\s\"']+)")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}")
_HEADER_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|token|secret|password)\s*[:=]\s*[^\s,;\"']+"
)
# Long opaque tokens (OpenRouter keys look like sk-or-v1-<hex>, Roboflow keys are ~20+ chars).
_OPENROUTER_KEY_RE = re.compile(r"(?i)\bsk-or-v1-[A-Za-z0-9\-_]{8,}")
_SK_KEY_RE = re.compile(r"(?i)\bsk-[A-Za-z0-9\-_]{16,}")

_MAX_REDACTION_DEPTH = 12

# Metadata fields that merely *describe* a secret (names, flags, statuses) and
# whose values are safe to keep — e.g. ``api_key_parameter_name`` or
# ``requires_user_api_key``.
_METADATA_KEY_SUFFIXES: tuple[str, ...] = (
    "_parameter_name",
    "_param",
    "_field",
    "_configured",
    "_present",
    "_status",
    "_masked",
    "_label",
    "_required",
)
_METADATA_KEY_PREFIXES: tuple[str, ...] = (
    "requires_",
    "needs_",
    "has_",
    "use_",
    "is_",
    "allow_",
    "masked_",
)


def is_sensitive_key(key: Any) -> bool:
    """True when a mapping key names a secret whose value must be redacted."""
    name = str(key or "").strip().lower().replace("-", "_")
    if not name:
        return False
    if name.startswith(_METADATA_KEY_PREFIXES) or name.endswith(_METADATA_KEY_SUFFIXES):
        return False
    return any(part.replace("-", "_") in name for part in SENSITIVE_KEY_PARTS)


def redact_text(text: Any, *, max_len: int | None = None) -> str:
    """Redact secrets embedded in free-form text (URLs, headers, exceptions)."""
    cleaned = str(text if text is not None else "")
    cleaned = _OPENROUTER_KEY_RE.sub(REDACTED, cleaned)
    cleaned = _SK_KEY_RE.sub(REDACTED, cleaned)
    cleaned = _QUERY_SECRET_RE.sub(rf"\1={REDACTED}", cleaned)
    cleaned = _BEARER_RE.sub(f"Bearer {REDACTED}", cleaned)
    cleaned = _HEADER_SECRET_RE.sub(rf"\1={REDACTED}", cleaned)
    cleaned = cleaned.replace("\x00", "")
    if max_len is not None and len(cleaned) > max_len:
        return cleaned[: max(0, max_len - 3)] + "..."
    return cleaned


def redact_secrets(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redact secrets from nested dicts, lists, tuples and strings.

    Mapping values are redacted by key name; strings are scrubbed by pattern so
    keys leaked inside URLs or exception text are caught even when the
    surrounding key name looks harmless.
    """
    if _depth > _MAX_REDACTION_DEPTH:
        return "<max-depth>"

    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(key):
                out[key] = REDACTED if item is not None else None
            else:
                out[key] = redact_secrets(item, _depth=_depth + 1)
        return out

    if isinstance(value, (list, tuple, set)):
        rendered = [redact_secrets(item, _depth=_depth + 1) for item in value]
        if isinstance(value, tuple):
            return tuple(rendered)
        if isinstance(value, set):
            return rendered
        return rendered

    if isinstance(value, str):
        return redact_text(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"

    return redact_text(repr(value), max_len=500)


def mask_secret(value: str | None, *, keep: int = 4) -> str:
    """Return a non-reversible display form such as ``sk-or…a1b2`` for UI echo."""
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= keep * 2:
        return "*" * len(text)
    return f"{text[:keep]}…{text[-keep:]}"


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

# Argon2id with parameters that stay responsive on Streamlit Community Cloud's
# shared CPU while exceeding OWASP's minimum recommendation.
_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=64 * 1024,
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 256

# Rejected outright regardless of length — these show up in POC deployments.
_BANNED_PASSWORD_SUBSTRINGS = (
    "password",
    "passw0rd",
    "changeme",
    "change_me",
    "letmein",
    "welcome123",
    "qwerty",
    "admin123",
    "123456",
    "iloveyou",
    "inventory123",
)


class PasswordPolicyError(ValueError):
    """Raised when a candidate password violates policy."""


def hash_password(password: str) -> str:
    """Return an Argon2id PHC-format hash. Never store or log the input."""
    if not isinstance(password, str) or not password:
        raise PasswordPolicyError("Password must be a non-empty string.")
    return _HASHER.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Constant-time-ish verification that never raises on bad input."""
    if not password_hash or not isinstance(password, str) or not password:
        return False
    try:
        return bool(_HASHER.verify(password_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:  # noqa: BLE001 — corrupt hash must not surface internals
        return False


def needs_rehash(password_hash: str) -> bool:
    """True when the stored hash uses outdated Argon2 parameters."""
    if not password_hash:
        return False
    try:
        return bool(_HASHER.check_needs_rehash(password_hash))
    except Exception:  # noqa: BLE001
        return False


def validate_password_policy(
    password: str,
    *,
    username: str | None = None,
    email: str | None = None,
) -> list[str]:
    """Return a list of human-readable policy violations (empty when valid)."""
    problems: list[str] = []
    candidate = password if isinstance(password, str) else ""

    if len(candidate) < MIN_PASSWORD_LENGTH:
        problems.append(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )
    if len(candidate) > MAX_PASSWORD_LENGTH:
        problems.append(
            f"Password must be at most {MAX_PASSWORD_LENGTH} characters long."
        )
    if candidate != candidate.strip():
        problems.append("Password must not start or end with whitespace.")
    if not candidate.strip():
        problems.append("Password must not be blank.")

    lowered = candidate.lower()
    if any(banned in lowered for banned in _BANNED_PASSWORD_SUBSTRINGS):
        problems.append("Password contains a commonly used or placeholder phrase.")

    for label, value in (("username", username), ("email", email)):
        text = str(value or "").strip().lower()
        if text and len(text) >= 3 and text in lowered:
            problems.append(f"Password must not contain your {label}.")

    if candidate and len(set(candidate)) < 5:
        problems.append("Password must use at least 5 distinct characters.")

    classes = sum(
        (
            any(c.islower() for c in candidate),
            any(c.isupper() for c in candidate),
            any(c.isdigit() for c in candidate),
            any(not c.isalnum() for c in candidate),
        )
    )
    if classes < 3:
        problems.append(
            "Password must combine at least three of: lowercase, uppercase, "
            "digits, symbols."
        )

    # Deduplicate while preserving order so the UI shows a stable list.
    seen: set[str] = set()
    unique: list[str] = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            unique.append(problem)
    return unique


def generate_temporary_password(length: int = 16) -> str:
    """Generate a policy-compliant temporary password for admin resets."""
    size = max(MIN_PASSWORD_LENGTH, int(length))
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*?-_"
    for _ in range(200):
        candidate = "".join(secrets.choice(alphabet) for _ in range(size))
        if not validate_password_policy(candidate):
            return candidate
    # Deterministic fallback that satisfies every rule above.
    return "Tmp!" + secrets.token_urlsafe(size)[:size] + "9zQ"


# ---------------------------------------------------------------------------
# Identifier normalization
# ---------------------------------------------------------------------------

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")


def normalize_username(raw: str | None) -> str:
    """Casefold and strip a username so lookups are stable and unique."""
    text = unicodedata.normalize("NFKC", str(raw or "")).strip().lower()
    return re.sub(r"\s+", "", text)


def validate_username(raw: str | None) -> str:
    """Return the normalized username or raise with a user-safe message."""
    username = normalize_username(raw)
    if not _USERNAME_RE.match(username):
        raise ValueError(
            "Username must be 3–32 characters using letters, digits, dot, "
            "underscore or hyphen, and start with a letter or digit."
        )
    return username


def normalize_email(raw: str | None) -> str:
    text = unicodedata.normalize("NFKC", str(raw or "")).strip().lower()
    return text


def validate_email(raw: str | None, *, allow_empty: bool = True) -> str:
    email = normalize_email(raw)
    if not email:
        if allow_empty:
            return ""
        raise ValueError("Email address is required.")
    if not _EMAIL_RE.match(email) or len(email) > 254:
        raise ValueError("Enter a valid email address.")
    return email
