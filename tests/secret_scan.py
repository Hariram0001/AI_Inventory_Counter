"""Shared helper for asserting that persisted artifacts hold no secret values."""

from __future__ import annotations

import re
from typing import Any

from security import is_sensitive_key

# Values that look like real credentials rather than metadata.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bsk-or-v1-[A-Za-z0-9\-_]{8,}"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\bapi[_-]?key=[^&\s\"']{8,}"),
)

_REDACTION_MARKERS = ("***", "REDACTED", "<omitted>")


def find_persisted_secrets(payload: Any, path: str = "$") -> list[str]:
    """Return paths where a credential value appears to have been persisted.

    Field *names* that merely describe a secret (``requires_user_api_key``,
    ``api_key_parameter_name``) are not violations; stored key material is.
    """
    problems: list[str] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{path}.{key}"
            if is_sensitive_key(key) and isinstance(value, str) and value.strip():
                if not any(marker in value for marker in _REDACTION_MARKERS):
                    problems.append(child)
            problems.extend(find_persisted_secrets(value, child))
        return problems

    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            problems.extend(find_persisted_secrets(item, f"{path}[{index}]"))
        return problems

    if isinstance(payload, str):
        if any(pattern.search(payload) for pattern in _SECRET_VALUE_PATTERNS):
            problems.append(path)

    return problems
