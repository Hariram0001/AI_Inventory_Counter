"""OpenRouter bring-your-own-key verification and availability rules.

Key verification uses OpenRouter's free key-introspection endpoint, so
verifying a key never performs a paid inference call. The plaintext key is
held only in Streamlit session state by ``auth_session`` and is never written
to disk, logs or the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
from security import mask_secret, redact_text

COST_NOTICE_TITLE = "OpenRouter usage is billed to your own account"

COST_NOTICE_BODY = (
    "Running an OpenRouter vision model sends your image to OpenRouter using "
    "the API key you provided. Each run consumes credits on **your** OpenRouter "
    "account and is billed to you, not to this application. Costs depend on the "
    "model, image size and number of photos analysed. Verifying a key does not "
    "cost anything — charges begin only when you run an analysis."
)

COST_NOTICE_POINTS = (
    "Your key is kept in this browser session only and is never saved to disk.",
    "Your key is cleared when you sign out or the session times out.",
    "Every analysis run with an OpenRouter model consumes your credits.",
    "Administrators can cap how many OpenRouter runs you may perform per day.",
)

_KEY_PREFIXES = ("sk-or-v1-", "sk-or-")
_MIN_KEY_LENGTH = 20


@dataclass(frozen=True)
class KeyVerification:
    """Sanitized verification outcome. Contains no key material."""

    verified: bool
    status: str  # verified | invalid_format | unauthorized | rate_limited | network_error | error
    message: str
    masked_key: str = ""
    label: str = ""
    credit_limit: float | None = None
    usage: float | None = None
    is_free_tier: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "status": self.status,
            "message": self.message,
            "masked": self.masked_key,
            "label": self.label,
            "credit_limit": self.credit_limit,
            "usage": self.usage,
            "is_free_tier": self.is_free_tier,
        }


def looks_like_openrouter_key(raw: str | None) -> bool:
    text = str(raw or "").strip()
    if len(text) < _MIN_KEY_LENGTH:
        return False
    return any(text.startswith(prefix) for prefix in _KEY_PREFIXES)


def verify_api_key(api_key: str, *, timeout: float = 15.0) -> KeyVerification:
    """Check a key against OpenRouter's free key endpoint.

    Never raises and never echoes the key. Callers should store the plaintext
    key only through ``auth_session.set_openrouter_key``.
    """
    key = str(api_key or "").strip()
    masked = mask_secret(key)

    if not key:
        return KeyVerification(
            verified=False,
            status="invalid_format",
            message="Enter an OpenRouter API key.",
        )
    if not looks_like_openrouter_key(key):
        return KeyVerification(
            verified=False,
            status="invalid_format",
            masked_key=masked,
            message=(
                "That does not look like an OpenRouter API key. Keys start with "
                "'sk-or-'. Copy the key from openrouter.ai/keys."
            ),
        )

    url = getattr(config, "OPENROUTER_KEY_VERIFY_URL", "https://openrouter.ai/api/v1/key")
    try:
        import requests

        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — network stack errors vary widely
        return KeyVerification(
            verified=False,
            status="network_error",
            masked_key=masked,
            message=(
                "Could not reach OpenRouter to verify the key. Check your network "
                "connection and try again."
            ),
            details={"error": redact_text(f"{type(exc).__name__}", max_len=120)},
        )

    if response.status_code in (401, 403):
        return KeyVerification(
            verified=False,
            status="unauthorized",
            masked_key=masked,
            message=(
                "OpenRouter rejected this key. Confirm it is active and has not "
                "been revoked at openrouter.ai/keys."
            ),
        )
    if response.status_code == 429:
        return KeyVerification(
            verified=False,
            status="rate_limited",
            masked_key=masked,
            message="OpenRouter is rate limiting this key. Try again in a moment.",
        )
    if response.status_code >= 400:
        return KeyVerification(
            verified=False,
            status="error",
            masked_key=masked,
            message=(
                f"OpenRouter returned an unexpected response ({response.status_code}). "
                "Try again shortly."
            ),
        )

    try:
        payload = response.json() or {}
    except Exception:  # noqa: BLE001
        payload = {}
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else {}

    limit = _as_float(data.get("limit"))
    usage = _as_float(data.get("usage"))
    label = str(data.get("label") or "").strip()
    is_free = data.get("is_free_tier")

    remaining_note = ""
    if limit is not None:
        remaining = max(0.0, limit - (usage or 0.0))
        remaining_note = f" Approximately ${remaining:.2f} of credit remains."
    elif usage is not None:
        remaining_note = f" Usage so far: ${usage:.2f}."

    return KeyVerification(
        verified=True,
        status="verified",
        masked_key=masked,
        label=label,
        credit_limit=limit,
        usage=usage,
        is_free_tier=bool(is_free) if is_free is not None else None,
        message="Key verified. No charges were incurred by this check." + remaining_note,
    )


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def is_openrouter_model(model: Any) -> bool:
    """True when a ModelConfig / CatalogEntry routes through OpenRouter."""
    if model is None:
        return False
    if bool(getattr(model, "requires_user_api_key", False)):
        return True
    provider = str(getattr(model, "provider", "") or "").lower()
    if "openrouter" in provider:
        return True
    name = str(getattr(model, "name", "") or getattr(model, "display_name", "")).lower()
    workflow_id = str(getattr(model, "workflow_id", "") or "").lower()
    adapter = str(getattr(model, "adapter_type", "") or "").lower()
    configured = str(getattr(config, "OPENROUTER_WORKFLOW_ID", "") or "").lower()
    if adapter == "openrouter_vlm_detector":
        return True
    if configured and workflow_id == configured:
        return True
    return "openrouter" in name or "openrouter" in workflow_id


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AvailabilityDecision:
    available: bool
    reason: str = ""
    action: str = ""

    @property
    def blocked(self) -> bool:
        return not self.available


def evaluate_openrouter_availability(
    *,
    user_authenticated: bool,
    user_active: bool,
    policy_enabled: bool,
    global_enabled: bool | None = None,
    has_verified_key: bool,
    cost_notice_accepted: bool,
    workflow_metadata_valid: bool,
    inventory_supported: bool,
    quota_remaining: int | None = None,
) -> AvailabilityDecision:
    """Ordered gate check for offering an OpenRouter model to a user.

    The first failing condition wins so the user sees the most actionable
    message rather than a generic refusal.
    """
    if global_enabled is None:
        global_enabled = bool(getattr(config, "OPENROUTER_MODELS_ENABLED", True))

    if not user_authenticated:
        return AvailabilityDecision(False, "Sign in to use OpenRouter models.", "sign_in")
    if not user_active:
        return AvailabilityDecision(
            False, "Your account is deactivated. Contact an administrator.", "contact_admin"
        )
    if not global_enabled:
        return AvailabilityDecision(
            False, "OpenRouter models are disabled for this deployment.", "contact_admin"
        )
    if not policy_enabled:
        return AvailabilityDecision(
            False,
            "An administrator has disabled this model for your role.",
            "contact_admin",
        )
    if not workflow_metadata_valid:
        return AvailabilityDecision(
            False,
            "This model's workflow configuration is incomplete and cannot run.",
            "contact_admin",
        )
    if not inventory_supported:
        return AvailabilityDecision(
            False,
            "This model does not support the selected inventory type.",
            "change_inventory",
        )
    if not has_verified_key:
        return AvailabilityDecision(
            False,
            "Add and verify your OpenRouter API key to use this model.",
            "add_key",
        )
    if not cost_notice_accepted:
        return AvailabilityDecision(
            False,
            "Review and accept the OpenRouter cost notice before running this model.",
            "accept_cost_notice",
        )
    if quota_remaining is not None and quota_remaining <= 0:
        return AvailabilityDecision(
            False,
            "You have reached today's run limit for this model.",
            "wait_quota",
        )
    return AvailabilityDecision(True)
