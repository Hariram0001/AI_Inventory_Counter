"""Model access decisions combining admin policy, BYOK state and daily quotas.

This is the single place that decides whether a signed-in user may select and
run a given model, so the wizard, the admin console and the tests all agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import config
from auth import EVENT_QUOTA_BLOCKED, ROLE_ADMIN, ROLE_USER, AuthenticatedUser
from openrouter import evaluate_openrouter_availability, is_openrouter_model
from user_store import (
    ModelAccessPolicy,
    get_model_policy,
    get_usage_count,
    increment_usage,
    list_model_policies,
    record_audit_event,
    upsert_model_policy,
)

# Seeded on first run so administrators have something concrete to edit.
DEFAULT_POLICY_SEEDS: tuple[dict[str, Any], ...] = (
    {
        "model_key": "workflow:hariram-s-mzhvc/custom-workflow",
        "display_name": "YOLO-World",
        "is_enabled": True,
        "allowed_roles": (ROLE_ADMIN, ROLE_USER),
        "requires_user_api_key": False,
        "requires_cost_confirmation": False,
        "maximum_runs_per_user_per_day": None,
        "notes": "Shared Roboflow workflow billed to the deployment.",
    },
    {
        "model_key": "local:local-picket-counter",
        "display_name": "Local Picket Counter",
        "is_enabled": True,
        "allowed_roles": (ROLE_ADMIN, ROLE_USER),
        "requires_user_api_key": False,
        "requires_cost_confirmation": False,
        "maximum_runs_per_user_per_day": None,
        "notes": "Runs locally; no external API calls or cost.",
    },
    {
        "model_key": "workflow:hariram-s-mzhvc/playground-gpt-5-6-luna-od",
        "display_name": "OpenRouter VLM Detector",
        "is_enabled": True,
        "allowed_roles": (ROLE_ADMIN, ROLE_USER),
        "requires_user_api_key": True,
        "requires_cost_confirmation": True,
        "maximum_runs_per_user_per_day": 25,
        "notes": "Billed to each user's own OpenRouter account.",
    },
)


@dataclass(frozen=True)
class AccessDecision:
    """Whether a model may be offered, and what the user should do if not."""

    model_key: str
    allowed: bool
    reason: str = ""
    action: str = ""
    requires_user_api_key: bool = False
    requires_cost_confirmation: bool = False
    quota_limit: int | None = None
    quota_used: int = 0

    @property
    def quota_remaining(self) -> int | None:
        if self.quota_limit is None:
            return None
        return max(0, self.quota_limit - self.quota_used)


def ensure_default_policies(db_path: str | None = None) -> None:
    """Create the seed policies once; never overwrite administrator edits."""
    existing = {policy.model_key for policy in list_model_policies(db_path)}
    for seed in DEFAULT_POLICY_SEEDS:
        if seed["model_key"] in existing:
            continue
        upsert_model_policy(
            seed["model_key"],
            display_name=seed["display_name"],
            is_enabled=seed["is_enabled"],
            allowed_roles=seed["allowed_roles"],
            requires_user_api_key=seed["requires_user_api_key"],
            requires_cost_confirmation=seed["requires_cost_confirmation"],
            maximum_runs_per_user_per_day=seed["maximum_runs_per_user_per_day"],
            notes=seed["notes"],
            updated_by="system",
            db_path=db_path,
        )


def resolve_model_key(model: Any) -> str:
    """Catalog key for a ModelConfig, falling back to the derived key."""
    key = str(getattr(model, "key", "") or "").strip()
    if key:
        return key
    from model_adapters import model_key as derive_key

    return derive_key(model)


def evaluate_model_access(
    model: Any,
    user: AuthenticatedUser | None,
    *,
    inventory_key: str | None = None,
    has_verified_key: bool = False,
    cost_notice_accepted: bool = False,
    db_path: str | None = None,
) -> AccessDecision:
    """Decide whether ``user`` may run ``model`` right now."""
    key = resolve_model_key(model)
    policy = get_model_policy(key, db_path=db_path) or ModelAccessPolicy(model_key=key)
    byok = bool(getattr(model, "requires_user_api_key", False)) or bool(
        policy.requires_user_api_key
    )
    needs_cost = bool(policy.requires_cost_confirmation) or byok

    if user is None:
        return AccessDecision(
            model_key=key,
            allowed=False,
            reason="Sign in to use this model.",
            action="sign_in",
            requires_user_api_key=byok,
            requires_cost_confirmation=needs_cost,
        )

    quota_limit = policy.maximum_runs_per_user_per_day
    quota_used = (
        get_usage_count(user.user_id, key, db_path=db_path)
        if quota_limit is not None
        else 0
    )
    quota_remaining = None if quota_limit is None else max(0, quota_limit - quota_used)

    def decide(allowed: bool, reason: str = "", action: str = "") -> AccessDecision:
        return AccessDecision(
            model_key=key,
            allowed=allowed,
            reason=reason,
            action=action,
            requires_user_api_key=byok,
            requires_cost_confirmation=needs_cost,
            quota_limit=quota_limit,
            quota_used=quota_used,
        )

    if is_openrouter_model(model) or byok:
        inventory_ok = _inventory_supported(model, inventory_key)
        decision = evaluate_openrouter_availability(
            user_authenticated=True,
            user_active=user.is_active,
            policy_enabled=policy.is_enabled and policy.allows_role(user.role),
            has_verified_key=has_verified_key,
            cost_notice_accepted=cost_notice_accepted or not needs_cost,
            workflow_metadata_valid=_workflow_metadata_valid(model),
            inventory_supported=inventory_ok,
            quota_remaining=quota_remaining,
        )
        return decide(decision.available, decision.reason, decision.action)

    if not user.is_active:
        return decide(False, "Your account is deactivated.", "contact_admin")
    if not policy.is_enabled:
        return decide(
            False, "An administrator has disabled this model.", "contact_admin"
        )
    if not policy.allows_role(user.role):
        return decide(
            False, "This model is not available for your role.", "contact_admin"
        )
    if not _inventory_supported(model, inventory_key):
        return decide(
            False,
            "This model does not support the selected inventory type.",
            "change_inventory",
        )
    if quota_remaining is not None and quota_remaining <= 0:
        return decide(False, "You have reached today's run limit for this model.", "wait_quota")

    return decide(True)


def _workflow_metadata_valid(model: Any) -> bool:
    kind = str(getattr(model, "kind", "") or "").lower()
    if kind != "workflow":
        return True
    workspace = str(getattr(model, "workspace_name", "") or "").strip()
    workflow_id = str(getattr(model, "workflow_id", "") or "").strip()
    if not workspace or not workflow_id:
        return False
    return "replace-with" not in workspace and "replace-with" not in workflow_id


def _inventory_supported(model: Any, inventory_key: str | None) -> bool:
    if not inventory_key:
        return True
    supported = list(getattr(model, "supported_inventory_types", None) or [])
    if not supported:
        # An empty list means open-vocabulary / any inventory type.
        return True
    return inventory_key in supported


def partition_models(
    models: Iterable[Any],
    user: AuthenticatedUser | None,
    *,
    inventory_key: str | None = None,
    has_verified_key: bool = False,
    cost_notice_accepted: bool = False,
    db_path: str | None = None,
) -> tuple[list[Any], list[tuple[Any, AccessDecision]]]:
    """Split models into those the user may run and those with a blocking reason."""
    allowed: list[Any] = []
    blocked: list[tuple[Any, AccessDecision]] = []
    for model in models:
        decision = evaluate_model_access(
            model,
            user,
            inventory_key=inventory_key,
            has_verified_key=has_verified_key,
            cost_notice_accepted=cost_notice_accepted,
            db_path=db_path,
        )
        if decision.allowed:
            allowed.append(model)
        else:
            blocked.append((model, decision))
    return allowed, blocked


def register_run(
    user: AuthenticatedUser,
    model: Any,
    *,
    images: int = 1,
    db_path: str | None = None,
) -> int:
    """Record a completed run against the user's daily quota."""
    key = resolve_model_key(model)
    return increment_usage(user.user_id, key, amount=max(1, int(images)), db_path=db_path)


def note_quota_block(user: AuthenticatedUser, decision: AccessDecision) -> None:
    record_audit_event(
        EVENT_QUOTA_BLOCKED,
        actor_user_id=user.user_id,
        actor_username=user.username,
        target_type="model",
        target_id=decision.model_key,
        outcome="failure",
        detail={
            "quota_limit": decision.quota_limit,
            "quota_used": decision.quota_used,
        },
    )


def openrouter_globally_enabled() -> bool:
    return bool(getattr(config, "OPENROUTER_MODELS_ENABLED", True))
