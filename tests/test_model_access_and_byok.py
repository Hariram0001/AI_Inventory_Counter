"""Model access policies, OpenRouter availability, BYOK routing and samples."""

from __future__ import annotations

import json

import pytest

import admin_samples
import database
import model_access
import openrouter
import user_store
from auth import to_authenticated_user
from detector import extract_workflow_error_status, translate_byok_error
from model_adapters import OpenRouterVLMAdapter, get_adapter, resolve_adapter_type
from openrouter import evaluate_openrouter_availability, verify_api_key
from schemas import ModelConfig

STRONG = "Str0ng!Passphrase42"
OPENROUTER_KEY = "workflow:hariram-s-mzhvc/playground-gpt-5-6-luna-od"


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "access.db")
    database.apply_migrations(path)
    return path


@pytest.fixture
def regular_user(db):
    return to_authenticated_user(
        user_store.create_user(
            username="regular",
            password=STRONG,
            role="user",
            force_password_change=False,
            db_path=db,
        )
    )


def openrouter_model() -> ModelConfig:
    return ModelConfig(
        name="OpenRouter VLM Detector",
        kind="workflow",
        enabled=True,
        workspace_name="hariram-s-mzhvc",
        workflow_id="playground-gpt-5-6-luna-od",
        key=OPENROUTER_KEY,
        provider="openrouter",
        supports_prompt=True,
        dynamic_classes=True,
        requires_user_api_key=True,
    )


def yolo_model() -> ModelConfig:
    return ModelConfig(
        name="YOLO-World",
        kind="workflow",
        enabled=True,
        workspace_name="hariram-s-mzhvc",
        workflow_id="custom-workflow",
        key="workflow:hariram-s-mzhvc/custom-workflow",
        supports_prompt=True,
        dynamic_classes=True,
    )


# ---------------------------------------------------------------------------
# Availability gate ordering
# ---------------------------------------------------------------------------


def base_kwargs(**overrides):
    kwargs = {
        "user_authenticated": True,
        "user_active": True,
        "policy_enabled": True,
        "global_enabled": True,
        "has_verified_key": True,
        "cost_notice_accepted": True,
        "workflow_metadata_valid": True,
        "inventory_supported": True,
        "quota_remaining": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_openrouter_available_when_every_condition_holds():
    assert evaluate_openrouter_availability(**base_kwargs()).available


@pytest.mark.parametrize(
    "override,expected_action",
    [
        ({"user_authenticated": False}, "sign_in"),
        ({"user_active": False}, "contact_admin"),
        ({"global_enabled": False}, "contact_admin"),
        ({"policy_enabled": False}, "contact_admin"),
        ({"workflow_metadata_valid": False}, "contact_admin"),
        ({"inventory_supported": False}, "change_inventory"),
        ({"has_verified_key": False}, "contact_admin"),
        ({"quota_remaining": 0}, "wait_quota"),
    ],
)
def test_each_condition_blocks_with_actionable_guidance(override, expected_action):
    decision = evaluate_openrouter_availability(**base_kwargs(**override))
    assert decision.blocked
    assert decision.action == expected_action
    assert decision.reason


def test_cost_notice_flag_is_ignored_for_availability():
    # Billing is on the admin key; users are not asked to accept a cost notice.
    decision = evaluate_openrouter_availability(
        **base_kwargs(cost_notice_accepted=False)
    )
    assert decision.available


def test_authentication_is_checked_before_key_presence():
    """The most actionable message wins when several conditions fail."""
    decision = evaluate_openrouter_availability(
        **base_kwargs(user_authenticated=False, has_verified_key=False)
    )
    assert decision.action == "sign_in"


# ---------------------------------------------------------------------------
# Policy-driven access
# ---------------------------------------------------------------------------


def _enable_openrouter(db) -> None:
    user_store.upsert_model_policy(
        OPENROUTER_KEY,
        is_enabled=True,
        requires_user_api_key=False,
        requires_cost_confirmation=False,
        db_path=db,
    )


def test_default_policies_are_seeded_once(db):
    model_access.ensure_default_policies(db)
    first = {p.model_key: p for p in user_store.list_model_policies(db)}
    assert OPENROUTER_KEY in first
    # Admin-managed key: no per-user BYOK; starts disabled until admin enables it.
    assert first[OPENROUTER_KEY].requires_user_api_key is False
    assert first[OPENROUTER_KEY].is_enabled is False

    user_store.upsert_model_policy(
        OPENROUTER_KEY, maximum_runs_per_user_per_day=3, db_path=db
    )
    model_access.ensure_default_policies(db)
    assert (
        user_store.get_model_policy(OPENROUTER_KEY, db_path=db).maximum_runs_per_user_per_day
        == 3
    )


def test_openrouter_blocked_without_admin_key_then_allowed_when_enabled(db, regular_user):
    model_access.ensure_default_policies(db)
    _enable_openrouter(db)
    model = openrouter_model()

    blocked = model_access.evaluate_model_access(
        model, regular_user, has_verified_key=False, db_path=db
    )
    assert not blocked.allowed
    assert blocked.action == "contact_admin"
    assert "administrator" in blocked.reason.lower()

    allowed = model_access.evaluate_model_access(
        model, regular_user, has_verified_key=True, db_path=db
    )
    assert allowed.allowed
    assert allowed.requires_user_api_key is False


def test_openrouter_stays_blocked_while_policy_disabled(db, regular_user):
    model_access.ensure_default_policies(db)
    # Default seed leaves OpenRouter disabled until an admin enables it.
    decision = model_access.evaluate_model_access(
        openrouter_model(),
        regular_user,
        has_verified_key=True,
        db_path=db,
    )
    assert not decision.allowed
    assert decision.action == "contact_admin"


def test_daily_quota_blocks_after_limit(db, regular_user):
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(
        OPENROUTER_KEY,
        is_enabled=True,
        maximum_runs_per_user_per_day=2,
        db_path=db,
    )
    model = openrouter_model()

    for _ in range(2):
        assert model_access.evaluate_model_access(
            model, regular_user, has_verified_key=True, db_path=db
        ).allowed
        model_access.register_run(regular_user, model, db_path=db)

    exhausted = model_access.evaluate_model_access(
        model, regular_user, has_verified_key=True, db_path=db
    )
    assert not exhausted.allowed
    assert exhausted.action == "wait_quota"
    assert exhausted.quota_remaining == 0


def test_role_restriction_hides_model_from_regular_users(db, regular_user):
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(
        "workflow:hariram-s-mzhvc/custom-workflow",
        allowed_roles=("admin",),
        db_path=db,
    )
    decision = model_access.evaluate_model_access(yolo_model(), regular_user, db_path=db)
    assert not decision.allowed
    assert decision.action == "contact_admin"


def test_disabled_policy_blocks_model(db, regular_user):
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(
        "workflow:hariram-s-mzhvc/custom-workflow", is_enabled=False, db_path=db
    )
    assert not model_access.evaluate_model_access(
        yolo_model(), regular_user, db_path=db
    ).allowed


def test_partition_models_splits_allowed_and_blocked(db, regular_user):
    model_access.ensure_default_policies(db)
    allowed, blocked = model_access.partition_models(
        [yolo_model(), openrouter_model()],
        regular_user,
        has_verified_key=False,
        cost_notice_accepted=False,
        db_path=db,
    )
    assert [m.name for m in allowed] == ["YOLO-World"]
    assert [m.name for m, _ in blocked] == ["OpenRouter VLM Detector"]


def test_anonymous_caller_is_never_allowed(db):
    assert not model_access.evaluate_model_access(yolo_model(), None, db_path=db).allowed


def test_incomplete_workflow_metadata_blocks_run(db, regular_user):
    model_access.ensure_default_policies(db)
    _enable_openrouter(db)
    broken = openrouter_model()
    broken.workflow_id = ""
    decision = model_access.evaluate_model_access(
        broken, regular_user, has_verified_key=True, db_path=db
    )
    assert not decision.allowed


# ---------------------------------------------------------------------------
# Key verification
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def patch_requests(monkeypatch, response=None, error: Exception | None = None):
    import requests

    def fake_get(url, headers=None, timeout=None):
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(requests, "get", fake_get)


def test_verify_key_rejects_wrong_format_without_network(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("verification must not call the network")

    import requests

    monkeypatch.setattr(requests, "get", explode)

    assert verify_api_key("").status == "invalid_format"
    assert verify_api_key("not-a-key").status == "invalid_format"


def test_verify_key_success_reports_sanitized_summary(monkeypatch):
    patch_requests(
        monkeypatch,
        FakeResponse(200, {"data": {"label": "demo", "limit": 10.0, "usage": 2.5}}),
    )
    key = "sk-or-v1-" + "a" * 32
    result = verify_api_key(key)

    assert result.verified
    assert result.label == "demo"
    public = result.to_public_dict()
    assert key not in json.dumps(public)
    assert "7.50" in result.message  # remaining credit


def test_verify_key_maps_error_statuses(monkeypatch):
    for status_code, expected in (
        (401, "unauthorized"),
        (403, "unauthorized"),
        (429, "rate_limited"),
        (500, "error"),
    ):
        patch_requests(monkeypatch, FakeResponse(status_code))
        result = verify_api_key("sk-or-v1-" + "b" * 32)
        assert result.status == expected
        assert not result.verified


def test_verify_key_handles_network_failure(monkeypatch):
    patch_requests(monkeypatch, error=OSError("no route to host"))
    result = verify_api_key("sk-or-v1-" + "c" * 32)
    assert result.status == "network_error"
    assert "no route to host" not in result.message


def test_key_format_detection():
    assert openrouter.looks_like_openrouter_key("sk-or-v1-" + "d" * 20)
    assert not openrouter.looks_like_openrouter_key("sk-or-v1-short")
    assert not openrouter.looks_like_openrouter_key(None)


# ---------------------------------------------------------------------------
# Adapter routing and error translation
# ---------------------------------------------------------------------------


def test_openrouter_model_routes_to_byok_adapter():
    model = openrouter_model()
    assert resolve_adapter_type(model) == "openrouter_vlm_detector"
    adapter = get_adapter(model, model_api_key="sk-or-v1-" + "e" * 32)
    assert isinstance(adapter, OpenRouterVLMAdapter)
    assert adapter.validate_configuration().ok is True


def test_metadata_only_catalog_does_not_hide_openrouter(tmp_path, monkeypatch):
    """Workspace sync used to mark Luna as metadata_only and hide it from Analyze."""
    from model_catalog import (
        ADAPTER_LEGACY_WORKFLOW,
        STATUS_METADATA_ONLY,
        CatalogEntry,
        _entry_is_analysis_ready,
        get_selectable_models,
        save_catalog_entries,
    )

    catalog_path = tmp_path / "model_catalog.json"
    monkeypatch.setattr("model_catalog.CATALOG_PATH", catalog_path)
    stale = CatalogEntry(
        key=OPENROUTER_KEY,
        display_name="Playground GPT-5.6 Luna (Object Detection) (openrouter)",
        source="workspace",
        provider="roboflow",
        task_type="object_detection",
        adapter_type=ADAPTER_LEGACY_WORKFLOW,
        workspace="hariram-s-mzhvc",
        workflow_id="playground-gpt-5-6-luna-od",
        enabled=True,
        validated=False,
        kind="workflow",
        status=STATUS_METADATA_ONLY,
    )
    save_catalog_entries([stale], backup=False)

    model = openrouter_model()
    # Even before migration, readiness must trust models.json OpenRouter metadata.
    assert _entry_is_analysis_ready(stale, model) is True
    assert resolve_adapter_type(model) == "openrouter_vlm_detector"

    names = {m.name for m in get_selectable_models("Fence Panel", allow_demo=False)}
    assert "OpenRouter VLM Detector" in names


def test_openrouter_adapter_refuses_without_key():
    adapter = get_adapter(openrouter_model(), model_api_key="")
    validation = adapter.validate_configuration()
    assert not validation.ok
    assert "administrator" in validation.message.lower()


def test_byok_workflow_passes_key_as_parameter_and_redacts_logs(caplog):
    """The key reaches the workflow parameters but never the log stream."""
    from detector import RoboflowDetector

    key = "sk-or-v1-" + "f" * 32
    captured = {}

    class FakeClient:
        def run_workflow(self, **kwargs):
            captured.update(kwargs)
            return [{"predictions": {"predictions": []}}]

    detector = RoboflowDetector(api_key="rf-key", demo_mode=False, model_api_key=key)
    with caplog.at_level("INFO"):
        detector._run_byok_workflow(
            FakeClient(), openrouter_model(), {"image": "/tmp/x.jpg"}, ["fence panel"]
        )

    assert captured["parameters"]["model_api_key"] == key
    assert captured["parameters"]["classes"] == ["fence panel"]
    assert captured["images"] == {"image": "/tmp/x.jpg"}
    assert key not in caplog.text
    assert "REDACTED" in caplog.text


def test_byok_workflow_requires_key_and_classes():
    from detector import DetectorError, RoboflowDetector

    class FakeClient:
        def run_workflow(self, **kwargs):
            raise AssertionError("must not be called")

    no_key = RoboflowDetector(api_key="rf", demo_mode=False, model_api_key="")
    with pytest.raises(DetectorError, match="administrator"):
        no_key.__class__._run_byok_workflow(
            no_key, FakeClient(), openrouter_model(), {"image": "x"}, ["panel"]
        )

    with_key = RoboflowDetector(
        api_key="rf", demo_mode=False, model_api_key="sk-or-v1-" + "g" * 32
    )
    with pytest.raises(DetectorError, match="detection class"):
        with_key._run_byok_workflow(
            FakeClient(), openrouter_model(), {"image": "x"}, []
        )


def test_workflow_error_status_is_extracted():
    assert extract_workflow_error_status([{"error_status": "rate limited"}]) == "rate limited"
    assert extract_workflow_error_status([{"error_status": False}]) == ""
    assert extract_workflow_error_status([{"outputs": [{"error_status": True}]}])
    assert extract_workflow_error_status([{"predictions": []}]) == ""


@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("401 Unauthorized", "Reconnect it in API Connections"),
        ("402 insufficient credit", "enough credit"),
        ("429 rate limit exceeded", "rate limiting"),
        ("Request timed out", "did not respond in time"),
        ("404 model_not_found", "not available to your account"),
        ("flagged by moderation", "content policy"),
        ("503 Service Unavailable", "temporarily unavailable"),
    ],
)
def test_byok_errors_translate_to_actionable_guidance(raw, expected_fragment):
    assert expected_fragment in translate_byok_error(raw)


def test_byok_error_translation_redacts_secrets():
    message = translate_byok_error(
        "failed for https://x?api_key=supersecretvalue123 Bearer eyJabcdefgh"
    )
    assert "supersecretvalue123" not in message
    assert "eyJabcdefgh" not in message


# ---------------------------------------------------------------------------
# Administrator samples
# ---------------------------------------------------------------------------


def png_bytes(width=128, height=128) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 140, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_sample_upload_validates_and_stores(db, tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    sample = admin_samples.add_sample(
        data=png_bytes(),
        title="Yard A Panels",
        inventory_type="Fence Panel",
        uploaded_by="admin",
        db_path=db,
    )
    assert sample.exists
    assert sample.width == 128
    assert sample.filename.endswith(".png")
    assert admin_samples.read_sample_bytes(sample)

    listed = admin_samples.list_samples(db_path=db)
    assert [s.sample_id for s in listed] == [sample.sample_id]


def test_sample_upload_rejects_non_images_and_oversize(db, tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)

    with pytest.raises(admin_samples.SampleValidationError):
        admin_samples.add_sample(
            data=b"<html>not an image</html>",
            title="Bad",
            inventory_type="Fence Panel",
            db_path=db,
        )
    with pytest.raises(admin_samples.SampleValidationError):
        admin_samples.add_sample(
            data=png_bytes(16, 16), title="Tiny", inventory_type="Fence Panel", db_path=db
        )
    with pytest.raises(admin_samples.SampleValidationError):
        admin_samples.add_sample(
            data=png_bytes(), title="  ", inventory_type="Fence Panel", db_path=db
        )


def test_sample_filenames_cannot_escape_the_sample_directory(db, tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    sample = admin_samples.add_sample(
        data=png_bytes(),
        title="../../etc/passwd",
        inventory_type="Fence Panel",
        db_path=db,
    )
    assert "/" not in sample.filename
    assert "\\" not in sample.filename
    assert ".." not in sample.filename
    assert sample.path.parent.resolve() == admin_samples.samples_dir().resolve()


def test_sample_update_and_delete(db, tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    sample = admin_samples.add_sample(
        data=png_bytes(), title="Keep", inventory_type="Fence Panel", db_path=db
    )
    updated = admin_samples.update_sample(sample.id, is_enabled=False, db_path=db)
    assert updated.is_enabled is False
    assert admin_samples.list_samples(include_disabled=False, db_path=db) == []

    path = sample.path
    assert admin_samples.delete_sample(sample.id, db_path=db) == sample.sample_id
    assert not path.exists()
    assert admin_samples.list_samples(db_path=db) == []


def test_slugify_produces_safe_tokens():
    assert admin_samples.slugify("Yard A / Panels #1") == "yard_a_panels_1"
    assert admin_samples.slugify("../../etc") == "etc"
    assert admin_samples.slugify("") == "sample"
