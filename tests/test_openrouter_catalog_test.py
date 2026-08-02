"""Offline tests for OpenRouter Catalog Test credential mapping and preflight."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import app as app_module
import database
import model_access
import openrouter_runtime as orun
import openrouter_store
import user_store
from auth import to_authenticated_user
from detector import RoboflowDetector, translate_byok_error
from model_adapters import get_adapter
from schemas import ModelConfig

STRONG = "Str0ng!Passphrase42"
OPENROUTER_KEY = "workflow:hariram-s-mzhvc/playground-gpt-5-6-luna-od"


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = str(tmp_path / "catalog_or.db")
    database.apply_migrations(path)
    import config

    monkeypatch.setattr(config, "DB_PATH", path)
    return path


@pytest.fixture
def admin_user(db):
    return to_authenticated_user(
        user_store.create_user(
            username="adminor",
            password=STRONG,
            role="admin",
            force_password_change=False,
            db_path=db,
        )
    )


@pytest.fixture
def regular_user(db):
    return to_authenticated_user(
        user_store.create_user(
            username="regularor",
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
        prompt_parameter_name="classes",
        api_key_parameter_name="model_api_key",
        image_input_name="image",
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


def local_model() -> ModelConfig:
    return ModelConfig(
        name="Local Picket Counter",
        kind="local",
        enabled=True,
        model_id="local-picket-counter",
        key="local:local-picket-counter",
    )


def _save_key(db, *, verified: bool = True):
    openrouter_store.save_deployment_key(
        "sk-or-v1-" + "a" * 32,
        verification={
            "verified": verified,
            "masked": "sk-o…aaaa",
            "credit_limit": 10.0,
            "usage": 0.0,
            "is_free_tier": True,
        },
        updated_by="adminor",
        db_path=db,
    )


def test_analyze_and_catalog_share_same_secure_accessor():
    analyze_src = inspect.getsource(app_module)
    assert "get_openrouter_inference_key" in analyze_src
    # Catalog Test and Analyze both use the shared runtime accessor — not ad-hoc loaders.
    assert analyze_src.count("get_openrouter_inference_key") >= 2
    runtime_src = inspect.getsource(orun.get_openrouter_inference_key)
    assert "get_deployment_key" in runtime_src
    assert "models.json" not in runtime_src


def test_verified_key_reaches_adapter(db, monkeypatch):
    orun._FALLBACK_SESSION.clear()
    _save_key(db)
    monkeypatch.setattr(
        orun,
        "get_deployment_key",
        lambda db_path=None: openrouter_store.get_deployment_key(db),
    )
    key = orun.get_openrouter_inference_key()
    assert key.startswith("sk-or-v1-")
    adapter = get_adapter(openrouter_model(), model_api_key=key)
    assert adapter.validate_configuration().ok
    assert getattr(adapter, "model_api_key", "") == key


def test_no_key_blocks_before_inference(db, monkeypatch):
    orun._FALLBACK_SESSION.clear()
    monkeypatch.setattr(orun, "has_verified_deployment_key", lambda db_path=None: False)
    monkeypatch.setattr(orun, "get_deployment_key", lambda db_path=None: "")
    model = openrouter_model()
    adapter = get_adapter(model, model_api_key="")
    assert not adapter.validate_configuration().ok
    assert not orun.openrouter_credential_ready()
    pre = orun.preflight_openrouter_catalog_test(
        model,
        SimpleNamespace(user_id=1, is_active=True, role="admin", username="a"),
        has_test_image=True,
        paid_confirmed=True,
        fetch_schema=False,
        db_path=db,
    )
    assert not pre.ok
    assert pre.reason_code == "missing_key"


def test_key_not_readable_from_catalog_files():
    src = inspect.getsource(orun.get_openrouter_inference_key)
    assert "model_catalog.json" not in src
    assert "models.json" not in src
    assert "OPENROUTER_API_KEY" not in src


def test_redacted_parameters_never_include_key():
    params = orun.redacted_workflow_parameters(
        image_name="probe.jpg",
        classes=["cardboard box", "shipping box", "package box"],
    )
    blob = str(params)
    assert "sk-or" not in blob
    assert params["model_api_key"] == "[REDACTED]"
    assert params["classes"][0] == "cardboard box"


def test_stale_credential_failure_becomes_ready_to_test(db, tmp_path, monkeypatch):
    from model_catalog import (
        STATUS_FAILED,
        CatalogEntry,
        load_catalog_entries,
        save_catalog_entries,
    )

    catalog_path = tmp_path / "model_catalog.json"
    monkeypatch.setattr(
        "model_catalog.CATALOG_PATH", catalog_path, raising=False
    )
    # Patch load/save to use tmp catalog via monkeypatch on module paths used by runtime.
    entry = CatalogEntry(
        key=OPENROUTER_KEY,
        display_name="OpenRouter VLM Detector",
        source="foundation",
        provider="openrouter",
        adapter_type="openrouter_vlm_detector",
        workflow_id="playground-gpt-5-6-luna-od",
        workspace="hariram-s-mzhvc",
        requires_user_api_key=True,
        status=STATUS_FAILED,
        validated=False,
        last_test_status="Failed",
        validation_message="OpenRouter is not configured. An administrator must add an API key.",
    )
    entry.normalize_schema_fields()

    saved = {"entries": [entry]}

    monkeypatch.setattr(
        "model_catalog.load_catalog_entries", lambda: list(saved["entries"])
    )

    def _save(entries):
        saved["entries"] = list(entries)

    monkeypatch.setattr("model_catalog.save_catalog_entries", _save)
    _save_key(db)
    real_get = openrouter_store.get_deployment_key
    monkeypatch.setattr(orun, "has_verified_deployment_key", lambda db_path=None: True)
    monkeypatch.setattr(orun, "get_deployment_key", lambda db_path=None: real_get(db))
    orun.clear_stale_credential_test_state(OPENROUTER_KEY)
    refreshed = saved["entries"][0]
    assert refreshed.last_test_status == "Ready to test"
    assert refreshed.validated is False
    state = orun.get_user_model_test_state(OPENROUTER_KEY)
    assert state.test_status == "ready_to_test"


def test_reverification_does_not_auto_live_validate(db, monkeypatch):
    _save_key(db)
    monkeypatch.setattr(
        openrouter_store,
        "has_verified_deployment_key",
        lambda db_path=None: True,
    )
    orun.clear_stale_credential_test_state(OPENROUTER_KEY)
    state = orun.get_user_model_test_state(OPENROUTER_KEY)
    assert state.test_status == "ready_to_test"
    assert state.available_for_analyze is False


def test_published_predictions_required(monkeypatch):
    report = orun.WorkflowSchemaReport(
        ok=False,
        workflow_id="playground-gpt-5-6-luna-od",
        output_names=("label_visualization",),
        has_image_input=True,
        has_classes_input=True,
        has_model_api_key_input=True,
        has_predictions_output=False,
        message=orun.PUBLISH_PREDICTIONS_MESSAGE,
    )
    monkeypatch.setattr(orun, "inspect_published_workflow_schema", lambda *a, **k: report)
    model_access.ensure_default_policies(db_path=None)
    user_store.upsert_model_policy(OPENROUTER_KEY, is_enabled=True)
    user = SimpleNamespace(user_id=1, is_active=True, role="admin", username="a")
    monkeypatch.setattr(orun, "openrouter_credential_ready", lambda: True)
    pre = orun.preflight_openrouter_catalog_test(
        openrouter_model(),
        user,
        has_test_image=True,
        paid_confirmed=True,
        fetch_schema=True,
    )
    assert not pre.ok
    assert pre.reason_code == "predictions_not_published"
    assert "published Serverless version" in pre.message


def test_visualization_only_payload_rejected():
    assert orun.payload_has_predictions([{"label_visualization": {"width": 1}}]) is False
    assert orun.payload_has_label_visualization(
        [{"label_visualization": {"width": 1}}]
    )
    assert orun.payload_has_predictions(
        [{"predictions": {"predictions": []}}]
    )


def test_explicit_paid_confirmation_required(monkeypatch, db):
    monkeypatch.setattr(orun, "openrouter_credential_ready", lambda: True)
    monkeypatch.setattr(
        orun,
        "inspect_published_workflow_schema",
        lambda *a, **k: orun.WorkflowSchemaReport(
            ok=True,
            has_image_input=True,
            has_classes_input=True,
            has_model_api_key_input=True,
            has_predictions_output=True,
            output_names=("predictions", "error_status", "label_visualization"),
        ),
    )
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(OPENROUTER_KEY, is_enabled=True, db_path=db)
    user = SimpleNamespace(user_id=1, is_active=True, role="admin", username="a")
    pre = orun.preflight_openrouter_catalog_test(
        openrouter_model(),
        user,
        has_test_image=True,
        paid_confirmed=False,
        fetch_schema=True,
        db_path=db,
    )
    assert not pre.ok
    assert pre.reason_code == "confirmation_required"


def test_preflight_failure_does_not_increment_usage(db, regular_user, monkeypatch):
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(OPENROUTER_KEY, is_enabled=True, db_path=db)
    before = user_store.get_usage_count(regular_user.user_id, OPENROUTER_KEY, db_path=db)
    monkeypatch.setattr(orun, "openrouter_credential_ready", lambda: False)
    pre = orun.preflight_openrouter_catalog_test(
        openrouter_model(),
        regular_user,
        has_test_image=True,
        paid_confirmed=True,
        fetch_schema=False,
        db_path=db,
    )
    assert not pre.ok
    after = user_store.get_usage_count(regular_user.user_id, OPENROUTER_KEY, db_path=db)
    assert after == before


def test_register_run_increments_once(db, regular_user):
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(OPENROUTER_KEY, is_enabled=True, db_path=db)
    assert model_access.register_run(regular_user, openrouter_model(), db_path=db) == 1
    assert model_access.register_run(regular_user, openrouter_model(), db_path=db) == 2


def test_auth_rejection_clears_only_session_key(db, monkeypatch):
    orun._FALLBACK_SESSION.clear()
    _save_key(db)
    real_get = openrouter_store.get_deployment_key
    monkeypatch.setattr(orun, "get_deployment_key", lambda db_path=None: real_get(db))
    assert orun.get_openrouter_inference_key()
    orun.mark_session_key_rejected(reason="401 Unauthorized")
    assert (orun._FALLBACK_SESSION.get(orun.SESSION_REJECTED_KEY) or {}).get("rejected")
    assert orun.get_openrouter_inference_key() == ""
    # Deployment secret remains for other sessions/users until admin reconnects.
    assert real_get(db)
    orun.clear_session_key_rejection()


def test_translate_auth_message_mentions_reconnect():
    msg = translate_byok_error("401 Unauthorized")
    assert "Reconnect it in API Connections" in msg


def test_byok_workflow_rejects_visualization_only():
    class FakeClient:
        def run_workflow(self, **kwargs):
            return [{"label_visualization": {"image": "x"}}]

    det = RoboflowDetector(
        api_key="rf", demo_mode=False, model_api_key="sk-or-v1-" + "b" * 32
    )
    with pytest.raises(Exception, match="did not return predictions"):
        det._run_byok_workflow(
            FakeClient(), openrouter_model(), {"image": "x.jpg"}, ["cardboard box"]
        )


def test_byok_workflow_accepts_zero_detections():
    class FakeClient:
        def run_workflow(self, **kwargs):
            assert kwargs["parameters"]["model_api_key"].startswith("sk-or-")
            assert kwargs["parameters"]["classes"] == ["cardboard box"]
            return [{"predictions": {"predictions": []}, "error_status": False}]

    det = RoboflowDetector(
        api_key="rf", demo_mode=False, model_api_key="sk-or-v1-" + "c" * 32
    )
    payload = det._run_byok_workflow(
        FakeClient(), openrouter_model(), {"image": "x.jpg"}, ["cardboard box"]
    )
    assert orun.payload_has_predictions(payload)
    assert det.last_raw_prediction_count == 0


def test_yolo_and_local_paths_unchanged():
    from model_adapters import resolve_adapter_type

    assert resolve_adapter_type(yolo_model()) == "yolo_world_workflow"
    assert resolve_adapter_type(local_model()) == "local_classical"
    assert resolve_adapter_type(openrouter_model()) == "openrouter_vlm_detector"


def test_error_status_sanitized():
    class FakeClient:
        def run_workflow(self, **kwargs):
            return [{"predictions": {"predictions": []}, "error_status": "sk-or-v1-SHOULDNOTLEAK"}]

    det = RoboflowDetector(
        api_key="rf", demo_mode=False, model_api_key="sk-or-v1-" + "d" * 32
    )
    with pytest.raises(Exception) as excinfo:
        det._run_byok_workflow(
            FakeClient(), openrouter_model(), {"image": "x.jpg"}, ["box"]
        )
    assert "sk-or-v1-SHOULDNOTLEAK" not in str(excinfo.value)


def test_one_user_without_key_cannot_pass_preflight(db, regular_user, monkeypatch):
    model_access.ensure_default_policies(db)
    user_store.upsert_model_policy(OPENROUTER_KEY, is_enabled=True, db_path=db)
    monkeypatch.setattr(orun, "openrouter_credential_ready", lambda: False)
    pre = orun.preflight_openrouter_catalog_test(
        openrouter_model(),
        regular_user,
        has_test_image=True,
        paid_confirmed=True,
        fetch_schema=False,
        db_path=db,
    )
    assert not pre.ok
    assert pre.reason_code == "missing_key"
