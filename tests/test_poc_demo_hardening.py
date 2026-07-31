"""Offline tests for stakeholder demo hardening (no network)."""

from __future__ import annotations

import inspect
from pathlib import Path

import app as app_module
from inventory_profiles import parse_custom_prompts, prompt_is_unsafe
from poc_ux import (
    CONN_CONFIG_MISSING,
    CONN_NOT_TESTED,
    classify_user_error,
    connection_status_payload,
    list_demo_sample_cards,
    resolve_connection_label,
    sanitize_public_text,
    stamp_connection_probe,
)
from sample_images import clear_sample_library_cache, list_enabled_samples
from ui_helpers import default_form, reset_active_analysis


def test_demo_cards_only_verified_samples():
    clear_sample_library_cache()
    cards = list_demo_sample_cards()
    ids = {c["sample_id"] for c in cards}
    assert "fence_picket_panel_01" in ids
    assert "fence_gate_driveway_01" in ids
    assert "cardboard" not in " ".join(ids).lower()
    for c in cards:
        assert c["inventory_key"] in {"Fence Panel", "Gates"}
        assert Path(c["path"]).is_file()


def test_sample_inventory_keys_canonical():
    clear_sample_library_cache()
    fence = list_enabled_samples(inventory_key="Fence Panel")
    gates = list_enabled_samples(inventory_key="Gates")
    assert any(s.id == "fence_picket_panel_01" for s in fence)
    assert any(s.id == "fence_gate_driveway_01" for s in gates)


def test_home_sample_does_not_auto_run_inference():
    src = inspect.getsource(app_module._start_demo_sample)
    assert "navigate_to" in src
    assert 'stage="photos"' in src
    assert "run_inference" not in src
    assert "get_adapter" not in src
    assert "_execute_analysis_run" not in src
    welcome = inspect.getsource(app_module.view_welcome)
    assert "Try a Sample" in welcome
    assert "Get Started" in welcome
    assert "Workflow ID" not in welcome
    assert "POC_NOTICE" in welcome or "proof of concept" in welcome.lower()


def test_error_sanitization_redacts_secrets():
    text = sanitize_public_text("failed api_key=SUPERSECRET123 Authorization: Bearer abc.def")
    assert "SUPERSECRET123" not in text
    assert "abc.def" not in text
    assert "***" in text
    err = classify_user_error(error_type="unauthorized", message="api_key=XYZ")
    assert err.title == "Authentication failure"
    assert "XYZ" not in err.detail
    zero = classify_user_error(success_zero=True)
    assert "did not find matching objects" in zero.message.lower()


def test_empty_state_messages_in_analyze():
    src = inspect.getsource(app_module.stage_analyze)
    assert "No compatible validated model is available" in src
    assert "Only one compatible validated model is currently available" in src


def test_connection_test_isolation_contract():
    src = inspect.getsource(app_module.render_configuration_summary)
    assert "Test Connection" in src
    assert "wizard_guard" in src
    assert "connection_probe" in src
    assert "uploaded_images" in src
    label = resolve_connection_label(api_configured=False, last_probe=None)
    assert label == CONN_CONFIG_MISSING
    assert resolve_connection_label(api_configured=True, last_probe=None) == CONN_NOT_TESTED
    payload = connection_status_payload(
        api_configured=True,
        workspace="hariram-s-mzhvc",
        workflow_available=True,
        validated_model_count=2,
        last_probe=None,
    )
    assert payload["validated_models"] == 2
    stamped = stamp_connection_probe(
        {"ok": True, "auth": "Successful", "message": "api_key=ABC", "workflow": "YOLO"}
    )
    assert "ABC" not in stamped["message"]
    assert "api_key" not in stamped or stamped.get("api_key") is None


def test_history_isolation_caption():
    src = inspect.getsource(app_module._render_history_section)
    assert "does not rerun inference" in src
    assert "No inventory counts have been saved yet" in src or "saved yet" in src.lower()


def test_custom_item_rejects_html():
    assert prompt_is_unsafe("<script>alert(1)</script>")
    assert prompt_is_unsafe("box onload=evil")
    prompts, errors = parse_custom_prompts("<b>box</b>", None)
    assert not prompts
    assert errors


def test_duplicate_inference_protection():
    src = inspect.getsource(app_module.stage_analyze)
    assert "analyze_running" in src
    assert "analysis_run_id" in src
    run_src = inspect.getsource(app_module.stage_running)
    assert "_analysis_executing" in src or "_analysis_executing" in run_src


def test_project_relative_paths():
    from config import DATA_DIR, MODELS_JSON_PATH, PROJECT_ROOT

    assert MODELS_JSON_PATH.is_relative_to(PROJECT_ROOT) or str(MODELS_JSON_PATH).startswith(
        str(PROJECT_ROOT)
    )
    assert DATA_DIR.is_relative_to(PROJECT_ROOT) or str(DATA_DIR).startswith(str(PROJECT_ROOT))


def test_reset_supports_no_rerun(monkeypatch):
    """reset_active_analysis(rerun=False) must not call st.rerun."""
    calls = {"rerun": 0}

    class _SS(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class FakeST:
        session_state = _SS(
            form=default_form(),
            uploader_nonce=0,
            review_state={},
            uploaded_images=[{"id": "x"}],
            analysis_results=[1],
        )

        def rerun(self):
            calls["rerun"] += 1

    import ui_helpers

    monkeypatch.setattr(ui_helpers, "_st", lambda: FakeST())
    reset_active_analysis(go_home=False, start_wizard=False, rerun=False)
    assert calls["rerun"] == 0
    assert FakeST.session_state["uploaded_images"] == []
    assert FakeST.session_state.get("analysis_run_id") is None


def test_acceptance_docs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "POC_ACCEPTANCE_CHECKLIST.md").is_file()
    assert (root / "docs" / "STAKEHOLDER_DEMO_SCRIPT.md").is_file()
    text = (root / "docs" / "POC_ACCEPTANCE_CHECKLIST.md").read_text(encoding="utf-8")
    assert "Deferred" in text
    assert "YOLO-World" in text


def test_progress_phases_present():
    from poc_ux import PROGRESS_PHASES

    src = inspect.getsource(app_module._execute_analysis_run)
    for phase in PROGRESS_PHASES:
        assert phase.split()[0] in src or "progress_phase_label" in src
    assert "compare_progress_caption" in src
