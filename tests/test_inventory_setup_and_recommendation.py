"""UI refinement tests: inventory cards, Add Photos status, simplified Analyze, settings ownership."""

from __future__ import annotations

import inspect
from pathlib import Path

import app as app_module
import config
from app_constants import PHOTO_REL_DISPLAY, PHOTO_REL_INTERNAL_TO_DISPLAY
from inventory_config import (
    FIXED_PHOTO_RELATIONSHIP,
    SELECTABLE_INVENTORY_KEY,
    inventory_display_name,
    is_inventory_selectable,
    resolve_recommended_model,
)
from model_registry import get_selectable_analysis_models, load_models_from_file
from schemas import ModelConfig
from ui_helpers import default_form, inject_css


def test_fence_panels_remains_clickable():
    src = inspect.getsource(app_module.stage_setup)
    assert 'key=f"inv_tile_{inv}"' in src
    assert is_inventory_selectable("Fence Panel")
    assert inventory_display_name("Fence Panel") == "Fence Panels"


def test_fence_panels_selected_state_is_red_primary():
    src = inspect.getsource(app_module.stage_setup)
    assert 'type="primary" if selected else "secondary"' in src
    css = inspect.getsource(inject_css)
    assert "rgba(255,75,75" in css or "aic-card-selected" in css


def test_disabled_cards_same_structure_with_red_indicator():
    src = inspect.getsource(app_module.stage_setup)
    assert "aic-inv-card" in src
    assert "aic-inv-unavailable" in src
    assert "Coming Soon" in src
    assert "title=\"Coming Soon\"" in src or "Coming Soon" in src
    # No clickable buttons for disabled types
    assert "is_inventory_selectable(inv)" in src


def test_disabled_cards_cannot_modify_state():
    src = inspect.getsource(app_module.stage_setup)
    # Only selectable branch writes inventory_choice via button callback
    assert "_form_set(inventory_choice=inv)" in src
    # Unavailable branch is markdown-only
    unavailable_block = src.split("else:")[-1]
    assert "_form_set" not in unavailable_block.split("inv_choice")[0]


def test_no_horizontal_scroll_inventory_grid():
    src = inspect.getsource(app_module.stage_setup)
    assert "st.columns" in src
    assert "overflow-x" not in src
    assert "horizontal scroll" not in src.lower()


def test_photo_relationship_still_fixed():
    form = default_form()
    assert form["photo_relationship"] == FIXED_PHOTO_RELATIONSHIP
    setup = inspect.getsource(app_module.stage_setup)
    assert "#### Photo Relationship" not in setup
    assert "Same inventory from multiple angles" not in setup
    internal = PHOTO_REL_DISPLAY["Same inventory from multiple angles"]
    assert PHOTO_REL_INTERNAL_TO_DISPLAY[internal]


def test_add_photos_supports_multi_upload_and_camera():
    src = inspect.getsource(app_module.stage_photos)
    assert "accept_multiple_files=True" in src
    assert "Upload Images" in src
    assert "Use Camera" in src
    assert "st.camera_input" in src
    assert "MAX_UPLOAD_BYTES" in src or "MB per file" in src


def test_add_photos_compact_green_status():
    src = inspect.getsource(app_module.stage_photos)
    assert "aic-photos-status" in src
    assert "Status:" in src
    assert "Selected inventory:" not in src  # no large repeated summary card


def test_analysis_has_no_recommendation_ui():
    src = inspect.getsource(app_module.stage_analyze)
    for banned in (
        "Find Best Model",
        "Recommended AI Setup",
        "Why was this selected",
        "Automatically selected",
        "Suggested model",
        "AI-assisted",
        "Accept suggestion",
    ):
        assert banned not in src
    assert not hasattr(app_module, "_render_recommended_ai_setup")
    assert not hasattr(app_module, "_run_find_best_model_trial")


def test_analysis_single_and_compare_modes():
    src = inspect.getsource(app_module.stage_analyze)
    assert "Single Model" in src
    assert "Compare Models" in src
    assert "Run Analysis" in src
    assert "Run Comparison" in src
    assert (
        "At least two configured and validated models are required for comparison"
        in src
    )


def test_comparison_requires_two_valid_models():
    models = load_models_from_file()
    selectable = get_selectable_analysis_models(
        models, SELECTABLE_INVENTORY_KEY, allow_demo=False
    )
    # Live POC: Roboflow + optional Local Picket; demo fixtures excluded.
    assert len(selectable) >= 1
    names = {m.name for m in selectable}
    assert "Demo Fence Detector" not in names
    assert "Local Picket Counter" in names
    assert "YOLO-World" in names


def test_demo_models_excluded_from_live_selector():
    models = load_models_from_file()
    selectable = get_selectable_analysis_models(
        models, "Fence Panel", allow_demo=False
    )
    assert all(not m.is_demo_model_id() for m in selectable)
    assert "Local Picker" not in {m.name for m in selectable}
    assert "Ticket Counter" not in {m.name for m in selectable}
    # Local classical counter is optional and selectable when enabled
    assert "Local Picket Counter" in {m.name for m in selectable}
    assert "YOLO-World" in {m.name for m in selectable}
    assert "YOLO-World Fence Panel" not in {m.name for m in selectable}


def test_local_picker_and_ticket_counter_do_not_exist_as_models():
    blob = Path("models.json").read_text(encoding="utf-8")
    assert "Local Picker" not in blob
    assert "Ticket Counter" not in blob
    assert "Watch Demo" not in blob
    # Closest real names
    assert "Local Picket Counter" in blob
    assert "Demo Fence Detector" in blob


def test_watch_demo_not_in_working_ui():
    welcome = inspect.getsource(app_module.view_welcome)
    analyze = inspect.getsource(app_module.stage_analyze)
    assert "Watch Demo" not in welcome
    assert "Watch Demo" not in analyze


def test_settings_ownership_no_duplicate_model_history_in_diagnostics():
    ai = inspect.getsource(app_module._render_ai_configuration_section)
    hist = inspect.getsource(app_module._render_history_section)
    diag = inspect.getsource(app_module._render_diagnostics_section)
    assert "Model test history" in ai
    assert "Model Catalog" in ai or "render_model_catalog_section" in ai
    assert "Inventory History" in hist
    assert "model_test_results" not in diag
    assert "summarize_models" not in diag


def test_demo_mode_false_does_not_return_mock_via_detector_guard():
    from detector import RoboflowDetector

    assert config.DEMO_MODE is False or isinstance(config.DEMO_MODE, bool)
    # Live path rejects demo model ids when not in demo mode
    demo = ModelConfig(
        name="Demo Fence Detector",
        kind="model",
        enabled=True,
        model_id="demo-fence-panels/1",
    )
    assert demo.is_demo_model_id()
    det = RoboflowDetector(demo_mode=False, api_key="x")
    try:
        det.run_direct_model(demo, image_path="noop.jpg")
        raised = False
    except Exception as exc:  # noqa: BLE001
        raised = True
        assert "demo" in str(exc).lower() or "DEMO_MODE" in str(exc)
    assert raised


def test_backend_default_resolution_still_works():
    models = load_models_from_file()
    resolved = resolve_recommended_model(
        "Fence Panel", models, config.INVENTORY_MODEL_RECOMMENDATIONS, allow_demo=False
    )
    assert resolved["ok"]
    assert resolved["model_name"] in {m.name for m in models if m.enabled}
