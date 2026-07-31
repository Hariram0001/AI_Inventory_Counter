"""Offline tests for dynamic inventory profiles and YOLO-World prompts."""

from __future__ import annotations

import inspect
import json

import config
import detector
from inventory_config import (
    inventory_display_name,
    is_inventory_selectable,
    resolve_recommended_model,
)
from inventory_profiles import (
    MAX_PROMPT_LEN,
    MAX_PROMPTS,
    AnalysisRunContext,
    build_run_context,
    clear_profiles_cache,
    effective_prompts_for_inventory,
    load_inventory_profiles,
    normalize_prompts,
    parse_custom_prompts,
    prompts_to_csv,
    validate_prompts,
)
from model_registry import load_models_from_file


def setup_function() -> None:
    clear_profiles_cache()


def test_presets_loaded_and_selectable():
    profiles = load_inventory_profiles(force_reload=True)
    keys = {p["key"] for p in profiles}
    for expected in (
        "Fence Panel",
        "Pallets",
        "Boxes",
        "Poles",
        "Gates",
        "Chairs",
        "Traffic Cones",
        "Custom Item",
    ):
        assert expected in keys
        assert is_inventory_selectable(expected)
    assert inventory_display_name("Fence Panel") == "Fence Panels"
    assert inventory_display_name("Traffic Cones") == "Traffic Cones"


def test_custom_item_requires_name():
    prompts, errs = parse_custom_prompts("", "road cone")
    assert prompts == []
    assert errs


def test_custom_item_alternatives_and_dedupe():
    prompts, errs = parse_custom_prompts(
        "traffic cone",
        "road cone, safety cone, Traffic Cone,  road cone ",
    )
    assert not errs
    assert prompts[0].casefold() == "traffic cone"
    # case-insensitive de-dupe of Traffic Cone / traffic cone
    folded = [p.casefold() for p in prompts]
    assert folded.count("traffic cone") == 1
    assert "road cone" in folded
    assert "safety cone" in folded


def test_prompt_length_and_count_limits():
    too_long = "x" * (MAX_PROMPT_LEN + 5)
    prompts, errs = validate_prompts([too_long, "ok term"])
    assert errs
    assert "ok term" in prompts

    many = [f"item {i}" for i in range(MAX_PROMPTS + 3)]
    prompts, errs = validate_prompts(many)
    assert len(prompts) <= MAX_PROMPTS
    assert any("at most" in e.lower() for e in errs)


def test_normalize_prompts_trims_and_dedupes():
    out = normalize_prompts(["  A  ", "a", "B", "b ", "", "C"])
    assert out == ["A", "B", "C"]


def test_reject_html_in_custom_prompts():
    prompts, errs = parse_custom_prompts('<script>alert(1)</script>', None)
    assert prompts == [] or errs
    assert errs


def test_effective_prompts_for_preset_not_empty():
    prompts, errs = effective_prompts_for_inventory("Pallets")
    assert not errs
    assert "wooden pallet" in [p.casefold() for p in prompts]


def test_yolo_world_adapter_uses_dynamic_prompt_helper():
    src = inspect.getsource(detector.prompt_to_class_names)
    assert "normalize_prompts" in src or "inventory_profiles" in src
    assert "fence panel" not in src.lower()
    # Injection path still present
    assert "inject_class_names_into_workflow_spec" in inspect.getsource(detector)
    names = detector.prompt_to_class_names("traffic cone, road cone, traffic cone")
    assert names == ["traffic cone", "road cone"]


def test_run_context_persistence_shape():
    ctx, errs = build_run_context(
        inventory_key="Traffic Cones",
        selected_model_key="workflow:test/custom-workflow",
        selected_model_display_name="YOLO-World",
        confidence_threshold=0.3,
        uploaded_image_ids=["img-a", "img-b"],
    )
    assert not errs or ctx is not None
    assert ctx is not None
    assert ctx.selected_model_display_name == "YOLO-World"
    assert "fence" not in ctx.inventory_display_name.casefold()
    blob = ctx.to_dict()
    # JSON-serializable for AIC_META
    json.dumps(blob)
    restored = AnalysisRunContext.from_dict(blob)
    assert restored is not None
    assert restored.effective_prompts == ctx.effective_prompts
    assert restored.uploaded_image_ids == ["img-a", "img-b"]


def test_custom_run_context_display_and_unit():
    ctx, errs = build_run_context(
        inventory_key="Custom Item",
        custom_item_name="traffic cone",
        custom_alternatives="road cone",
        selected_model_display_name="YOLO-World",
    )
    assert ctx is not None
    assert not errs or ctx.effective_prompts
    assert "traffic cone" in ctx.inventory_display_name.casefold()
    assert "traffic cone" in ctx.counting_unit.casefold()
    assert ctx.custom_item_name == "traffic cone"


def test_old_fence_panels_history_key_still_resolves():
    """Older records store inventory_type='Fence Panel' — must still resolve."""
    assert is_inventory_selectable("Fence Panel")
    assert inventory_display_name("Fence Panel") == "Fence Panels"
    prompts, errs = effective_prompts_for_inventory("Fence Panel")
    assert prompts
    assert not errs
    models = load_models_from_file()
    resolved = resolve_recommended_model(
        "Fence Panel",
        models,
        getattr(config, "INVENTORY_MODEL_RECOMMENDATIONS", {}),
        allow_demo=False,
    )
    assert resolved.get("ok")
    assert resolved.get("model_name") == "YOLO-World"


def test_model_display_name_is_generic_yolo_world():
    models = load_models_from_file()
    yolo = next(m for m in models if m.name == "YOLO-World")
    assert "Fence" not in yolo.name
    assert yolo.supports_prompt or yolo.dynamic_classes


def test_no_secret_leakage_in_profiles_module():
    import inventory_profiles as mod

    src = inspect.getsource(mod)
    for banned in ("ROBOFLOW_API_KEY", "api_key=", "sk-", "password"):
        assert banned not in src


def test_prompts_to_csv_roundtrip():
    assert prompts_to_csv(["a", "b"]) == "a, b"


def test_app_passes_run_context_not_hardcoded_fence_in_execute():
    import app as app_module

    src = inspect.getsource(app_module._execute_analysis_run)
    assert "run_context" in src or "run_ctx" in src
    assert 'prompt = "fence panel"' not in src
    analyze = inspect.getsource(app_module.stage_analyze)
    assert "Detection terms" in analyze or "effective_prompts" in analyze
    assert "Find Best Model" not in analyze
