"""Multi-item Custom Item: one-scan analysis + type focus in Review."""

from __future__ import annotations

import inspect

from inventory_profiles import (
    AnalysisRunContext,
    build_run_context,
    class_names_for_primary_type,
    parse_custom_item_specs,
    preset_primary_and_aliases,
)
from image_processing import annotate_image
from PIL import Image
from schemas import Detection


def test_class_names_for_primary_type_isolates_aliases():
    specs, _ = parse_custom_item_specs(
        "traffic cone\nbarrier",
        "traffic cone: road cone, safety cone",
    )
    assert len(specs) == 2
    ctx = AnalysisRunContext(
        inventory_key="Custom Item",
        inventory_display_name="Custom",
        counting_unit="individual items by type",
        effective_prompts=["traffic cone", "road cone", "safety cone", "barrier"],
        primary_item_types=["traffic cone", "barrier"],
        class_alias_map={
            "traffic cone": "traffic cone",
            "road cone": "traffic cone",
            "safety cone": "traffic cone",
            "barrier": "barrier",
        },
    )
    cone = class_names_for_primary_type(ctx, "traffic cone")
    assert cone[0].casefold() == "traffic cone"
    assert "road cone" in cone
    assert "barrier" not in {c.casefold() for c in cone}
    bar = class_names_for_primary_type(ctx, "barrier")
    assert bar == ["barrier"] or bar[0].casefold() == "barrier"


def test_preset_primary_collapses_synonym_terms():
    primary, alias_map = preset_primary_and_aliases(
        "Fence Panel",
        ["fence panel", "wooden fence panel", "privacy fence panel"],
    )
    assert primary  # display name or key
    assert len({alias_map[k] for k in alias_map}) == 1
    assert alias_map["fence panel"] == primary
    assert alias_map["wooden fence panel"] == primary

    ctx, errs = build_run_context(inventory_key="Fence Panel")
    assert not errs
    assert ctx is not None
    assert len(ctx.primary_item_types) == 1
    assert len(ctx.effective_prompts) >= 2
    for term in ctx.effective_prompts:
        assert ctx.class_alias_map[term.casefold()] == ctx.primary_item_types[0]


def test_custom_multi_keeps_n_primaries_one_prompt_list():
    ctx, errs = build_run_context(
        inventory_key="Custom Item",
        custom_item_name="traffic cone\nbarrier",
        custom_alternatives="traffic cone: road cone",
    )
    assert ctx is not None
    assert ctx.primary_item_types == ["traffic cone", "barrier"]
    assert "road cone" in ctx.effective_prompts
    assert "barrier" in ctx.effective_prompts


def test_execute_analysis_is_one_pass():
    import app as app_module

    src = inspect.getsource(app_module._execute_analysis_run)
    assert "type_passes" not in src
    assert "per_type_passes" not in src
    assert "class_names_for_primary_type" not in src
    assert '"per_type_runs": False' in src or "'per_type_runs': False" in src
    assert "One inference per image" in src or "all primary types" in src


def test_review_type_focus_on_single_result():
    import app as app_module

    src = inspect.getsource(app_module.stage_review)
    assert "Show all markers on the picture" in src
    assert "solo=solo_canvas" in src
    assert "rev_type_focus_" in src
    assert "rev_type_pass_" not in src
    assert "rev_det_prev_top" in src
    assert "rev_excl_top" in src
    assert "Exclude this item" in src
    assert "_toggle_review_detection_exclusion" in src
    assert "One scan found all" in src or "switch between types" in src.lower()


def test_annotate_solo_draws_only_selected():
    img = Image.new("RGB", (200, 200), (240, 240, 240))
    dets = [
        Detection(
            detection_id="a",
            class_name="cone",
            confidence=0.9,
            x1=10,
            y1=10,
            x2=50,
            y2=50,
            center_x=30,
            center_y=30,
            width=40,
            height=40,
            source_model="t",
            source_image="t.jpg",
            marker_number=1,
        ),
        Detection(
            detection_id="b",
            class_name="cone",
            confidence=0.8,
            x1=120,
            y1=120,
            x2=160,
            y2=160,
            center_x=140,
            center_y=140,
            width=40,
            height=40,
            source_model="t",
            source_image="t.jpg",
            marker_number=2,
        ),
    ]
    out = annotate_image(
        img,
        dets,
        model_name="t",
        style="markers",
        selected_detection_id="a",
        solo=True,
        show_legend=False,
    )
    assert out.size == img.size
