"""Multi-item Custom Item: separate analysis passes + solo annotate."""

from __future__ import annotations

import inspect

from inventory_profiles import (
    AnalysisRunContext,
    class_names_for_primary_type,
    parse_custom_item_specs,
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


def test_execute_analysis_loops_per_type():
    import app as app_module

    src = inspect.getsource(app_module._execute_analysis_run)
    assert "per_type_passes" in src
    assert "class_names_for_primary_type" in src
    assert "item_type_pass" in src
    assert "type_passes" in src


def test_review_defaults_to_solo_focus():
    import app as app_module

    src = inspect.getsource(app_module.stage_review)
    assert "Show all markers on the picture" in src
    assert "solo=solo_canvas" in src
    assert "Results by item type" in src
    assert "rev_det_prev_top" in src
    assert "rev_excl_top" in src
    assert "Exclude this item" in src
    assert "_toggle_review_detection_exclusion" in src
    assert "per_type_runs" in src


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
            x2=180,
            y2=180,
            center_x=150,
            center_y=150,
            width=60,
            height=60,
            source_model="t",
            source_image="t.jpg",
            marker_number=2,
        ),
    ]
    solo = annotate_image(
        img, dets, style="markers", selected_detection_id="a", solo=True
    )
    both = annotate_image(
        img, dets, style="markers", selected_detection_id="a", solo=False
    )
    assert solo.size == both.size
    # Solo and full overlays must differ when two detections exist.
    assert list(solo.getdata()) != list(both.getdata())
