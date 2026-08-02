"""Review visualization, marker colors, and compact UI tests (offline)."""

from __future__ import annotations

import inspect
import io

from PIL import Image

import app as app_module
from detection_viz import (
    DETECTION_COLOR_PALETTE,
    assign_marker_numbers,
    color_for_detection,
    color_for_detection_id,
    contrasting_text_color,
)
from image_processing import annotate_image
from schemas import Detection
from ui_helpers import inject_css


def _det(i: int, *, det_id: str | None = None) -> Detection:
    return Detection(
        detection_id=det_id or f"det-{i}",
        class_name="fence panel",
        confidence=0.8,
        x1=10 + i * 5,
        y1=10,
        x2=40 + i * 5,
        y2=80,
        center_x=25 + i * 5,
        center_y=45,
        width=30,
        height=70,
        source_model="m",
        source_image="i.jpg",
    )


def test_marker_colors_stable_for_same_id():
    a = color_for_detection_id("abc-123")
    b = color_for_detection_id("abc-123")
    assert a == b
    assert a in DETECTION_COLOR_PALETTE


def test_box_and_marker_use_same_color_palette():
    d = _det(1, det_id="same-id")
    c1 = color_for_detection(d, 1)
    img = Image.new("RGB", (120, 100), color=(220, 220, 220))
    d.marker_number = 1
    out = annotate_image(img, [d], style="both", selected_detection_id=None)
    assert out.size == img.size
    # Selected style does not change dimensions
    out2 = annotate_image(img, [d], style="both", selected_detection_id="same-id")
    assert out2.size == img.size
    assert c1 == color_for_detection_id("same-id")


def test_aspect_ratio_preserved_on_annotate():
    portrait = Image.new("RGB", (80, 200), color=(200, 200, 200))
    landscape = Image.new("RGB", (200, 80), color=(200, 200, 200))
    d = _det(0)
    d.marker_number = 1
    assert annotate_image(portrait, [d], style="markers").size == portrait.size
    assert annotate_image(landscape, [d], style="boxes").size == landscape.size


def test_no_large_black_legend_by_default():
    img = Image.new("RGB", (100, 100), color=(180, 180, 180))
    # Empty detections: with show_legend=False the canvas must stay the source gray
    out = annotate_image(img, [], style="both", show_legend=False)
    px = out.getpixel((20, 20))
    assert px == (180, 180, 180)
    src = inspect.getsource(annotate_image)
    assert "fill=(30, 30, 30)" not in src


def test_visualization_switch_does_not_need_inference():
    src = inspect.getsource(app_module.stage_review)
    assert "Roboflow Labels" in src
    assert "Bounding Boxes" in src
    assert "Numbered Markers" in src
    assert "Both" in src
    assert "run_inference" not in src
    assert "adapter.predict" not in src


def test_roboflow_style_draws_class_label_chip():
    img = Image.new("RGB", (240, 180), color=(210, 210, 210))
    d = _det(1, det_id="rf-1")
    d.class_name = "traffic_cone"
    d.y1 = 40  # leave room for a label chip above the box
    d.marker_number = 1
    out = annotate_image(img, [d], style="roboflow", selected_detection_id="rf-1")
    assert out.size == img.size
    # Label chip uses class color fill just above the box — not only a center dot.
    chip_px = out.getpixel((int(d.x1) + 6, int(d.y1) - 8))
    assert chip_px != (210, 210, 210)
    # Box outline should also change pixels on the border.
    edge_px = out.getpixel((int(d.x1), int((d.y1 + d.y2) / 2)))
    assert edge_px != (210, 210, 210)
    src = inspect.getsource(annotate_image)
    assert "roboflow" in src.lower()
    assert "color_for_class" in src


def test_review_layout_uses_wide_canvas():
    src = inspect.getsource(app_module.stage_review)
    assert "aic-review-layout" in src
    assert "aic-review-canvas" in src
    assert "[2.55, 1.0]" in src or "2.55" in src
    css = inspect.getsource(inject_css)
    assert "align-items: flex-start" in css
    assert "aic-review-canvas" in css


def test_review_hides_internal_id_by_default():
    src = inspect.getsource(app_module.stage_review)
    assert "Details" in src
    assert "selected.detection_id" in src or "detection_id" in src
    # Primary list formatting should not lead with raw id=
    assert 'id={tag}' not in src
    assert "detection_id[:8]" not in src


def test_review_uses_tabs_and_single_active_image():
    src = inspect.getsource(app_module.stage_review)
    assert 'st.tabs(' in src
    assert "Detection" in src
    assert "Adjust" in src
    assert "Issues" in src
    assert "review_active_image" in src
    assert "review_active_model" in src
    assert "aic-review-canvas" in src
    assert "Full-width annotated image" not in src


def test_analysis_preview_keeps_source_image():
    src = inspect.getsource(app_module._execute_analysis_run)
    assert "_show_analysis_preview" in src
    assert "aic-img-card" in src
    assert "Analyzing image" in src


def test_css_removes_black_image_container_patterns():
    css = inspect.getsource(inject_css)
    assert "aic-img-card" in css
    assert "object-fit: contain" in css
    assert "background: #000" not in css
    assert "background-color: black" not in css


def test_duplicate_and_warning_use_visible_numbers():
    src = inspect.getsource(app_module.stage_review)
    assert (
        "#{selected.marker_number}" in src
        or "#{d.marker_number}" in src
        or "marker_number" in src
    )
    assert "possible duplicate" in src.lower() or "possible duplicate" in src


def test_marker_numbers_assigned_consistently():
    dets = [_det(3), _det(1), _det(2)]
    ordered = assign_marker_numbers(dets)
    assert [d.marker_number for d in ordered] == [1, 2, 3]


def test_contrasting_text_for_light_and_dark():
    assert contrasting_text_color((255, 255, 0)) == (20, 20, 20)
    assert contrasting_text_color((20, 20, 120)) == (255, 255, 255)


def test_manual_marker_still_drawable():
    img = Image.new("RGB", (100, 100), color=(210, 210, 210))
    d = _det(9, det_id="manual_abc")
    d.is_manual = True
    d.marker_number = 1
    out = annotate_image(img, [d], style="markers")
    assert out.size == img.size
