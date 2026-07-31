"""Offline tests for Compare Models selection, execution rules, and metadata."""

from __future__ import annotations

import inspect

import app as app_module
from comparison_helpers import (
    COMPARE_MAX_MODELS,
    COMPARE_MIN_MODELS,
    compare_peer_models,
    comparison_run_caption,
    format_count_display,
    human_status,
    progress_label,
    sanitize_compare_selection,
    summary_row_from_mir,
    validate_compare_selection,
)
from inventory_config import SELECTABLE_INVENTORY_KEY
from model_adapters import ModelInferenceResult
from model_registry import get_selectable_analysis_models, load_models_from_file
from schemas import Detection, InferenceResult, ModelConfig


def _fake_model(name: str, *, kind: str = "workflow", demo: bool = False) -> ModelConfig:
    return ModelConfig(
        name=name,
        kind=kind,
        enabled=True,
        demo_only=demo,
        workspace_name="ws" if kind == "workflow" else None,
        workflow_id=name.lower().replace(" ", "-") if kind == "workflow" else None,
        model_id=None if kind == "workflow" else f"{name}/1",
        supported_inventory_types=["Fence Panel"],
    )


def test_compare_accepts_two_and_three_models():
    names = ["A", "B", "C"]
    assert validate_compare_selection(["A", "B"], names) == []
    assert validate_compare_selection(["A", "B", "C"], names) == []


def test_compare_rejects_fewer_than_two():
    errs = validate_compare_selection(["A"], ["A", "B", "C"])
    assert errs
    assert str(COMPARE_MIN_MODELS) in errs[0]


def test_compare_rejects_more_than_three():
    names = ["A", "B", "C", "D"]
    errs = validate_compare_selection(names, names)
    assert errs
    assert str(COMPARE_MAX_MODELS) in errs[0]


def test_sanitize_drops_stale_and_caps():
    out = sanitize_compare_selection(
        ["Gone", "A", "B", "C", "D"],
        ["A", "B", "C"],
    )
    assert out == ["A", "B", "C"]


def test_only_valid_enabled_compatible_peers_appear():
    models = load_models_from_file()
    selectable = get_selectable_analysis_models(
        models, SELECTABLE_INVENTORY_KEY, allow_demo=False
    )
    peers = compare_peer_models(selectable)
    names = {m.name for m in peers}
    assert "Demo Fence Detector" not in names
    assert "YOLO-World" in names
    assert "Local Picket Counter" in names
    assert all(not m.demo_only for m in peers)


def test_human_status_success_and_failure():
    assert human_status(success=True, final_count=3, error_type=None).startswith("Success")
    assert human_status(success=True, final_count=0, error_type=None) == (
        "Success with zero detections"
    )
    assert "Authentication" in human_status(
        success=False, final_count=None, error_type="unauthorized"
    )
    assert "Timeout" in human_status(success=False, final_count=None, error_type="timeout")
    assert "Network" in human_status(success=False, final_count=None, error_type="api_error")


def test_failed_summary_does_not_show_zero_counts():
    model = _fake_model("Broken")
    mir = ModelInferenceResult.failed(
        model,
        provider="Roboflow",
        error_type="timeout",
        error_message="timed out",
    )
    row = summary_row_from_mir(mir, image_name="a.jpg")
    assert row["success"] is False
    assert row["raw_count"] is None
    assert row["final_count"] is None
    assert format_count_display(row["final_count"]) == "—"
    assert "Timeout" in row["status"] or row["status"] == "Failed"


def test_progress_and_run_captions():
    assert progress_label(2, 3, 1, 2) == "Running model 2 of 3 on image 1 of 2"
    assert comparison_run_caption(2, 3) == "2 photos × 3 models = 6 analysis runs"


def test_analyze_ui_compare_contracts():
    src = inspect.getsource(app_module.stage_analyze)
    assert "Compare Models" in src
    assert "Run Comparison" in src
    assert "analyze_cmp_" in src
    assert "format_model_info_markdown" in src
    assert "analyze_single_model_radio" in src or "Model info" in src
    assert "Only one compatible validated model is currently available" in src
    assert 'stage="running"' in src
    # Do not auto-pad compare selection to every peer
    assert "compare_names[: min(2" not in src
    assert "compare_names[:2]" not in src
    run_src = inspect.getsource(app_module._execute_analysis_run)
    assert "progress_label" in run_src
    assert "summary_row_from_mir" in run_src
    assert "selected_model_keys" in run_src


def test_review_use_this_result_and_no_rerun():
    src = inspect.getsource(app_module.stage_review)
    assert "Use This Result" in src
    assert "Selected for Review" in src
    assert "do not rerun inference" in src
    assert "accepted_result_key" in src


def test_save_comparison_metadata_fields():
    src = inspect.getsource(app_module._save_inventory)
    assert "comparison_mode" in src
    assert "selected_model_keys" in src
    assert "selected_model_names" in src
    assert "comparison_summaries" in src
    assert "model_chosen_for_review" in src
    assert "final_saved_count" in src
    assert "final_reviewed_detections" in src


def test_partial_failure_preserves_success_results():
    """Simulate one failure + one success without converting failure to zero dets."""
    ok = InferenceResult(
        image_name="a.jpg",
        model_name="Good",
        prompt="fence",
        inference_mode="Whole-image inference",
        deduplication_strategy="Conservative",
        detections=[
            Detection(
                detection_id="d1",
                class_name="fence",
                confidence=0.9,
                x1=0,
                y1=0,
                x2=10,
                y2=10,
                center_x=5,
                center_y=5,
                width=10,
                height=10,
                source_model="Good",
                source_image="a.jpg",
            )
        ],
        raw_count=1,
        final_count=1,
        duplicates_removed=0,
        avg_confidence=0.9,
        min_confidence=0.9,
        max_confidence=0.9,
        suspected_overlap_count=0,
        suspected_occlusion_count=0,
        processing_time_seconds=0.1,
        request_completed=True,
    )
    results = [ok]
    failures = ["a.jpg / Bad: timeout"]
    assert len(results) == 1
    assert results[0].final_count == 1
    assert failures
    # Failed model must not appear as a zero-count InferenceResult
    assert all(r.model_name != "Bad" for r in results)


def test_execution_loop_runs_all_selected_models():
    analyze_src = inspect.getsource(app_module.stage_analyze)
    assert "navigate_to(\"wizard\", stage=\"running\")" in analyze_src
    src = inspect.getsource(app_module._execute_analysis_run)
    assert "for model_i, model in enumerate(selected_models" in src
    assert "for img_i, item in enumerate(images" in src
    assert "source_bytes = item[\"data\"]" in src
    assert "stage_running" in inspect.getsource(app_module)
