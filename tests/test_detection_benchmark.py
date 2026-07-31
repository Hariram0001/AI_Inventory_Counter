"""Offline tests for the Detection Benchmark workflow."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import app as app_module
import benchmark as bench
import sample_images as sample_mod
from benchmark import (
    MAX_PROMPT_SETS,
    BenchmarkRunOutcome,
    compute_benchmark_metrics,
    compute_detection_counts,
    filter_benchmark_history,
    load_benchmark_results,
    parse_benchmark_metadata,
    parse_prompt_sets,
    save_benchmark_result,
    sanitize_benchmark_record,
    update_profile_prompt_terms,
    validate_expected_count,
)
from inventory_profiles import clear_profiles_cache, prompts_to_csv
from sample_images import clear_sample_library_cache, load_sample_library


def test_expected_count_validation():
    assert validate_expected_count(0) == (0, [])
    assert validate_expected_count(8)[0] == 8
    n, errs = validate_expected_count(-1)
    assert n is None and errs
    n, errs = validate_expected_count(None)
    assert n is None and errs
    n, errs = validate_expected_count("x")
    assert n is None and errs


def test_zero_expected_count_metrics():
    m = compute_benchmark_metrics(
        ai_count=0,
        expected_count=0,
        labels=[],
        missed_count=0,
    )
    assert m.evaluation == "successful_zero_detections"
    assert m.count_error == 0
    assert m.count_accuracy == 1.0

    m2 = compute_benchmark_metrics(
        ai_count=3,
        expected_count=0,
        labels=["false_positive"] * 3,
        missed_count=0,
    )
    assert m2.evaluation == "overcount"
    assert m2.count_accuracy == 0.0


def test_prompt_set_parsing_and_max_three():
    sets, errs = parse_prompt_sets(
        [
            "traffic cone",
            "traffic cone, road cone, safety cone",
            "individual traffic cone\norange traffic cone\nroad safety cone",
            "fourth should be truncated",
        ]
    )
    assert len(sets) == MAX_PROMPT_SETS
    assert any("At most" in e for e in errs)
    assert sets[0] == ["traffic cone"]
    assert "road cone" in sets[1]


def test_independent_prompt_set_execution_shape():
    """Each prompt set produces a separate outcome; detections are not merged."""
    a = BenchmarkRunOutcome(
        prompt_set_label="Set A",
        prompt_set=["traffic cone"],
        final_count=2,
        detections=[{"class_name": "traffic cone"}, {"class_name": "traffic cone"}],
        success=True,
    )
    b = BenchmarkRunOutcome(
        prompt_set_label="Set B",
        prompt_set=["traffic cone", "road cone"],
        final_count=1,
        detections=[{"class_name": "road cone"}],
        success=True,
    )
    a.apply_review(expected_count=2, labels=["correct", "correct"], missed_count=0)
    b.apply_review(expected_count=2, labels=["correct"], missed_count=1)
    assert a.final_count != b.final_count or a.prompt_set != b.prompt_set
    assert len(a.detections) + len(b.detections) == 3  # not merged into one list


def test_metric_calculations_precision_recall_count_error():
    tp, fp, fn = compute_detection_counts(
        ["correct", "correct", "false_positive", "wrong_class", "ignore"],
        missed_count=2,
    )
    assert (tp, fp, fn) == (2, 2, 2)
    m = compute_benchmark_metrics(
        ai_count=5,
        expected_count=4,
        labels=["correct", "correct", "false_positive", "wrong_class", "duplicate"],
        missed_count=1,
    )
    assert m.true_positives == 2
    assert m.false_positives == 3
    assert m.false_negatives == 1
    assert m.precision == pytest.approx(2 / 5)
    assert m.recall == pytest.approx(2 / 3)
    assert m.count_error == 1
    assert m.count_accuracy == pytest.approx(1 - 1 / 4)
    assert m.evaluation == "overcount"


def test_exact_match_and_undercount():
    m = compute_benchmark_metrics(
        ai_count=3,
        expected_count=3,
        labels=["correct"] * 3,
        missed_count=0,
    )
    assert m.evaluation == "exact_count_match"
    m2 = compute_benchmark_metrics(
        ai_count=1,
        expected_count=4,
        labels=["correct"],
        missed_count=3,
    )
    assert m2.evaluation == "undercount"


def test_execution_failed_vs_successful_zero():
    failed = compute_benchmark_metrics(
        ai_count=0,
        expected_count=5,
        execution_failed=True,
        request_completed=False,
    )
    assert failed.evaluation == "execution_failed"
    zero = compute_benchmark_metrics(
        ai_count=0,
        expected_count=0,
        execution_failed=False,
        request_completed=True,
    )
    assert zero.evaluation == "successful_zero_detections"


def test_benchmark_storage_and_history(tmp_path: Path):
    path = tmp_path / "benchmarks.json"
    saved = save_benchmark_result(
        {
            "inventory_key": "Traffic Cones",
            "prompt_set": ["traffic cone"],
            "prompt_set_label": "Set A",
            "expected_count": 8,
            "final_count": 7,
            "true_positives": 7,
            "false_positives": 0,
            "false_negatives": 1,
            "precision": 1.0,
            "recall": 0.875,
            "count_error": 1,
            "evaluation": "undercount",
            "image_hash": "abc",
            "image_source": "upload",
            "image_name": "cones.jpg",
            "model_key": "workflow:test/custom-workflow",
            "api_key": "SECRET",
            "annotated_image_bytes": b"not-stored",
        },
        path=path,
    )
    assert saved["benchmark_id"]
    assert "api_key" not in saved
    assert "annotated_image_bytes" not in saved
    loaded = load_benchmark_results(path)
    assert len(loaded) == 1
    filtered = filter_benchmark_history(loaded, undercount=True)
    assert len(filtered) == 1
    assert filter_benchmark_history(loaded, exact_match=True) == []
    assert filter_benchmark_history(loaded, inventory_key="Fence Panel") == []


def test_sanitize_no_secret_leakage():
    clean = sanitize_benchmark_record(
        {
            "inventory_key": "Fence Panel",
            "ROBOFLOW_API_KEY": "x",
            "authorization": "Bearer x",
            "technical": {"api_key": "x", "invocation_mode": "published_specification_with_prompt"},
            "final_count": 1,
        }
    )
    blob = json.dumps(clean)
    assert "Bearer" not in blob
    assert "ROBOFLOW_API_KEY" not in blob
    assert clean["technical"].get("invocation_mode") == "published_specification_with_prompt"


def test_sample_manifest_without_benchmark_metadata(tmp_path, monkeypatch):
    root = tmp_path / "assets" / "sample_images"
    root.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (32, 32), (10, 20, 30)).save(root / "a.jpg")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": "a",
                        "filename": "a.jpg",
                        "title": "A",
                        "inventory_type": "fence_panels",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    clear_sample_library_cache()
    monkeypatch.setattr(sample_mod, "SAMPLE_IMAGE_DIR", root)
    monkeypatch.setattr(sample_mod, "MANIFEST_PATH", root / "manifest.json")
    status = load_sample_library(force_reload=True)
    assert status.valid_count == 1
    assert status.samples[0].benchmark is None


def test_sample_manifest_with_benchmark_metadata(tmp_path, monkeypatch):
    root = tmp_path / "assets" / "sample_images"
    root.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (32, 32), (10, 20, 30)).save(root / "cone.jpg")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": "cone1",
                        "filename": "cone.jpg",
                        "title": "Cones",
                        "inventory_type": "traffic_cones",
                        "enabled": True,
                        "benchmark": {
                            "inventory_key": "traffic_cones",
                            "expected_count": 8,
                            "object_definition": "Count each individual visible traffic cone.",
                            "verified": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    clear_sample_library_cache()
    monkeypatch.setattr(sample_mod, "SAMPLE_IMAGE_DIR", root)
    monkeypatch.setattr(sample_mod, "MANIFEST_PATH", root / "manifest.json")
    status = load_sample_library(force_reload=True)
    s = status.samples[0]
    assert s.benchmark is not None
    assert s.benchmark["inventory_key"] == "Traffic Cones"
    assert s.benchmark["expected_count"] == 8
    assert s.benchmark["verified"] is True
    assert s.app_inventory_key == "Traffic Cones"


def test_parse_benchmark_metadata_helpers():
    assert parse_benchmark_metadata({}) is None
    assert parse_benchmark_metadata({"benchmark": "bad"}) is None
    meta = parse_benchmark_metadata(
        {"benchmark": {"inventory_key": "fence_panels", "expected_count": 1, "verified": True}}
    )
    assert meta["inventory_key"] == "Fence Panel"
    assert meta["expected_count"] == 1


def test_prompt_profile_update_with_backup(tmp_path: Path, monkeypatch):
    src = Path("inventory_profiles.json").read_text(encoding="utf-8")
    profiles_path = tmp_path / "inventory_profiles.json"
    profiles_path.write_text(src, encoding="utf-8")
    monkeypatch.setattr(bench, "PROFILES_PATH", profiles_path)
    monkeypatch.setattr(bench, "PROFILE_BACKUPS_DIR", tmp_path / "backups")
    clear_profiles_cache()
    ok, msg = update_profile_prompt_terms(
        "Traffic Cones",
        ["individual traffic cone", "orange traffic cone"],
        profiles_path=profiles_path,
    )
    assert ok, msg
    raw = json.loads(profiles_path.read_text(encoding="utf-8"))
    cone = next(p for p in raw["profiles"] if p["key"] == "Traffic Cones")
    assert cone["prompt_terms"] == ["individual traffic cone", "orange traffic cone"]
    assert (tmp_path / "backups").is_dir()
    assert list((tmp_path / "backups").glob("*.json"))
    assert profiles_path.with_suffix(".backup.json").exists()
    # Restore cache for other tests
    clear_profiles_cache()
    monkeypatch.undo()
    clear_profiles_cache()


def test_custom_item_cannot_promote_preset():
    ok, msg = update_profile_prompt_terms("Custom Item", ["widget"])
    assert not ok
    assert "Custom Item" in msg


def test_app_benchmark_isolated_from_wizard():
    src = inspect.getsource(app_module._run_benchmark_yolo_world)
    assert "analysis_results" in src
    assert "run_context" in src
    assert "uploaded_images" in src
    assert "Wizard session keys were restored" in src
    ui_src = inspect.getsource(app_module._render_ai_configuration_section)
    assert "Detection Benchmark" in ui_src
    assert "render_detection_benchmark_section" in ui_src


def test_benchmark_form_state_helpers():
    # Preset / custom prompt resolution used by the form
    from inventory_profiles import effective_prompts_for_inventory

    preset, errs = effective_prompts_for_inventory("Fence Panel")
    assert preset and not errs
    custom, cerrs = effective_prompts_for_inventory(
        "Custom Item",
        custom_item_name="bottle",
        custom_alternatives="glass bottle",
    )
    assert "bottle" in custom
    assert "glass bottle" in custom


def test_no_unmodified_fallback_flagged_in_runner():
    src = inspect.getsource(app_module._run_benchmark_yolo_world)
    assert "fallback" in src.lower()
    assert "not allowed for benchmark" in src


def test_project_fence_sample_has_verified_benchmark_metadata():
    clear_sample_library_cache()
    status = load_sample_library(force_reload=True)
    picket = next((s for s in status.samples if s.id == "fence_picket_panel_01"), None)
    assert picket is not None
    assert picket.benchmark is not None
    assert picket.benchmark.get("verified") is True
    assert picket.benchmark.get("expected_count") == 1
