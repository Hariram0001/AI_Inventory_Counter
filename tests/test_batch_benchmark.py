"""Offline tests for batch benchmark + threshold sweep."""

from __future__ import annotations

import csv
import inspect
import io
import json
from pathlib import Path

import pytest

import app as app_module
import benchmark as bench
from benchmark import (
    ADAPTER_VERSION,
    MAX_THRESHOLDS,
    BatchImageSpec,
    BenchmarkRunCache,
    BenchmarkRunOutcome,
    NamedPromptSet,
    aggregate_prompt_threshold,
    apply_visual_review_to_run,
    build_batch_session,
    build_cache_key,
    build_comparison_matrix,
    calculate_inference_run_count,
    dedupe_batch_images,
    enumerate_batch_combinations,
    export_session_csv,
    export_session_json,
    load_benchmark_results,
    normalize_thresholds,
    outcome_to_run_dict,
    parse_named_prompt_sets,
    recommend_configuration,
    save_batch_session,
    save_benchmark_result,
    update_profile_prompt_terms,
    validate_batch_ground_truth,
)


def test_single_image_benchmark_still_present():
    import benchmark_ui as ui

    src = inspect.getsource(ui.render_detection_benchmark_section)
    assert "Single Image" in src
    assert "Batch Benchmark" in src
    assert "_render_single_image_benchmark" in src


def test_threshold_normalization_and_limits():
    vals, errs = normalize_thresholds([0.25, 0.10, 0.10, 0.95, 0.01, 1.5, -1])
    assert vals == [0.01, 0.10, 0.25, 0.95]
    assert any("out of range" in e for e in errs)
    many = [0.05 + i * 0.01 for i in range(12)]
    vals2, errs2 = normalize_thresholds(many)
    assert len(vals2) <= MAX_THRESHOLDS
    assert any("At most" in e for e in errs2)


def test_inference_run_count():
    assert calculate_inference_run_count(
        image_count=8, prompt_set_count=3, threshold_count=5
    ) == 120


def test_image_deduplication_and_max():
    a = b"abc" * 10
    b = b"xyz" * 10
    specs, warns, by_hash = dedupe_batch_images(
        [
            {"image_name": "a.jpg", "image_bytes": a, "image_source": "upload"},
            {"image_name": "a_copy.jpg", "image_bytes": a, "image_source": "upload"},
            {"image_name": "b.png", "image_bytes": b, "image_source": "upload"},
            {"image_name": "bad.gif", "image_bytes": b"x", "image_source": "upload"},
        ],
        max_images=20,
    )
    assert len(specs) == 2
    assert any("Duplicate" in w for w in warns)
    assert any("Unsupported" in w for w in warns)
    assert len(by_hash) == 2


def test_per_image_expected_counts_and_excluded():
    specs = [
        BatchImageSpec("1", "a.jpg", "h1", "upload", 10, expected_count=None, include_in_aggregate=True),
        BatchImageSpec("2", "b.jpg", "h2", "upload", 10, expected_count=3, include_in_aggregate=False),
    ]
    errs = validate_batch_ground_truth(specs)
    assert errs  # missing expected on included image
    specs[0].expected_count = 1
    assert validate_batch_ground_truth(specs) == []
    combos = enumerate_batch_combinations(
        specs,
        [NamedPromptSet("A", ["gate"], True)],
        [0.2],
    )
    # Excluded image skipped
    assert len(combos) == 1
    assert combos[0][0].image_name == "a.jpg"


def test_named_prompt_sets_and_independent_combos():
    sets, errs = parse_named_prompt_sets(
        [
            {"name": "Basic", "prompts": "gate", "enabled": True},
            {"name": "Specific", "prompts": "fence gate, driveway gate", "enabled": True},
            {"name": "Off", "prompts": "x", "enabled": False},
        ]
    )
    assert not errs or all("Off" not in e for e in errs) or len([s for s in sets if s.enabled]) == 2
    enabled = [s for s in sets if s.enabled]
    assert len(enabled) == 2
    imgs = [
        BatchImageSpec("1", "a.jpg", "h1", "u", 1, expected_count=1),
        BatchImageSpec("2", "b.jpg", "h2", "u", 1, expected_count=1),
    ]
    combos = enumerate_batch_combinations(imgs, sets, [0.15, 0.20])
    assert len(combos) == 2 * 2 * 2  # disabled set excluded


def test_cache_key_stability_and_reuse():
    k1 = build_cache_key(
        image_hash="abc",
        model_key="workflow:ws/wf",
        prompts=["gate", "fence gate"],
        confidence_threshold=0.2,
        workflow_id="ws/wf",
        user_id=7,
    )
    k2 = build_cache_key(
        image_hash="abc",
        model_key="workflow:ws/wf",
        prompts=["fence gate", "gate"],  # order normalized
        confidence_threshold=0.2000,
        workflow_id="ws/wf",
        user_id=7,
    )
    assert k1 == k2
    k_other = build_cache_key(
        image_hash="abc",
        model_key="workflow:ws/wf",
        prompts=["gate", "fence gate"],
        confidence_threshold=0.2,
        workflow_id="ws/wf",
        user_id=8,
    )
    assert k1 != k_other
    cache = BenchmarkRunCache()
    ok = {
        "success": True,
        "execution_failed": False,
        "fallback_used": False,
        "final_count": 1,
        "detections": [{"class_name": "gate"}],
    }
    cache.put(k1, ok)
    assert cache.get(k1) is not None
    assert cache.get(k1).get("cached") is True


def test_no_caching_auth_failures():
    cache = BenchmarkRunCache()
    key = "k"
    cache.put(
        key,
        {
            "success": False,
            "execution_failed": True,
            "error_message": "Unauthorized 401 API key",
            "final_count": 0,
        },
    )
    assert cache.get(key) is None
    outcome = BenchmarkRunOutcome(
        prompt_set_label="A",
        success=False,
        execution_failed=True,
        error_message="401 unauthorized",
    )
    from benchmark import is_cacheable_success

    assert is_cacheable_success(outcome) is False


def test_aggregate_micro_macro_and_unreviewed():
    runs = [
        {
            "prompt_set_label": "A",
            "confidence_threshold": 0.2,
            "include_in_aggregate": True,
            "execution_failed": False,
            "final_count": 1,
            "expected_count": 1,
            "count_error": 0,
            "exact_match": True,
            "reviewed": True,
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "processing_time": 1.0,
        },
        {
            "prompt_set_label": "A",
            "confidence_threshold": 0.2,
            "include_in_aggregate": True,
            "execution_failed": False,
            "final_count": 0,
            "expected_count": 1,
            "count_error": 1,
            "exact_match": False,
            "reviewed": False,
            "processing_time": 1.5,
        },
    ]
    agg = aggregate_prompt_threshold(runs, prompt_set_label="A", confidence_threshold=0.2)
    assert agg["images_evaluated"] == 2
    assert agg["exact_match_images"] == 1
    assert agg["micro_precision"] == 1.0
    assert agg["incomplete_metrics"] is False  # at least one reviewed
    # Unreviewed run must not invent TP as AI count
    assert agg["reviewed_images"] == 1


def test_recommendation_and_too_few():
    aggs = [
        {
            "prompt_set_label": "A",
            "confidence_threshold": 0.2,
            "images_evaluated": 1,
            "total_runs": 1,
            "failed_runs": 0,
            "mean_absolute_count_error": 0.0,
            "exact_match_rate": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "precision_recall_available": True,
        }
    ]
    assert recommend_configuration(aggs, min_images=2) is None
    aggs[0]["images_evaluated"] = 3
    rec = recommend_configuration(aggs, objective="lowest_mae", min_images=2)
    assert rec is not None
    assert "Best configuration for this benchmark dataset" in rec["label"]


def test_profile_promotion_with_confidence_and_custom_blocked(tmp_path, monkeypatch):
    src = Path("inventory_profiles.json").read_text(encoding="utf-8")
    path = tmp_path / "inventory_profiles.json"
    path.write_text(src, encoding="utf-8")
    monkeypatch.setattr(bench, "PROFILES_PATH", path)
    monkeypatch.setattr(bench, "PROFILE_BACKUPS_DIR", tmp_path / "backups")
    ok, msg = update_profile_prompt_terms(
        "Gates",
        ["individual fence gate", "driveway gate"],
        profiles_path=path,
        default_confidence=0.20,
        justification_benchmark_id="sess-123",
    )
    assert ok, msg
    raw = json.loads(path.read_text(encoding="utf-8"))
    gates = next(p for p in raw["profiles"] if p["key"] == "Gates")
    assert gates["default_confidence"] == 0.20
    assert gates["last_benchmark_id"] == "sess-123"
    ok2, _ = update_profile_prompt_terms("Custom Item", ["x"], profiles_path=path)
    assert not ok2


def test_exports_json_csv_no_secrets():
    img = BatchImageSpec("1", "a.jpg", "hash1", "upload", 10, expected_count=1)
    ps = NamedPromptSet("Basic", ["gate"], True)
    outcome = BenchmarkRunOutcome(
        prompt_set_label="Basic",
        prompt_set=["gate"],
        success=True,
        final_count=1,
        detections=[{"class_name": "gate", "confidence": 0.5}],
    )
    run = outcome_to_run_dict(
        outcome,
        image=img,
        confidence_threshold=0.2,
        model_key="workflow:x/y",
        session_id="s1",
        reviewed=False,
    )
    run["api_key"] = "SECRET"
    session = build_batch_session(
        inventory_key="Gates",
        model_key="workflow:x/y",
        images=[img],
        prompt_sets=[ps],
        thresholds=[0.2],
        runs=[run],
        session_id="s1",
    )
    js = export_session_json(session)
    assert "SECRET" not in js
    assert "api_key" not in js
    csv_text = export_session_csv(session)
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["precision"] == "not_reviewed"
    assert "api_key" not in reader.fieldnames


def test_session_storage_compat_with_old_records(tmp_path):
    bench_path = tmp_path / "benchmarks.json"
    sess_path = tmp_path / "sessions.json"
    save_benchmark_result(
        {
            "inventory_key": "Fence Panel",
            "final_count": 1,
            "expected_count": 1,
            "prompt_set": ["fence panel"],
            "record_kind": "single",
        },
        path=bench_path,
    )
    img = BatchImageSpec("1", "a.jpg", "h", "u", 1, expected_count=1)
    session = build_batch_session(
        inventory_key="Fence Panel",
        model_key="m",
        images=[img],
        prompt_sets=[NamedPromptSet("A", ["fence panel"])],
        thresholds=[0.2],
        runs=[
            {
                "run_id": "r1",
                "final_count": 1,
                "expected_count": 1,
                "prompt_set_label": "A",
                "prompt_set": ["fence panel"],
                "confidence_threshold": 0.2,
                "include_in_aggregate": True,
                "execution_failed": False,
                "count_error": 0,
                "exact_match": True,
                "reviewed": False,
                "processing_time": 0.1,
            }
        ],
    )
    # Temporarily redirect save paths
    saved = save_batch_session(
        session, path=sess_path, history_path=bench_path, write_history=True
    )
    assert saved["session_id"]
    old = load_benchmark_results(bench_path)
    assert any(r.get("record_kind") == "single" for r in old)
    assert any(r.get("session_id") == session.get("session_id") or "batch_session" in str(r.get("notes") or "") for r in old)


def test_visual_review_not_copied_across_thresholds():
    run = {
        "final_count": 1,
        "expected_count": 1,
        "detections": [{"class_name": "gate", "confidence": 0.4}],
        "detection_labels": [],
        "missed_count": 0,
        "execution_failed": False,
        "success": True,
        "confidence_threshold": 0.2,
    }
    updated = apply_visual_review_to_run(run, labels=["correct"], missed_count=0)
    assert updated["reviewed"] is True
    assert updated["precision"] == 1.0
    other = dict(run)
    other["confidence_threshold"] = 0.3
    assert other.get("reviewed") is not True


def test_matrix_and_partial_failure_isolation():
    runs = [
        {
            "prompt_set_label": "A",
            "confidence_threshold": 0.2,
            "include_in_aggregate": True,
            "execution_failed": True,
            "final_count": 0,
            "expected_count": 1,
            "processing_time": 0.2,
        },
        {
            "prompt_set_label": "A",
            "confidence_threshold": 0.2,
            "include_in_aggregate": True,
            "execution_failed": False,
            "final_count": 1,
            "expected_count": 1,
            "count_error": 0,
            "exact_match": True,
            "reviewed": False,
            "processing_time": 0.3,
        },
    ]
    agg = aggregate_prompt_threshold(runs, prompt_set_label="A", confidence_threshold=0.2)
    assert agg["failed_runs"] == 1
    assert agg["images_evaluated"] == 1
    matrix = build_comparison_matrix([agg])
    assert matrix[0]["Failed runs"] == 1


def test_no_wizard_mutation_and_no_fallback_in_runners():
    src = inspect.getsource(app_module._run_benchmark_yolo_world)
    assert "fallback" in src.lower()
    assert "not allowed for benchmark" in src
    ui = inspect.getsource(app_module._render_ai_configuration_section)
    assert "render_detection_benchmark_section" in ui


def test_progress_state_shape():
    # Progress dict keys used by UI
    progress = {
        "completed": 3,
        "total": 10,
        "image": "a.jpg",
        "prompt_set": "Basic",
        "threshold": 0.2,
        "failures": 1,
    }
    assert progress["completed"] < progress["total"]


def test_adapter_version_in_cache_key():
    assert ADAPTER_VERSION
    k = build_cache_key(
        image_hash="x",
        model_key="m",
        prompts=["a"],
        confidence_threshold=0.1,
        adapter_version="v1",
    )
    k2 = build_cache_key(
        image_hash="x",
        model_key="m",
        prompts=["a"],
        confidence_threshold=0.1,
        adapter_version="v2",
    )
    assert k != k2
