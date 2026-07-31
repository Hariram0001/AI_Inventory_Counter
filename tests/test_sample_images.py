"""Offline tests for the built-in sample-image library."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
from pathlib import Path

import pytest
from PIL import Image

import app as app_module
import sample_images as sample_mod
from config import PROJECT_ROOT, SAMPLE_IMAGE_DIR
from inventory_config import SELECTABLE_INVENTORY_KEY
from sample_images import (
    clear_sample_library_cache,
    get_sample_by_id,
    list_enabled_samples,
    load_sample_library,
    read_sample_bytes,
)


def _write_png(path: Path, color=(40, 80, 120)) -> bytes:
    img = Image.new("RGB", (64, 48), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    path.write_bytes(data)
    return data


@pytest.fixture()
def sample_tmpdir(tmp_path, monkeypatch):
    root = tmp_path / "assets" / "sample_images"
    root.mkdir(parents=True)
    clear_sample_library_cache()
    monkeypatch.setattr(sample_mod, "SAMPLE_IMAGE_DIR", root)
    monkeypatch.setattr(sample_mod, "MANIFEST_PATH", root / "manifest.json")
    yield root
    clear_sample_library_cache()


def test_manifest_loads_empty(sample_tmpdir):
    (sample_tmpdir / "manifest.json").write_text('{"images": []}', encoding="utf-8")
    status = load_sample_library(force_reload=True)
    assert status.manifest_valid
    assert status.valid_count == 0
    assert status.enabled_count == 0


def test_missing_manifest_handled(sample_tmpdir):
    status = load_sample_library(force_reload=True)
    assert not status.manifest_exists
    assert status.warnings


def test_invalid_json_handled(sample_tmpdir):
    (sample_tmpdir / "manifest.json").write_text("{not-json", encoding="utf-8")
    status = load_sample_library(force_reload=True)
    assert not status.manifest_valid
    assert status.manifest_error


def test_missing_image_reported(sample_tmpdir):
    manifest = {
        "images": [
            {
                "id": "missing_one",
                "filename": "nope.jpg",
                "title": "Missing",
                "inventory_type": "fence_panels",
                "enabled": True,
            }
        ]
    }
    (sample_tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    status = load_sample_library(force_reload=True)
    assert "nope.jpg" in status.missing_files
    assert status.valid_count == 0


def test_duplicate_sample_id_reported(sample_tmpdir):
    _write_png(sample_tmpdir / "a.png")
    _write_png(sample_tmpdir / "b.png", color=(1, 2, 3))
    manifest = {
        "images": [
            {
                "id": "dup",
                "filename": "a.png",
                "title": "A",
                "inventory_type": "fence_panels",
                "enabled": True,
            },
            {
                "id": "dup",
                "filename": "b.png",
                "title": "B",
                "inventory_type": "fence_panels",
                "enabled": True,
            },
        ]
    }
    (sample_tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    status = load_sample_library(force_reload=True)
    assert "dup" in status.duplicate_ids
    assert status.valid_count == 1


def test_unsupported_file_rejected(sample_tmpdir):
    (sample_tmpdir / "x.gif").write_bytes(b"GIF89a")
    manifest = {
        "images": [
            {
                "id": "gif",
                "filename": "x.gif",
                "title": "Gif",
                "inventory_type": "fence_panels",
                "enabled": True,
            }
        ]
    }
    (sample_tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    status = load_sample_library(force_reload=True)
    assert "x.gif" in status.invalid_files


def test_valid_and_disabled_samples(sample_tmpdir):
    _write_png(sample_tmpdir / "on.png")
    _write_png(sample_tmpdir / "off.png", color=(9, 9, 9))
    manifest = {
        "images": [
            {
                "id": "on",
                "filename": "on.png",
                "title": "On",
                "description": "Enabled fence",
                "inventory_type": "fence_panels",
                "enabled": True,
                "featured": True,
            },
            {
                "id": "off",
                "filename": "off.png",
                "title": "Off",
                "inventory_type": "fence_panels",
                "enabled": False,
            },
            {
                "id": "other",
                "filename": "on.png",
                "title": "Other inv",
                "inventory_type": "poles",
                "enabled": True,
            },
        ]
    }
    # other reuses on.png — second id with same file is ok; inventory filter hides it
    (sample_tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Fix: other needs its own file to be valid — rewrite
    _write_png(sample_tmpdir / "other.png", color=(50, 50, 50))
    manifest["images"][2]["filename"] = "other.png"
    (sample_tmpdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    clear_sample_library_cache()
    enabled = list_enabled_samples(inventory_key=SELECTABLE_INVENTORY_KEY)
    ids = {s.id for s in enabled}
    assert "on" in ids
    assert "off" not in ids
    assert "other" not in ids
    sample = get_sample_by_id("on")
    assert sample is not None
    data = read_sample_bytes(sample)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_relative_paths_no_windows_absolute_required():
    assert SAMPLE_IMAGE_DIR == PROJECT_ROOT / "assets" / "sample_images"
    # Must resolve from project/module location, not a hard-coded user home path in code
    src = Path(sample_mod.__file__).read_text(encoding="utf-8")
    assert "C:\\\\Users\\\\hello" not in src
    assert "PROJECT_ROOT" in src
    assert SAMPLE_IMAGE_DIR.parts[-2:] == ("assets", "sample_images")


def test_photos_stage_has_sample_tab():
    src = inspect.getsource(app_module.stage_photos)
    assert "Sample Images" in src
    assert "_render_sample_images_tab" in src


def test_sample_uses_shared_pipeline():
    tab_src = inspect.getsource(app_module._render_sample_images_tab)
    helper_src = inspect.getsource(app_module._add_sample_by_id)
    assert "_add_sample_by_id" in tab_src
    assert '_add_image_bytes(' in helper_src
    assert 'source="sample"' in helper_src
    assert "sample_id" in helper_src
    assert "sample_preview_id" not in tab_src or "pop(\"sample_preview_id\"" in tab_src
    assert "Preview" not in tab_src
    add_src = inspect.getsource(app_module._add_image_bytes)
    assert "This image is already included." in add_src
    assert "content_hash" in inspect.getsource(app_module._image_meta)


def test_duplicate_content_prevented(sample_tmpdir):
    data = _write_png(sample_tmpdir / "dup.png")
    meta1 = app_module._image_meta("dup.png", data, source="sample", sample_id="dup")
    meta2 = app_module._image_meta("other_name.png", data, source="upload")
    assert meta1["id"] == meta2["id"]
    assert meta1["content_hash"] == hashlib.sha256(data).hexdigest()


def test_settings_sample_library_section():
    src = inspect.getsource(app_module._render_sample_library_settings)
    assert "Built-in sample library" in src or "Built-in Sample Library" in src
    assert "Valid" in src
    assert "Duplicate IDs" in src


def test_project_manifest_exists_and_is_valid():
    clear_sample_library_cache()
    status = load_sample_library(force_reload=True)
    # Real project folder should exist with a valid (possibly empty) manifest
    assert (PROJECT_ROOT / "assets" / "sample_images").is_dir()
    assert status.directory_exists
    assert status.manifest_exists
    assert status.manifest_valid
    # No fake samples registered
    for s in status.samples:
        assert s.path is not None and s.path.is_file()


def test_gitignore_allows_sample_assets():
    text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!assets/sample_images/**" in text


def test_old_history_notes_without_comparison_still_ok():
    # Saving without comparison_mode must remain a valid JSON append pattern
    src = inspect.getsource(app_module._save_inventory)
    assert "AIC_META=" in src
    assert "comparison_mode" in src
