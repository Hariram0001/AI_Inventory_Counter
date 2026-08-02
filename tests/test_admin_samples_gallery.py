"""Admin Console samples must appear on Add Photos filtered by inventory type."""

from __future__ import annotations

import io

import pytest
from PIL import Image

import admin_samples
import config
import sample_images as sample_mod
from sample_images import (
    clear_sample_library_cache,
    get_sample_by_id,
    list_enabled_samples,
    read_sample_bytes,
)


def _png_bytes(color=(220, 90, 40)) -> bytes:
    img = Image.new("RGB", (96, 72), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def gallery_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    samples_root = tmp_path / "assets" / "sample_images"
    samples_root.mkdir(parents=True)
    (samples_root / "manifest.json").write_text('{"images": []}', encoding="utf-8")
    db = str(tmp_path / "gallery.db")

    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(sample_mod, "SAMPLE_IMAGE_DIR", samples_root)
    monkeypatch.setattr(sample_mod, "MANIFEST_PATH", samples_root / "manifest.json")
    clear_sample_library_cache()
    yield db
    clear_sample_library_cache()


def test_admin_sample_shows_for_matching_inventory(gallery_env):
    db = gallery_env
    sample = admin_samples.add_sample(
        data=_png_bytes(),
        title="Road Cones",
        inventory_type="Traffic Cones",
        description="admin upload",
        expected_count=12,
        uploaded_by="admin",
        is_enabled=True,
        db_path=db,
    )
    cones = list_enabled_samples(inventory_key="Traffic Cones", db_path=db)
    assert any(s.id == sample.sample_id for s in cones)
    hit = next(s for s in cones if s.id == sample.sample_id)
    assert hit.source == "admin_sample"
    assert hit.app_inventory_key == "Traffic Cones"
    assert hit.title == "Road Cones"

    fence = list_enabled_samples(inventory_key="Fence Panel", db_path=db)
    assert all(s.id != sample.sample_id for s in fence)


def test_disabled_admin_sample_hidden(gallery_env):
    db = gallery_env
    sample = admin_samples.add_sample(
        data=_png_bytes((10, 20, 30)),
        title="Hidden Cones",
        inventory_type="Traffic Cones",
        is_enabled=False,
        db_path=db,
    )
    cones = list_enabled_samples(inventory_key="Traffic Cones", db_path=db)
    assert all(s.id != sample.sample_id for s in cones)


def test_add_photos_can_resolve_and_read_admin_sample(gallery_env):
    db = gallery_env
    sample = admin_samples.add_sample(
        data=_png_bytes((11, 22, 33)),
        title="Readable Cones",
        inventory_type="Traffic Cones",
        db_path=db,
    )
    found = get_sample_by_id(sample.sample_id, db_path=db)
    assert found is not None
    assert found.source == "admin_sample"
    data = read_sample_bytes(found)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
