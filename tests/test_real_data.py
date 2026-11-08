# tests/test_real_data.py
"""Opt-in: set SPIRES_SCENE_CACHE=/scratch/spires/scenes with one scene per sensor sub-directory."""
import os
from pathlib import Path

import pytest
from spires_contract.spectra import validate_solar_angles, validate_target_spectra

from spires_io.api import open_surface_reflectance

CACHE = os.environ.get("SPIRES_SCENE_CACHE")


def _first_scene(sensor: str) -> Path:
    if CACHE is None:
        pytest.skip("SPIRES_SCENE_CACHE not set")
    scenes = sorted(p for p in (Path(CACHE) / sensor).glob("*") if p.is_dir() and not p.name.endswith(".partial"))
    if not scenes:
        pytest.skip(f"no scenes under {CACHE}/{sensor}")
    return scenes[0]


@pytest.mark.real_data
@pytest.mark.parametrize("sensor", ["landsat", "hls_l30", "hls_s30", "sentinel2"])
def test_real_scene_passes_contract(sensor):
    ds = open_surface_reflectance(_first_scene(sensor), sensor=sensor)
    validate_target_spectra(ds["reflectance"])
    validate_solar_angles(ds["solar_zenith"])
    assert 0.05 < float(ds["valid"].mean()) < 1.0


def test_first_scene_ignores_partial_directories(tmp_path, monkeypatch):
    import test_real_data as mod

    monkeypatch.setattr(mod, "CACHE", str(tmp_path))
    (tmp_path / "landsat" / "A.partial").mkdir(parents=True)
    (tmp_path / "landsat" / "B").mkdir()
    assert mod._first_scene("landsat").name == "B"
