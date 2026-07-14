import json

import pytest

from spires_io import SceneManifest, SpiresRunConfig
from spires_io.configs import SceneManifestItem


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_spires_run_config_parses_global_settings(tmp_path):
    config_path = _write_json(
        tmp_path,
        "run_config.json",
        {
            "sensor": {"name": "modis", "selected_bands": ["1", "2"]},
            "reader": {"lut_file": "lut.mat"},
            "inversion": {"max_eval": 50},
            "clustering": {
                "enabled": True,
                "features": ["reflectance", "solar_zenith"],
                "solar_zenith_tol": 3.0,
            },
            "spatial": {"resampling_method": "nearest"},
            "canopy": {"viewable_fraction": True},
            "output_policy": {"root": "outputs"},
            "ancillary_paths": {"dem": "dem.tif"},
            "description": "example batch",
        },
    )

    config = SpiresRunConfig.from_file(config_path)

    assert config.sensor.name == "modis"
    assert config.sensor.selected_bands == ["1", "2"]
    assert list(config.sensor.band_names) == ["1", "2"]
    assert config.reader.extra == {"lut_file": "lut.mat"}
    assert config.inversion.max_eval == 50
    assert config.clustering.enabled
    assert config.clustering.features == ("reflectance", "solar_zenith")
    assert config.clustering.solar_zenith_tol == 3.0
    assert config.spatial.resampling_method == "nearest"
    assert config.canopy.viewable_fraction
    assert config.output_policy == {"root": "outputs"}
    assert config.ancillary_paths == {"dem": "dem.tif"}
    assert config.extra == {"description": "example batch"}


def test_spires_run_config_accepts_sensor_string_and_output_aliases(tmp_path):
    config_path = _write_json(
        tmp_path,
        "run_config.json",
        {
            "sensor": "viirs",
            "output": {"root": "outputs"},
            "ancillary": {"static_root": "ancillary"},
        },
    )

    config = SpiresRunConfig.from_file(config_path)

    assert config.sensor.name == "viirs"
    assert config.output_policy == {"root": "outputs"}
    assert config.ancillary_paths == {"static_root": "ancillary"}


def test_spires_run_config_rejects_missing_sensor(tmp_path):
    config_path = _write_json(tmp_path, "run_config.json", {"reader": {}})

    with pytest.raises(ValueError, match="missing required key 'sensor'"):
        SpiresRunConfig.from_file(config_path)


def test_spires_run_config_requires_json_file(tmp_path):
    config_path = _write_json(tmp_path, "run_config.txt", {"sensor": "modis"})

    with pytest.raises(ValueError, match="JSON files only"):
        SpiresRunConfig.from_file(config_path)


def test_spires_run_config_rejects_non_object_sections(tmp_path):
    config_path = _write_json(
        tmp_path,
        "run_config.json",
        {"sensor": "modis", "reader": ["bad"]},
    )

    with pytest.raises(ValueError, match="reader"):
        SpiresRunConfig.from_file(config_path)


def test_spires_run_config_rejects_legacy_sections(tmp_path):
    config_path = _write_json(
        tmp_path,
        "run_config.json",
        {"sensor": "modis", "reader_options": {"lut_file": "lut.mat"}},
    )

    with pytest.raises(ValueError, match="reader_options -> reader"):
        SpiresRunConfig.from_file(config_path)


def test_spires_run_config_uses_explicit_cluster_defaults(tmp_path):
    config_path = _write_json(tmp_path, "run_config.json", {"sensor": "modis"})

    config = SpiresRunConfig.from_file(config_path)

    assert config.clustering.features == ("reflectance", "background", "solar_zenith")
    assert config.clustering.reflectance_tol == 0.02
    assert config.clustering.background_tol == 0.02
    assert config.clustering.solar_zenith_tol == 2.0


def test_scene_manifest_parses_scene_items(tmp_path):
    manifest_path = _write_json(
        tmp_path,
        "manifest.json",
        {
            "scenes": [
                {
                    "image_path": "scene_001.hdf",
                    "background_image": "background_001.tif",
                    "output_path": "scene_001_spires.tif",
                    "tile": "h09v05",
                    "water_year": 2026,
                    "date": "2026-04-01",
                    "masks": {"cloud": "cloud_001.tif"},
                    "ancillary": {"dem": "dem_001.tif"},
                    "custom_id": "scene-001",
                }
            ],
            "batch_name": "april",
        },
    )

    manifest = SceneManifest.from_file(manifest_path)

    assert len(manifest.scenes) == 1
    assert manifest.extra == {"batch_name": "april"}

    item = manifest.scenes[0]
    assert isinstance(item, SceneManifestItem)
    assert item.image_path == "scene_001.hdf"
    assert item.background_image == "background_001.tif"
    assert item.output_path == "scene_001_spires.tif"
    assert item.tile == "h09v05"
    assert item.water_year == 2026
    assert item.date == "2026-04-01"
    assert item.masks == {"cloud": "cloud_001.tif"}
    assert item.ancillary == {"dem": "dem_001.tif"}
    assert item.extra == {"custom_id": "scene-001"}


def test_scene_manifest_requires_scenes_key(tmp_path):
    manifest_path = _write_json(tmp_path, "manifest.json", {"batch_name": "empty"})

    with pytest.raises(ValueError, match="missing required key 'scenes'"):
        SceneManifest.from_file(manifest_path)


def test_scene_manifest_requires_scenes_list(tmp_path):
    manifest_path = _write_json(tmp_path, "manifest.json", {"scenes": {}})

    with pytest.raises(ValueError, match="must be a list"):
        SceneManifest.from_file(manifest_path)


def test_scene_manifest_rejects_non_object_scene_items(tmp_path):
    manifest_path = _write_json(tmp_path, "manifest.json", {"scenes": ["bad"]})

    with pytest.raises(ValueError, match="scene 0 must be an object"):
        SceneManifest.from_file(manifest_path)


def test_scene_manifest_requires_scene_paths(tmp_path):
    manifest_path = _write_json(
        tmp_path,
        "manifest.json",
        {"scenes": [{"image_path": "scene.hdf"}]},
    )

    with pytest.raises(ValueError, match="background_image"):
        SceneManifest.from_file(manifest_path)


def test_scene_manifest_requires_json_file(tmp_path):
    manifest_path = _write_json(
        tmp_path,
        "manifest.txt",
        {"scenes": [{"image_path": "scene.hdf", "background_image": "background.tif"}]},
    )

    with pytest.raises(ValueError, match="JSON files only"):
        SceneManifest.from_file(manifest_path)
