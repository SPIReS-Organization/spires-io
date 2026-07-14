import json

import numpy as np
import pytest
import xarray as xr

from spires_io import SpiresData, SpiresDataLoader
from spires_io.configs import SceneManifestItem, SpiresRunConfig
from spires_io.loader import load_background_image


def _scene():
    reflectance = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        dims=("y", "x", "band"),
        coords={"y": [0, 1], "x": [10, 11], "band": ["a", "b"]},
        name="reflectance",
    )
    solar_zenith = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 11]},
        name="solar_zenith",
    )
    valid_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 11]},
        name="valid_inversion_mask",
    )
    sensor_zenith = xr.DataArray(
        np.full((2, 2), 20.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 11]},
        name="sensor_zenith",
    )
    sensor_azimuth = xr.DataArray(
        np.full((2, 2), 90.0, dtype=np.float32),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 11]},
        name="sensor_azimuth",
    )
    return xr.Dataset(
        {
            "reflectance": reflectance,
            "solar_zenith": solar_zenith,
            "valid_inversion_mask": valid_mask,
            "sensor_zenith": sensor_zenith,
            "sensor_azimuth": sensor_azimuth,
        }
    )


def _background(scene):
    return xr.DataArray(
        np.ones(scene["reflectance"].shape, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x", "band")},
    )


def _ancillary(scene):
    return xr.Dataset(
        {
            "canopy_fraction": xr.DataArray(
                np.full(scene["valid_inversion_mask"].shape, 0.5, dtype=np.float32),
                dims=("y", "x"),
                coords={dim: scene.coords[dim].values for dim in ("y", "x")},
            )
        }
    )


def _write_single_scene_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": {
                    "image_data": "scene.hdf",
                    "background_image": "background.nc",
                    "lut": "lut.mat",
                    "cloud_mask": "cloud.nc",
                },
                "sensor": {"name": "modis", "selected_bands": ["1", "2"]},
                "reader": {
                    "max_sensor_zenith": 50.0,
                    "max_solar_zenith": 70.0,
                },
            }
        )
    )
    return config_path


def test_loader_from_config_loads_single_scene_with_reader_settings(tmp_path):
    scene = _scene()
    background = _background(scene)
    calls = []

    def prepare(source, *, sensor, **kwargs):
        calls.append((source, sensor, kwargs))
        return scene

    def load_background(path, *, target_scene):
        calls.append(("background", path, target_scene is scene))
        return background

    loader = SpiresDataLoader.from_config(
        _write_single_scene_config(tmp_path),
        scene_preparer=prepare,
        background_loader=load_background,
    )

    data = loader.load()

    assert isinstance(data, SpiresData)
    assert data.background_spectra.identical(background)
    assert calls == [
        (
            "scene.hdf",
            "modis",
            {
                "lut_file": "lut.mat",
                "max_sensor_zenith": 50.0,
                "max_solar_zenith": 70.0,
                "bands": ["1", "2"],
                "cloud_mask_source": "cloud.nc",
            },
        ),
        ("background", "background.nc", True),
    ]


def test_loader_from_config_assigns_single_scene_ancillary(tmp_path):
    scene = _scene()
    background = _background(scene)
    ancillary = _ancillary(scene)
    calls = []
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": {
                    "image_data": "scene.hdf",
                    "background_image": "background.nc",
                    "lut": "lut.mat",
                    "canopy_fraction": "canopy.tif",
                },
                "sensor": {"name": "modis"},
            }
        )
    )

    def load_ancillary(sources, *, target_scene):
        calls.append((sources, target_scene is scene))
        return ancillary

    loader = SpiresDataLoader.from_config(
        config_path,
        scene_preparer=lambda source, *, sensor, **kwargs: scene,
        background_loader=lambda path, *, target_scene: background,
        ancillary_loader=load_ancillary,
    )

    data = loader.load()

    assert calls == [({"canopy_fraction": "canopy.tif"}, True)]
    assert data.ancillary.identical(ancillary)


def test_loader_from_config_assigns_viewable_canopy_fraction_when_enabled(tmp_path):
    scene = _scene()
    background = _background(scene)
    ancillary = _ancillary(scene)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": {
                    "image_data": "scene.hdf",
                    "background_image": "background.nc",
                    "lut": "lut.mat",
                    "canopy_fraction": "canopy.tif",
                },
                "sensor": {"name": "modis"},
                "canopy": {
                    "viewable_fraction": True,
                    "average_vertical_crown_radius": 2.0,
                    "average_horizontal_crown_radius": 1.0,
                },
            }
        )
    )

    loader = SpiresDataLoader.from_config(
        config_path,
        scene_preparer=lambda source, *, sensor, **kwargs: scene,
        background_loader=lambda path, *, target_scene: background,
        ancillary_loader=lambda sources, *, target_scene: ancillary,
    )

    data = loader.load()

    assert "viewable_canopy_fraction" in data.ancillary
    xr.testing.assert_identical(
        data.ancillary["canopy_fraction"],
        ancillary["canopy_fraction"],
    )


def test_spires_data_from_config_delegates_to_loader(monkeypatch):
    scene = _scene()
    expected = SpiresData.from_scene(scene)
    calls = []

    class FakeLoader:
        def load(self):
            calls.append("load")
            return expected

    def fake_from_config(cls, config_file):
        calls.append(config_file)
        return FakeLoader()

    monkeypatch.setattr(
        SpiresDataLoader,
        "from_config",
        classmethod(fake_from_config),
    )

    data = SpiresData.from_config("config.json")

    assert data is expected
    assert calls == ["config.json", "load"]


def test_loader_load_requires_single_scene_config():
    loader = SpiresDataLoader()

    with pytest.raises(ValueError, match="single-scene config"):
        loader.load()


def test_loader_load_item_uses_run_config_and_manifest_item():
    scene = _scene()
    background = _background(scene)
    calls = []

    def prepare(source, *, sensor, **kwargs):
        calls.append((source, sensor, kwargs))
        return scene

    def load_background(path, *, target_scene):
        calls.append(("background", path, target_scene is scene))
        return background

    run_config = SpiresRunConfig.from_mapping(
        {
            "sensor": {"name": "viirs", "selected_bands": ["I1", "M4"]},
            "reader": {
                "lut_file": "lut.mat",
                "keep_intermediate_reflectance": True,
            },
        }
    )
    item = SceneManifestItem(
        image_path="scene.h5",
        background_image="background.nc",
        output_path="out.tif",
    )

    loader = SpiresDataLoader(
        run_config,
        scene_preparer=prepare,
        background_loader=load_background,
    )
    data = loader.load_item(item)

    assert isinstance(data, SpiresData)
    assert data.background_spectra.identical(background)
    assert calls == [
        (
            "scene.h5",
            "viirs",
            {
                "lut_file": "lut.mat",
                "keep_intermediate_reflectance": True,
                "max_sensor_zenith": 65.0,
                "max_solar_zenith": 85.0,
                "bands": ["I1", "M4"],
            },
        ),
        ("background", "background.nc", True),
    ]


def test_loader_load_item_accepts_mapping_item():
    scene = _scene()
    background = _background(scene)
    calls = []

    def prepare(source, *, sensor, **kwargs):
        calls.append((source, sensor, kwargs))
        return scene

    loader = SpiresDataLoader(
        SpiresRunConfig.from_mapping({"sensor": "modis"}),
        scene_preparer=prepare,
        background_loader=lambda path, *, target_scene: background,
    )

    data = loader.load_item(
        {
            "image_path": "scene.hdf",
            "background_image": "background.nc",
        }
    )

    assert isinstance(data, SpiresData)
    assert calls == [
        (
            "scene.hdf",
            "modis",
            {
                "max_sensor_zenith": 65.0,
                "max_solar_zenith": 85.0,
            },
        )
    ]


def test_loader_load_item_requires_run_config():
    loader = SpiresDataLoader()

    with pytest.raises(ValueError, match="SpiresRunConfig"):
        loader.load_item({"image_path": "scene.hdf", "background_image": "background.nc"})


def test_load_background_image_reads_xarray_dataarray(tmp_path):
    background = _background(_scene())
    path = tmp_path / "background.nc"
    background.to_netcdf(path)

    loaded = load_background_image(str(path))

    expected = background.astype("float64")
    expected.name = "background_reflectance"
    xr.testing.assert_equal(loaded, expected)


def test_load_background_image_rejects_unknown_file_types():
    with pytest.raises(ValueError, match="background_image"):
        load_background_image("background.txt")


def test_loader_load_item_assigns_manifest_masks():
    scene = _scene()
    background = _background(scene)
    external_mask = xr.DataArray(
        [[False, True], [False, False]],
        dims=("y", "x"),
        coords={"y": scene.coords["y"].values, "x": scene.coords["x"].values},
        name="mask_manual",
    )
    calls = []

    def prepare(source, *, sensor, **kwargs):
        return scene

    def load_mask(path, *, target_scene, variable=None):
        calls.append((path, target_scene is scene, variable))
        return external_mask

    loader = SpiresDataLoader(
        SpiresRunConfig.from_mapping({"sensor": "modis"}),
        scene_preparer=prepare,
        background_loader=lambda path, *, target_scene: background,
        mask_loader=load_mask,
    )

    data = loader.load_item(
        {
            "image_path": "scene.hdf",
            "background_image": "background.nc",
            "masks": {"manual": {"path": "mask.nc", "variable": "mask_manual"}},
        }
    )

    assert calls == [("mask.nc", True, "mask_manual")]
    assert data.scene["mask_manual"].identical(external_mask)
    assert not bool(data.valid_mask.sel(y=0, x=11))


def test_loader_load_item_assigns_manifest_ancillary():
    scene = _scene()
    background = _background(scene)
    ancillary = _ancillary(scene)
    calls = []

    def load_ancillary(sources, *, target_scene):
        calls.append((sources, target_scene is scene))
        return ancillary

    loader = SpiresDataLoader(
        SpiresRunConfig.from_mapping({"sensor": "modis"}),
        scene_preparer=lambda source, *, sensor, **kwargs: scene,
        background_loader=lambda path, *, target_scene: background,
        ancillary_loader=load_ancillary,
    )

    data = loader.load_item(
        {
            "image_path": "scene.hdf",
            "background_image": "background.nc",
            "ancillary": {"dem": "dem.tif"},
        }
    )

    assert calls == [({"dem": "dem.tif"}, True)]
    assert data.ancillary.identical(ancillary)
