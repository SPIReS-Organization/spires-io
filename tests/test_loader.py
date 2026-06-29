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
    return xr.Dataset(
        {
            "reflectance": reflectance,
            "solar_zenith": solar_zenith,
            "valid_inversion_mask": valid_mask,
        }
    )


def _background(scene):
    return xr.DataArray(
        np.ones(scene["reflectance"].shape, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x", "band")},
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
                "options": {
                    "max_sensor_zenith": 50.0,
                    "max_solar_zenith": 70.0,
                },
            }
        )
    )
    return config_path


def test_loader_from_config_loads_single_scene_with_reader_policy(tmp_path):
    scene = _scene()
    background = _background(scene)
    calls = []

    def prepare(source, *, sensor, **kwargs):
        calls.append((source, sensor, kwargs))
        return scene

    def load_background(path):
        calls.append(("background", path))
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
        ("background", "background.nc"),
    ]


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

    def load_background(path):
        calls.append(("background", path))
        return background

    run_config = SpiresRunConfig.from_mapping(
        {
            "sensor": {"name": "viirs", "selected_bands": ["I1", "M4"]},
            "reader_options": {"lut_file": "lut.mat", "keep_intermediate_reflectance": True},
            "mask_policy": {"cloud_mask_policy": "ignore_cloud"},
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
                "cloud_mask_policy": "ignore_cloud",
                "bands": ["I1", "M4"],
            },
        ),
        ("background", "background.nc"),
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
        background_loader=lambda path: background,
    )

    data = loader.load_item(
        {
            "image_path": "scene.hdf",
            "background_image": "background.nc",
        }
    )

    assert isinstance(data, SpiresData)
    assert calls == [("scene.hdf", "modis", {})]


def test_loader_load_item_requires_run_config():
    loader = SpiresDataLoader()

    with pytest.raises(ValueError, match="SpiresRunConfig"):
        loader.load_item({"image_path": "scene.hdf", "background_image": "background.nc"})


def test_load_background_image_reads_xarray_dataarray(tmp_path):
    background = _background(_scene())
    path = tmp_path / "background.nc"
    background.to_netcdf(path)

    loaded = load_background_image(str(path))

    xr.testing.assert_equal(loaded, background)


def test_load_background_image_defers_non_xarray_files():
    with pytest.raises(NotImplementedError, match="deferred"):
        load_background_image("background.tif")
