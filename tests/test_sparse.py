import json

import numpy as np
import xarray as xr


def _scene() -> xr.Dataset:
    reflectance = xr.DataArray(
        np.ones((2, 2, 2), dtype=np.float32),
        dims=("y", "x", "band"),
        coords={"y": [0, 1], "x": [10, 11], "band": ["a", "b"]},
        name="reflectance",
    )
    return xr.Dataset(
        {
            "reflectance": reflectance,
            "solar_zenith": xr.DataArray(
                np.full((2, 2), 30.0, dtype=np.float32),
                dims=("y", "x"),
                coords={"y": reflectance.coords["y"], "x": reflectance.coords["x"]},
                name="solar_zenith",
            ),
            "valid_inversion_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": reflectance.coords["y"], "x": reflectance.coords["x"]},
                name="valid_inversion_mask",
            ),
        }
    )


def _background(scene: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        np.zeros(scene["reflectance"].shape, dtype=np.float32),
        dims=("y", "x", "band"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x", "band")},
        name="background_reflectance",
    )


def test_public_api_imports():
    import spires_io
    from spires_io import (
        SceneManifest,
        SpiresData,
        SpiresDataLoader,
        SpiresRunConfig,
        load_background_reflectance,
        prepare_scene_for_inversion,
    )

    assert spires_io.__version__
    assert SceneManifest is not None
    assert SpiresData is not None
    assert SpiresDataLoader is not None
    assert SpiresRunConfig is not None
    assert callable(load_background_reflectance)
    assert callable(prepare_scene_for_inversion)


def test_minimal_single_scene_config_parses(tmp_path):
    from spires_io.configs import SpiresConfig

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": {
                    "image_data": "scene.hdf",
                    "background_image": "background.nc",
                    "lut": "lut.mat",
                },
                "sensor": {"name": "modis"},
            }
        )
    )

    config = SpiresConfig(config_path)

    assert config.files.image_data == "scene.hdf"
    assert config.files.background_image == "background.nc"
    assert config.sensor.name == "modis"


def test_spires_data_wraps_minimal_scene():
    from spires_io import SpiresData

    data = SpiresData.from_scene(_scene())

    assert data.target_spectra.dims == ("y", "x", "band")
    assert data.solar_zenith.dims == ("y", "x")
    assert data.valid_mask.dims == ("y", "x")


def test_loader_from_config_loads_with_injected_readers(tmp_path):
    from spires_io import SpiresData, SpiresDataLoader

    scene = _scene()
    background = _background(scene)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "files": {
                    "image_data": "scene.hdf",
                    "background_image": "background.nc",
                    "lut": "lut.mat",
                },
                "sensor": {"name": "modis"},
            }
        )
    )

    loader = SpiresDataLoader.from_config(
        config_path,
        scene_preparer=lambda source, *, sensor, **kwargs: scene,
        background_loader=lambda path, *, target_scene: background,
        ancillary_loader=lambda sources, *, target_scene: None,
    )

    data = loader.load()

    assert isinstance(data, SpiresData)
    assert data.target_spectra.identical(scene["reflectance"])
    assert data.background_spectra.identical(background)
