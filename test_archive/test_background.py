import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from spires_contract.spectra import validate_background_spectra
from spires_io.background import load_background_reflectance


def _scene():
    reflectance = xr.DataArray(
        np.ones((2, 2, 2), dtype=np.float32),
        dims=("y", "x", "band"),
        coords={"y": [1.5, 0.5], "x": [10.5, 11.5], "band": ["red", "nir"]},
        name="reflectance",
    )
    return xr.Dataset(
        {
            "reflectance": reflectance,
            "solar_zenith": xr.DataArray(
                np.ones((2, 2), dtype=np.float32),
                dims=("y", "x"),
                coords={"y": reflectance.coords["y"], "x": reflectance.coords["x"]},
            ),
            "valid_inversion_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": reflectance.coords["y"], "x": reflectance.coords["x"]},
            ),
        }
    )


def test_load_background_reflectance_reads_spires_r0_xarray_product(tmp_path):
    scene = _scene()
    background = xr.DataArray(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        dims=("y", "x", "band"),
        coords={dim: scene.coords[dim].values for dim in ("y", "x", "band")},
        name="r0_reflectance",
    )
    path = tmp_path / "r0.nc"
    xr.Dataset({"r0_reflectance": background}).to_netcdf(path)

    loaded = load_background_reflectance(path, target_scene=scene)

    assert loaded.name == "background_reflectance"
    assert loaded.dtype == np.dtype("float32")
    validate_background_spectra(loaded)
    xr.testing.assert_equal(loaded, background.astype("float32").rename("background_reflectance"))


def test_load_background_reflectance_reads_multiband_geotiff(tmp_path):
    scene = _scene()
    path = tmp_path / "background.tif"
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="float32",
        crs="EPSG:32613",
        transform=from_origin(10, 2, 1, 1),
    ) as dataset:
        dataset.write(values)

    loaded = load_background_reflectance(path, target_scene=scene)

    assert loaded.dims == ("y", "x", "band")
    assert loaded.coords["band"].values.tolist() == ["red", "nir"]
    assert loaded.dtype == np.dtype("float32")
    np.testing.assert_array_equal(loaded.values, values.transpose(1, 2, 0))
    validate_background_spectra(loaded)


def test_load_background_reflectance_rejects_band_count_mismatch(tmp_path):
    scene = _scene()
    background = xr.DataArray(
        np.ones((2, 2, 1), dtype=np.float32),
        dims=("y", "x", "band"),
        coords={"y": scene.coords["y"], "x": scene.coords["x"], "band": ["red"]},
    )
    path = tmp_path / "background.nc"
    background.to_netcdf(path)

    with pytest.raises(ValueError, match="band count"):
        load_background_reflectance(path, target_scene=scene)
