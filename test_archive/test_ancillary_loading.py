import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr

from spires_io.ancillary import load_ancillary_layers


def _scene():
    coords = {"y": [1.5, 0.5], "x": [10.5, 11.5]}
    return xr.Dataset(
        {
            "reflectance": xr.DataArray(
                np.ones((2, 2, 1), dtype=np.float32),
                dims=("y", "x", "band"),
                coords={**coords, "band": ["red"]},
            ),
            "solar_zenith": xr.DataArray(
                np.ones((2, 2), dtype=np.float32),
                dims=("y", "x"),
                coords=coords,
            ),
            "valid_inversion_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords=coords,
            ),
        }
    )


def test_load_ancillary_layers_reads_and_cleans_xarray_layers(tmp_path):
    scene = _scene()
    canopy = xr.DataArray(
        [[50.0, np.nan], [125.0, 10.0]],
        dims=("y", "x"),
        coords={"y": scene.coords["y"].values, "x": scene.coords["x"].values},
        name="canopy_fraction",
    )
    dem = xr.DataArray(
        [[1000.0, -2000.0], [9000.0, 2500.0]],
        dims=("y", "x"),
        coords={"y": scene.coords["y"].values, "x": scene.coords["x"].values},
        name="dem",
    )
    canopy_path = tmp_path / "canopy.nc"
    dem_path = tmp_path / "dem.nc"
    canopy.to_netcdf(canopy_path)
    dem.to_netcdf(dem_path)

    ancillary = load_ancillary_layers(
        {
            "canopy_fraction": canopy_path,
            "dem": dem_path,
        },
        target_scene=scene,
    )

    assert set(ancillary.data_vars) == {"canopy_fraction", "dem"}
    np.testing.assert_allclose(
        ancillary["canopy_fraction"].values,
        [[0.5, 0.0], [np.nan, 0.1]],
    )
    np.testing.assert_allclose(
        ancillary["dem"].values,
        [[1000.0, np.nan], [np.nan, 2500.0]],
    )


def test_load_ancillary_layers_reads_geotiff_layer(tmp_path):
    scene = _scene()
    path = tmp_path / "slope.tiff"
    values = np.array([[[10.0, -1.0], [95.0, 45.0]]], dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32613",
        transform=from_origin(10, 2, 1, 1),
    ) as dataset:
        dataset.write(values)

    ancillary = load_ancillary_layers({"slope": path}, target_scene=scene)

    assert ancillary["slope"].dims == ("y", "x")
    np.testing.assert_allclose(
        ancillary["slope"].values,
        [[10.0, np.nan], [np.nan, 45.0]],
    )
