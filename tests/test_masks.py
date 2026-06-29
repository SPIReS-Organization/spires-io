import numpy as np
import xarray as xr

from spires_io.masks import load_external_mask


def _scene():
    coords = {"y": [0, 1], "x": [10, 11]}
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


def test_load_external_mask_reads_named_xarray_variable(tmp_path):
    scene = _scene()
    mask = xr.DataArray(
        [[0, 1], [1, 0]],
        dims=("y", "x"),
        coords={"y": scene.coords["y"].values, "x": scene.coords["x"].values},
        name="manual",
    )
    path = tmp_path / "mask.nc"
    xr.Dataset({"manual": mask, "other": mask}).to_netcdf(path)

    loaded = load_external_mask(path, target_scene=scene, variable="manual")

    assert loaded.dtype == np.dtype("bool")
    assert loaded.name == "manual"
    np.testing.assert_array_equal(loaded.values, [[False, True], [True, False]])
