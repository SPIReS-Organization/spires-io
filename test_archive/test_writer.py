import numpy as np
import pytest
import xarray as xr

from spires_io import SpiresData, SpiresDataWriter


def _scene():
    return xr.Dataset(
        {
            "reflectance": xr.DataArray(
                np.ones((2, 2, 2), dtype=np.float32),
                dims=("y", "x", "band"),
                coords={"y": [0, 1], "x": [10, 11], "band": ["a", "b"]},
                name="reflectance",
            ),
            "solar_zenith": xr.DataArray(
                np.full((2, 2), 30.0, dtype=np.float32),
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [10, 11]},
                name="solar_zenith",
            ),
            "valid_inversion_mask": xr.DataArray(
                np.ones((2, 2), dtype=bool),
                dims=("y", "x"),
                coords={"y": [0, 1], "x": [10, 11]},
                name="valid_inversion_mask",
            ),
        }
    )


def test_writer_from_data_preserves_context_and_policy():
    data = SpiresData.from_scene(_scene())
    output_policy = {"driver": "GTiff"}
    scene_metadata = {"tile": "h08v05", "water_year": 2024}

    writer = SpiresDataWriter.from_data(
        data,
        output_path="outputs/scene_spires.tif",
        output_policy=output_policy,
        scene_metadata=scene_metadata,
    )

    assert writer.data is data
    assert str(writer.output_path) == "outputs/scene_spires.tif"
    assert writer.output_policy == output_policy
    assert writer.scene_metadata == scene_metadata

    output_policy["driver"] = "netCDF"
    scene_metadata["tile"] = "mutated"
    assert writer.output_policy == {"driver": "GTiff"}
    assert writer.scene_metadata == {"tile": "h08v05", "water_year": 2024}


def test_writer_write_is_explicitly_unimplemented():
    writer = SpiresDataWriter.from_data(SpiresData.from_scene(_scene()))

    with pytest.raises(NotImplementedError, match="reserved"):
        writer.write({"snow_fraction": np.ones((2, 2))})
