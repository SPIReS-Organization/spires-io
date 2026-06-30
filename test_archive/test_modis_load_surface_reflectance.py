from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from spires_io.modis import (
    open_modis_surface_reflectance,
    parse_modis_surface_reflectance_filename,
    prepare_modis_scene_for_inversion,
)


def test_parse_modis_surface_reflectance_filename_for_terra():
    path = Path("MOD09GA.A2026123.h09v05.061.2026135123456.hdf")

    scene = parse_modis_surface_reflectance_filename(path)

    assert scene.product == "MOD09GA"
    assert scene.platform == "terra"
    assert scene.tile == "h09v05"
    assert scene.acquisition_date == "2026-05-03"
    assert scene.collection == "061"
    assert scene.processing_timestamp == "2026135123456"


def test_parse_modis_surface_reflectance_filename_for_aqua():
    path = Path("MYD09GA.A2026214.h10v04.061.2026216234567.hdf")

    scene = parse_modis_surface_reflectance_filename(path)

    assert scene.product == "MYD09GA"
    assert scene.platform == "aqua"
    assert scene.tile == "h10v04"
    assert scene.acquisition_date == "2026-08-02"
    assert scene.collection == "061"
    assert scene.processing_timestamp == "2026216234567"


def test_parse_modis_surface_reflectance_filename_rejects_unrecognized_names():
    with pytest.raises(ValueError, match="Unrecognized MODIS surface reflectance filename"):
        parse_modis_surface_reflectance_filename("not_a_modis_file.hdf")


def test_open_modis_surface_reflectance_requires_pyhdf_dependency(tmp_path):
    path = Path("SpiPy/data/modis/inputs/terra/reflectance/MOD09GA.A2023075.h08v05.061.2023077031843.hdf")
    if not path.exists():
        pytest.skip("Local MODIS sample file is not available")

    ds = open_modis_surface_reflectance(path)

    assert ds.attrs["platform"] == "terra"
    assert ds["reflectance_500m"].shape == (2400, 2400, 7)
    assert list(ds["band_500m"].values) == ["1", "2", "3", "4", "5", "6", "7"]
    assert ds["sensor_zenith"].shape == (1200, 1200)
    assert ds["state_1km"].shape == (1200, 1200)
    assert ds["qc_500m"].shape == (2400, 2400)
    assert "spatial_ref" in ds.coords


def build_mock_modis_raw_dataset():
    reflectance_500m = xr.DataArray(
        np.arange(28, dtype=np.float32).reshape(2, 2, 7),
        dims=("y_500m", "x_500m", "band_500m"),
        coords={"y_500m": [0, 1], "x_500m": [0, 1], "band_500m": ["1", "2", "3", "4", "5", "6", "7"]},
    )
    scalar_1km = xr.DataArray(
        np.ones((1, 1), dtype=np.float32),
        dims=("y_1km", "x_1km"),
        coords={"y_1km": [0], "x_1km": [0]},
    )
    scalar_500m = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("y_500m", "x_500m"),
        coords={"y_500m": [0, 1], "x_500m": [0, 1]},
    )
    qa_1km = xr.DataArray(
        np.full((1, 1), 8, dtype=np.uint16),
        dims=("y_1km", "x_1km"),
        coords={"y_1km": [0], "x_1km": [0]},
    )
    qc_500m = xr.DataArray(
        np.zeros((2, 2), dtype=np.uint32),
        dims=("y_500m", "x_500m"),
        coords={"y_500m": [0, 1], "x_500m": [0, 1]},
    )

    return xr.Dataset(
        data_vars={
            "reflectance_500m": reflectance_500m,
            "sensor_zenith": scalar_1km.copy(),
            "sensor_azimuth": scalar_1km.copy(),
            "solar_zenith": scalar_1km.copy(),
            "solar_azimuth": scalar_1km.copy(),
            "state_1km": qa_1km.copy(),
            "num_observations_1km": qa_1km.copy() + 2,
            "range_1km": qa_1km.copy(),
            "gflags_1km": qa_1km.copy(),
            "orbit_pnt_1km": qa_1km.copy(),
            "granule_pnt_1km": qa_1km.copy(),
            "num_observations_500m": scalar_500m.copy() + 2,
            "qc_500m": qc_500m,
            "obscov_500m": scalar_500m.copy(),
            "iobs_res_500m": scalar_500m.copy(),
            "q_scan_500m": scalar_500m.copy(),
        },
        coords={
            "y_1km": [0],
            "x_1km": [0],
            "y_500m": [0, 1],
            "x_500m": [0, 1],
            "band_500m": ["1", "2", "3", "4", "5", "6", "7"],
        },
        attrs={"platform": "terra"},
    )


def test_prepare_modis_scene_for_inversion_promotes_1km_fields_to_500m():
    raw = build_mock_modis_raw_dataset()
    ds = prepare_modis_scene_for_inversion(raw)

    assert ds["reflectance"].shape == (2, 2, 7)
    assert ds["sensor_zenith"].shape == (2, 2)
    assert ds["state_1km"].shape == (2, 2)
    assert "valid_inversion_mask" in ds
    assert "valid_r0_mask" in ds


def test_prepare_modis_scene_for_inversion_can_ignore_cloud_mask_for_inversion():
    raw = build_mock_modis_raw_dataset()
    cloud_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": raw["y_500m"].values, "x": raw["x_500m"].values},
    )

    strict = prepare_modis_scene_for_inversion(
        raw,
        cloud_mask_source=cloud_mask,
        cloud_mask_policy="strict",
    )
    relaxed = prepare_modis_scene_for_inversion(
        raw,
        cloud_mask_source=cloud_mask,
        cloud_mask_policy="ignore_cloud",
    )

    assert not bool(strict["valid_inversion_mask"].any())
    assert bool(relaxed["mask_cloud"].all())
    assert not bool(relaxed["mask_cloud_for_inversion"].any())
    assert bool(relaxed["valid_inversion_mask"].all())
    assert not bool(relaxed["valid_r0_mask"].any())
    assert relaxed.attrs["cloud_mask_policy"] == "ignore_cloud"


def test_prepare_modis_scene_for_inversion_keeps_water_valid_for_r0():
    raw = build_mock_modis_raw_dataset()
    raw["state_1km"] = xr.zeros_like(raw["state_1km"])

    ds = prepare_modis_scene_for_inversion(raw)

    assert bool(ds["mask_water"].all())
    assert not bool(ds["valid_inversion_mask"].any())
    assert bool(ds["valid_r0_mask"].all())
