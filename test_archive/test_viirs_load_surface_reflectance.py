from pathlib import Path

import h5py
import pytest
import numpy as np
import xarray as xr

from spires_io.viirs import (
    open_viirs_surface_reflectance,
    parse_viirs_surface_reflectance_filename,
    prepare_viirs_scene_for_inversion,
)
from spires_io.viirs import bands as viirs_bands
from spires_io.viirs.bands import infer_viirs_lut_band_names_from_path
from spires_io.viirs.fields import VIIRS_1KM_GRID, VIIRS_500M_GRID


def write_mock_viirs_hdf(path: Path) -> None:
    with h5py.File(path, "w") as hdf:
        info = hdf.create_group("HDFEOS INFORMATION")
        info.create_dataset(
            "StructMetadata.0",
            data=np.bytes_(
                """
GROUP=SwathStructure
END_GROUP=SwathStructure
GROUP=GridStructure
	GROUP=GRID_1
		GridName="VIIRS_Grid_1km_2D"
		XDim=1
		YDim=1
		UpperLeftPointMtrs=(0,1)
		LowerRightMtrs=(1,0)
		Projection=HE5_GCTP_SNSOID
		ProjParams=(6371007.181000,0,0,0,0,0,0,0,0,0,0,0,0)
		SphereCode=-1
		GridOrigin=HE5_HDFE_GD_UL
	END_GROUP=GRID_1
	GROUP=GRID_2
		GridName="VIIRS_Grid_500m_2D"
		XDim=2
		YDim=2
		UpperLeftPointMtrs=(0,1)
		LowerRightMtrs=(1,0)
		Projection=HE5_GCTP_SNSOID
		ProjParams=(6371007.181000,0,0,0,0,0,0,0,0,0,0,0,0)
		SphereCode=-1
		GridOrigin=HE5_HDFE_GD_UL
	END_GROUP=GRID_2
END_GROUP=GridStructure
END
""".strip()
            ),
        )
        grid_1km = hdf.create_group(VIIRS_1KM_GRID)
        grid_500m = hdf.create_group(VIIRS_500M_GRID)
        data_fields_1km = grid_1km.create_group("Data Fields")
        data_fields_500m = grid_500m.create_group("Data Fields")

        x_1km = grid_1km.create_dataset("XDim", data=np.array([0], dtype=np.int16))
        y_1km = grid_1km.create_dataset("YDim", data=np.array([0], dtype=np.int16))
        x_500m = grid_500m.create_dataset("XDim", data=np.array([0, 1], dtype=np.int16))
        y_500m = grid_500m.create_dataset("YDim", data=np.array([0, 1], dtype=np.int16))
        for coord in (x_1km, y_1km, x_500m, y_500m):
            coord.attrs["units"] = np.bytes_("meters")

        for idx, band in enumerate(["M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11"], start=1):
            dataset = data_fields_1km.create_dataset(f"SurfReflect_{band}_1", data=np.array([[idx]], dtype=np.int16))
            dataset.attrs["units"] = np.bytes_("reflectance")

        for idx, band in enumerate(["I1", "I2", "I3"], start=1):
            dataset = data_fields_500m.create_dataset(
                f"SurfReflect_{band}_1",
                data=np.full((2, 2), idx, dtype=np.int16),
            )
            dataset.attrs["units"] = np.bytes_("reflectance")

        scalar_1km = np.ones((1, 1), dtype=np.int16)
        scalar_500m = np.ones((2, 2), dtype=np.int16)
        for name in ["SolarZenith_1", "SolarAzimuth_1", "SensorZenith_1", "SensorAzimuth_1"]:
            data_fields_1km.create_dataset(name, data=scalar_1km)
        for name in [
            "SurfReflect_QF1_1",
            "SurfReflect_QF2_1",
            "SurfReflect_QF3_1",
            "SurfReflect_QF4_1",
            "SurfReflect_QF5_1",
            "SurfReflect_QF6_1",
            "SurfReflect_QF7_1",
            "obscov_1km_1",
            "orbit_pnt_1",
        ]:
            data_fields_1km.create_dataset(name, data=np.zeros((1, 1), dtype=np.uint16))
        data_fields_1km.create_dataset("land_water_mask_1", data=np.ones((1, 1), dtype=np.uint8))
        data_fields_1km.create_dataset("num_observations_1km", data=np.full((1, 1), 2, dtype=np.uint8))

        data_fields_500m.create_dataset("iobs_res_1", data=scalar_500m)
        data_fields_500m.create_dataset("num_observations_500m", data=np.full((2, 2), 2, dtype=np.uint8))
        data_fields_500m.create_dataset("obscov_500m_1", data=np.zeros((2, 2), dtype=np.uint16))


def build_mock_viirs_raw_dataset():
    reflectance_1km = xr.DataArray(
        np.arange(9, dtype=np.float32).reshape(1, 1, 9),
        dims=("y_1km", "x_1km", "band_1km"),
        coords={"y_1km": [0], "x_1km": [0], "band_1km": ["M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11"]},
    )
    reflectance_500m = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        dims=("y_500m", "x_500m", "band_500m"),
        coords={"y_500m": [0, 1], "x_500m": [0, 1], "band_500m": ["I1", "I2", "I3"]},
    )

    scalar_1km = xr.DataArray(np.ones((1, 1), dtype=np.float32), dims=("y_1km", "x_1km"), coords={"y_1km": [0], "x_1km": [0]})
    scalar_500m = xr.DataArray(np.ones((2, 2), dtype=np.float32), dims=("y_500m", "x_500m"), coords={"y_500m": [0, 1], "x_500m": [0, 1]})
    qa_1km = xr.DataArray(np.zeros((1, 1), dtype=np.uint16), dims=("y_1km", "x_1km"), coords={"y_1km": [0], "x_1km": [0]})

    return xr.Dataset(
        data_vars={
            "reflectance_1km": reflectance_1km,
            "reflectance_500m": reflectance_500m,
            "solar_zenith": scalar_1km.copy(),
            "solar_azimuth": scalar_1km.copy(),
            "sensor_zenith": scalar_1km.copy(),
            "sensor_azimuth": scalar_1km.copy(),
            "qa_qf1": qa_1km.copy(),
            "qa_qf2": qa_1km.copy(),
            "qa_qf3": qa_1km.copy(),
            "qa_qf4": qa_1km.copy(),
            "qa_qf5": qa_1km.copy(),
            "qa_qf6": qa_1km.copy(),
            "qa_qf7": qa_1km.copy(),
            "land_water_mask": qa_1km.copy() + 1,
            "num_observations_1km": qa_1km.copy() + 2,
            "obscov_1km": qa_1km.copy(),
            "orbit_pnt": qa_1km.copy(),
            "iobs_res": scalar_500m.copy(),
            "num_observations_500m": scalar_500m.copy() + 2,
            "obscov_500m": scalar_500m.copy(),
        },
        coords={
            "y_1km": [0],
            "x_1km": [0],
            "y_500m": [0, 1],
            "x_500m": [0, 1],
            "band_1km": ["M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11"],
            "band_500m": ["I1", "I2", "I3"],
        },
        attrs={"platform": "noaa20"},
    )


def test_parse_viirs_surface_reflectance_filename_vnp():
    scene = parse_viirs_surface_reflectance_filename("VNP09GA.A2026112.h08v05.002.2026113100255.h5")
    assert scene.product == "VNP09GA"
    assert scene.platform == "snpp"
    assert scene.tile == "h08v05"
    assert scene.acquisition_date == "2026-04-22"
    assert scene.collection == "002"


def test_parse_viirs_surface_reflectance_filename_vj1():
    scene = parse_viirs_surface_reflectance_filename("VJ109GA.A2026112.h08v05.002.2026113072313.h5")
    assert scene.product == "VJ109GA"
    assert scene.platform == "noaa20"
    assert scene.tile == "h08v05"
    assert scene.acquisition_date == "2026-04-22"
    assert scene.collection == "002"


def test_parse_viirs_surface_reflectance_filename_vj2():
    scene = parse_viirs_surface_reflectance_filename("VJ209GA.A2026112.h08v05.002.2026113072313.h5")
    assert scene.product == "VJ209GA"
    assert scene.platform == "noaa21"
    assert scene.tile == "h08v05"
    assert scene.acquisition_date == "2026-04-22"
    assert scene.collection == "002"


def test_infer_viirs_lut_band_names_from_path():
    bands = infer_viirs_lut_band_names_from_path(
        "SpiPy/tests/data/lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat"
    )
    assert bands == ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]


def test_infer_viirs_lut_band_names_from_path_returns_none_without_band_tokens():
    bands = infer_viirs_lut_band_names_from_path("SpiPy/tests/data/lut_viirs_noaa20.mat")
    assert bands is None


def test_resolve_viirs_inversion_bands_prefers_metadata_over_filename(monkeypatch):
    monkeypatch.setattr(
        viirs_bands,
        "infer_viirs_lut_band_names_from_metadata",
        lambda path: ["I1", "I2", "M4"],
    )
    monkeypatch.setattr(
        viirs_bands,
        "infer_viirs_lut_band_names_from_path",
        lambda path: ["I1", "I2", "I3", "M2"],
    )

    bands = viirs_bands.resolve_viirs_inversion_bands(
        lut_file="SpiPy/tests/data/lut_viirs_noaa20_i1_i2_i3_m2.mat"
    )

    assert bands == ["I1", "I2", "M4"]


def test_resolve_viirs_inversion_bands_falls_back_to_all_available_bands(monkeypatch):
    monkeypatch.setattr(
        viirs_bands,
        "infer_viirs_lut_band_names_from_metadata",
        lambda path: None,
    )
    monkeypatch.setattr(
        viirs_bands,
        "infer_viirs_lut_band_names_from_path",
        lambda path: None,
    )

    bands = viirs_bands.resolve_viirs_inversion_bands(lut_file="SpiPy/tests/data/lut_viirs_noaa20.mat")

    assert bands == list(viirs_bands.VIIRS_ANALYSIS_BANDS)


def test_prepare_viirs_scene_for_inversion_can_subset_bands_from_lut_path():
    raw = build_mock_viirs_raw_dataset()

    ds = prepare_viirs_scene_for_inversion(
        raw,
        lut_file="SpiPy/tests/data/lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat",
    )

    assert list(ds["band"].values) == ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]
    assert ds["reflectance"].attrs["band_selection_source"] in {"metadata", "filename"}
    assert "band_500m" not in ds.coords
    assert "band_1km" not in ds.coords
    assert "reflectance_500m_native" not in ds.data_vars
    assert "reflectance_1km_on_500m" not in ds.data_vars


def test_prepare_viirs_scene_for_inversion_can_keep_intermediate_reflectance():
    raw = build_mock_viirs_raw_dataset()

    ds = prepare_viirs_scene_for_inversion(
        raw,
        keep_intermediate_reflectance=True,
    )

    assert "reflectance_500m_native" in ds.data_vars
    assert "reflectance_1km_on_500m" in ds.data_vars


def test_prepare_viirs_scene_for_inversion_can_ignore_cloud_mask_for_inversion():
    raw = build_mock_viirs_raw_dataset()
    cloud_mask = xr.DataArray(
        np.ones((2, 2), dtype=bool),
        dims=("y", "x"),
        coords={"y": raw["y_500m"].values, "x": raw["x_500m"].values},
    )

    strict = prepare_viirs_scene_for_inversion(
        raw,
        cloud_mask_source=cloud_mask,
        cloud_mask_policy="strict",
    )
    relaxed = prepare_viirs_scene_for_inversion(
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


def test_prepare_viirs_scene_for_inversion_keeps_water_valid_for_r0():
    raw = build_mock_viirs_raw_dataset()
    raw["land_water_mask"] = xr.zeros_like(raw["land_water_mask"])

    ds = prepare_viirs_scene_for_inversion(raw)

    assert bool(ds["mask_water"].all())
    assert not bool(ds["valid_inversion_mask"].any())
    assert bool(ds["valid_r0_mask"].all())


def test_prepare_viirs_scene_for_inversion_treats_deep_ocean_as_water():
    raw = build_mock_viirs_raw_dataset()
    raw["land_water_mask"] = xr.full_like(raw["land_water_mask"], 7)

    ds = prepare_viirs_scene_for_inversion(raw)

    assert bool(ds["mask_water"].all())
    assert not bool(ds["valid_inversion_mask"].any())
    assert bool(ds["valid_r0_mask"].all())


def test_open_viirs_surface_reflectance_reads_only_requested_lut_bands(tmp_path):
    path = tmp_path / "VJ109GA.A2026112.h08v05.002.2026113072313.h5"
    write_mock_viirs_hdf(path)

    ds = open_viirs_surface_reflectance(
        path,
        lut_file="SpiPy/tests/data/lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat",
    )

    assert list(ds["band_500m"].values) == ["I1", "I2", "I3"]
    assert list(ds["band_1km"].values) == ["M2", "M4", "M8", "M11"]
    assert ds["reflectance_1km"].shape == (1, 1, 4)
    assert ds["reflectance_500m"].shape == (2, 2, 3)
    assert ds.attrs["selected_bands"] == ["I1", "I2", "I3", "M2", "M4", "M8", "M11"]
    assert ds.attrs["band_selection_source"] in {"metadata", "filename"}
    assert "spatial_ref" in ds
    assert ds["reflectance_500m"].attrs["grid_mapping"] == "spatial_ref"
    assert ds["spatial_ref"].attrs["grid_mapping_name"] == "sinusoidal"
    np.testing.assert_allclose(
        [float(value) for value in ds["spatial_ref"].attrs["GeoTransform"].split()],
        [0.0, 0.5, 0.0, 1.0, 0.0, -0.5],
    )


def test_prepare_viirs_scene_for_inversion_preserves_spatial_ref(tmp_path):
    path = tmp_path / "VJ109GA.A2026112.h08v05.002.2026113072313.h5"
    write_mock_viirs_hdf(path)

    ds = prepare_viirs_scene_for_inversion(
        path,
        lut_file="SpiPy/tests/data/lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat",
    )

    assert "spatial_ref" in ds
    assert ds["reflectance"].attrs["grid_mapping"] == "spatial_ref"
    assert ds["valid_inversion_mask"].attrs["grid_mapping"] == "spatial_ref"
    assert ds["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert ds["y"].attrs["standard_name"] == "projection_y_coordinate"


@pytest.mark.parametrize(
    "filename,expected_platform",
    [
        ("VNP09GA.A2026112.h08v05.002.2026113100255.h5", "snpp"),
        ("VJ109GA.A2026112.h08v05.002.2026113072313.h5", "noaa20"),
        ("VJ209GA.A2026112.h08v05.002.2026113072313.h5", "noaa21"),
    ],
)
def test_open_viirs_surface_reflectance_download_example(filename, expected_platform):
    path = Path.home() / "Downloads" / filename
    if not path.exists():
        pytest.skip(f"sample file not available: {path}")

    ds = open_viirs_surface_reflectance(path)

    assert ds.attrs["platform"] == expected_platform
    assert ds["reflectance_1km"].dims == ("y_1km", "x_1km", "band_1km")
    assert ds["reflectance_500m"].dims == ("y_500m", "x_500m", "band_500m")
    assert list(ds["band_1km"].values) == ["M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11"]
    assert list(ds["band_500m"].values) == ["I1", "I2", "I3"]


@pytest.mark.parametrize(
    "filename",
    [
        "VNP09GA.A2026112.h08v05.002.2026113100255.h5",
        "VJ109GA.A2026112.h08v05.002.2026113072313.h5",
        "VJ209GA.A2026112.h08v05.002.2026113072313.h5",
    ],
)
def test_prepare_viirs_scene_for_inversion_download_example(filename):
    path = Path.home() / "Downloads" / filename
    if not path.exists():
        pytest.skip(f"sample file not available: {path}")

    ds = prepare_viirs_scene_for_inversion(path)

    assert ds["reflectance"].dims == ("y", "x", "band")
    assert ds["solar_zenith"].dims == ("y", "x")
    assert ds["qa_raw_stack"].dims == ("y", "x", "qa_flag")
    assert ds["valid_inversion_mask"].dims == ("y", "x")
    assert ds["valid_r0_mask"].dims == ("y", "x")
    assert list(ds["band"].values) == ["I1", "I2", "I3", "M1", "M2", "M3", "M4", "M5", "M7", "M8", "M10", "M11"]
    assert ds["reflectance"].shape[:2] == ds["solar_zenith"].shape
