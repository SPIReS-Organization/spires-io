# tests/test_raster_scene.py
import numpy as np
import pytest
import xarray as xr
from spires_contract import ContractError
from spires_contract.spectra import validate_solar_angles, validate_target_spectra

from spires_io.landsat import SPEC as LANDSAT, open_landsat_scene
from spires_io.raster_scene import open_raster_scene
from spires_io.sensor_spec import SceneLayoutError
from conftest import write_fake_landsat_scene


def test_landsat_dataset_shape_and_contract(landsat_scene):
    ds = open_landsat_scene(landsat_scene)
    assert ds["reflectance"].dims == ("y", "x", "band")
    assert ds["reflectance"].dtype == np.float32
    assert list(ds.band.values) == ["blue", "green", "red", "nir", "swir1", "swir2"]
    assert ds["valid"].dtype == bool and ds["valid"].dims == ("y", "x")
    assert ds["solar_zenith"].dtype == np.float32 and ds["solar_zenith"].attrs["units"] == "degree"
    validate_target_spectra(ds["reflectance"])
    validate_solar_angles(ds["solar_zenith"])
    assert ds.rio.crs.to_epsg() == 32611
    assert ds["reflectance"].wavelength_nm.values.tolist() == [482.0, 561.0, 655.0, 865.0, 1609.0, 2201.0]


def test_landsat_scale_offset_applied(landsat_scene):
    ds = open_landsat_scene(landsat_scene)
    # DN 4000 * 2.75e-5 - 0.2 = -0.09 (fake DN; sign irrelevant, arithmetic is the point)
    assert ds["reflectance"].values == pytest.approx(4000 * 2.75e-5 - 0.2, abs=1e-6)


def test_landsat_nodata_becomes_nan_and_invalid(tmp_path):
    blue = np.full((8, 8), 4000, np.uint16)
    blue[0, 0] = 0
    scene = write_fake_landsat_scene(tmp_path, dn={"blue": blue})
    ds = open_landsat_scene(scene)
    assert np.isnan(ds["reflectance"].sel(band="blue").values[0, 0])
    assert not np.isnan(ds["reflectance"].sel(band="green").values[0, 0])
    assert not ds["valid"].values[0, 0]
    assert ds["valid"].values[1:].all()


def test_landsat_attrs(landsat_scene):
    ds = open_landsat_scene(landsat_scene)
    assert ds.attrs["sensor"] == "landsat"
    assert ds.attrs["platform"] == "LANDSAT_8"
    assert ds.attrs["acquisition_time"] == "2024-02-14T18:23:01"
    assert ds.attrs["resolution_m"] == 30
    assert ds.attrs["source_product_id"] == landsat_scene.name
    assert ds.rio.crs.to_epsg() == 32611  # CRS lives in the rio accessor, not a plain attr


def test_landsat_solar_zenith_is_plausible(landsat_scene):
    # Mammoth, mid-February, 18:23 UTC (10:23 PST): NOAA gives ~56.7 deg.
    sza = open_landsat_scene(landsat_scene)["solar_zenith"].values
    assert 50.0 < sza.mean() < 62.0
    assert np.ptp(sza) < 0.01     # 240 m scene: essentially constant


def test_band_subset(landsat_scene):
    ds = open_landsat_scene(landsat_scene, bands=["red", "nir"])
    assert list(ds.band.values) == ["red", "nir"]


def test_unknown_band_raises(landsat_scene):
    with pytest.raises(ValueError, match="swir3"):
        open_landsat_scene(landsat_scene, bands=["swir3"])


def test_missing_band_file_raises_before_reading(landsat_scene):
    (landsat_scene / f"{landsat_scene.name}_SR_B5.TIF").unlink()
    with pytest.raises(SceneLayoutError, match="nir"):
        open_landsat_scene(landsat_scene)


def test_ambiguous_band_file_raises(landsat_scene):
    extra = landsat_scene / "duplicate_SR_B5.TIF"
    extra.write_bytes((landsat_scene / f"{landsat_scene.name}_SR_B5.TIF").read_bytes())
    with pytest.raises(SceneLayoutError, match="2 matches"):
        open_landsat_scene(landsat_scene)


def test_incompatible_target_resolution_raises(landsat_scene):
    with pytest.raises(ValueError, match="integer"):
        open_landsat_scene(landsat_scene, target_resolution_m=20)


def test_custom_resampler_is_used(landsat_scene):
    calls = []

    def doubling(bands, target_m):
        calls.append(target_m)
        return {name: da * 2 for name, da in bands.items()}

    ds = open_landsat_scene(landsat_scene, resampler=doubling)
    assert calls == [30]
    assert ds["reflectance"].values == pytest.approx(2 * (4000 * 2.75e-5 - 0.2), abs=1e-6)


from spires_io.hls import open_hls_l30_scene, open_hls_s30_scene
from conftest import write_fake_hls_scene


def test_hls_l30_contract_and_values(hls_l30_scene):
    ds = open_hls_l30_scene(hls_l30_scene)
    validate_target_spectra(ds["reflectance"])
    validate_solar_angles(ds["solar_zenith"])
    assert list(ds.band.values) == ["coastal", "blue", "green", "red", "nir", "swir1", "swir2"]
    assert ds["reflectance"].values == pytest.approx(0.4, abs=1e-6)
    assert ds["solar_zenith"].values == pytest.approx(45.0, abs=1e-4)
    assert ds["valid"].values.all()
    assert ds.attrs["sensor"] == "hls_l30" and ds.attrs["acquisition_time"] == "2024-02-14T18:23:01"


def test_hls_s30_bands(hls_s30_scene):
    ds = open_hls_s30_scene(hls_s30_scene)
    assert list(ds.band.values) == ["coastal", "blue", "green", "red", "red_edge1", "red_edge2",
                                    "red_edge3", "nir", "nir_narrow", "swir1", "swir2"]
    assert ds.attrs["sensor"] == "hls_s30"


def test_hls_nodata_and_fmask_both_invalidate(tmp_path):
    red = np.full((8, 8), 4000, np.int16)
    red[2, 2] = -9999
    fmask = np.zeros((8, 8), np.uint8)
    fmask[3, 3] = 1 << 1                       # cloud
    ds = open_hls_l30_scene(write_fake_hls_scene(tmp_path, "L30", dn={"red": red}, qa={"fmask": fmask}))
    assert np.isnan(ds["reflectance"].sel(band="red").values[2, 2])
    assert not ds["valid"].values[2, 2] and not ds["valid"].values[3, 3]
    assert ds["valid"].values.sum() == 62


from spires_io.sentinel2 import open_sentinel2_scene
from conftest import write_fake_sentinel2_scene


def test_sentinel2_default_20m_grid_and_contract(sentinel2_scene):
    ds = open_sentinel2_scene(sentinel2_scene)
    validate_target_spectra(ds["reflectance"])
    validate_solar_angles(ds["solar_zenith"])
    assert ds["reflectance"].shape == (8, 8, 10)
    assert ds.attrs["resolution_m"] == 20
    assert ds.attrs["platform"] == "sentinel-2b"
    assert ds.attrs["acquisition_time"] == "2025-04-20T18:54:03"
    # DN 4000 * 1e-4, NO additional -0.1: E84 COGs are already offset-harmonized.
    assert ds["reflectance"].values == pytest.approx(0.4, abs=1e-6)
    assert ds["valid"].values.all()


def test_sentinel2_solar_zenith_matches_stac_sun_elevation(sentinel2_scene):
    # Real item S2B_11SLB_20250420_0_L2A reports view:sun_elevation 61.13 deg at tile centre;
    # our fake grid sits near the tile's NW corner, so allow a degree.
    sza = open_sentinel2_scene(sentinel2_scene)["solar_zenith"].values
    assert sza.mean() == pytest.approx(90.0 - 61.13, abs=1.0)


def test_sentinel2_10m_bands_are_block_averaged(tmp_path):
    blue = np.indices((16, 16)).sum(axis=0) % 2 * 2000 + 3000     # 3000/5000 checkerboard
    ds = open_sentinel2_scene(write_fake_sentinel2_scene(tmp_path, dn={"blue": blue}))
    assert ds["reflectance"].sel(band="blue").values == pytest.approx(0.4, abs=1e-6)


def test_sentinel2_upsample_to_10m_with_nearest(sentinel2_scene):
    from spires_io.sensor_spec import nearest
    ds = open_sentinel2_scene(sentinel2_scene, target_resolution_m=10, resampler=nearest)
    assert ds["reflectance"].shape == (16, 16, 10)
    assert ds["valid"].shape == (16, 16)


def test_sentinel2_scl_cloud_invalidates(tmp_path):
    scl = np.full((8, 8), 4, np.uint8)
    scl[0, :] = 9
    ds = open_sentinel2_scene(write_fake_sentinel2_scene(tmp_path, qa={"scl": scl}))
    assert not ds["valid"].values[0].any() and ds["valid"].values[1:].all()


def test_bbox_clips_reflectance_valid_and_sza(hls_l30_scene):
    full = open_hls_l30_scene(hls_l30_scene)
    bbox = (300120.0, 4169760.0, 300240.0, 4169880.0)   # lower-right 4x4 of the 8x8 fake
    ds = open_hls_l30_scene(hls_l30_scene, bbox=bbox)
    assert ds["reflectance"].shape == (4, 4, 7)
    assert ds["valid"].shape == (4, 4)
    assert ds["solar_zenith"].shape == (4, 4)
    np.testing.assert_allclose(ds.x.values, full.x.values[4:])
    np.testing.assert_allclose(ds.y.values, full.y.values[4:])
    np.testing.assert_array_equal(ds["reflectance"].values, full["reflectance"].values[4:, 4:, :])
    validate_target_spectra(ds["reflectance"])
    validate_solar_angles(ds["solar_zenith"])


def test_bbox_outside_scene_raises(hls_l30_scene):
    with pytest.raises(ValueError, match="bbox"):
        open_hls_l30_scene(hls_l30_scene, bbox=(0.0, 0.0, 100.0, 100.0))


def test_bbox_none_is_full_scene(hls_l30_scene):
    assert open_hls_l30_scene(hls_l30_scene, bbox=None)["reflectance"].shape == (8, 8, 7)
