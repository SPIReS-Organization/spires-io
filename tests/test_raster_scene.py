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
