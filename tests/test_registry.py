# tests/test_registry.py
import pytest

from spires_io.api import open_surface_reflectance, prepare_scene_for_inversion
from spires_io.registry import get_sensor_adapter, list_supported_sensor_platforms, list_supported_sensors


def test_new_sensors_registered():
    assert set(list_supported_sensors()) >= {"modis", "viirs", "landsat", "hls_l30", "hls_s30", "sentinel2"}


def test_platforms():
    platforms = list_supported_sensor_platforms()
    assert platforms["landsat"] == ("landsat8", "landsat9")
    assert platforms["hls_l30"] == ("landsat8", "landsat9")
    assert platforms["hls_s30"] == ("sentinel2a", "sentinel2b", "sentinel2c")
    assert platforms["sentinel2"] == ("sentinel2a", "sentinel2b", "sentinel2c")


@pytest.mark.parametrize("alias, canonical", [
    ("landsat", "landsat"), ("Landsat-8", "landsat"), ("LC08", "landsat"), ("LC09", "landsat"),
    ("hls_l30", "hls_l30"), ("HLS.L30", "hls_l30"), ("hls_s30", "hls_s30"), ("HLSS30", "hls_s30"),
    ("sentinel2", "sentinel2"), ("Sentinel-2", "sentinel2"), ("S2", "sentinel2"), ("MSI", "sentinel2"),
])
def test_sensor_aliases(alias, canonical):
    assert get_sensor_adapter(alias).sensor == canonical


def test_platform_aliases():
    assert get_sensor_adapter("landsat", "LANDSAT_8").sensor == "landsat"
    assert get_sensor_adapter("sentinel2", "S2B").sensor == "sentinel2"
    with pytest.raises(ValueError, match="Unsupported platform"):
        get_sensor_adapter("hls_l30", "sentinel2a")


def test_dispatch_opens_landsat(landsat_scene):
    ds = open_surface_reflectance(landsat_scene, sensor="landsat")
    assert ds.attrs["sensor"] == "landsat"
    assert prepare_scene_for_inversion(landsat_scene, sensor="landsat")["reflectance"].dims == ("y", "x", "band")


def test_dispatch_opens_hls_and_sentinel2(hls_s30_scene, sentinel2_scene):
    assert open_surface_reflectance(hls_s30_scene, sensor="hls_s30").attrs["sensor"] == "hls_s30"
    assert open_surface_reflectance(sentinel2_scene, sensor="sentinel2", bands=["red", "swir1"]).band.size == 2
