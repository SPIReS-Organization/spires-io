# tests/test_fetch_sources.py
import fnmatch

import pytest

pytest.importorskip("pystac")
from fetch_helpers import hls_item, landsat_item, s2_item

from spires_io.fetch.sources import AOIS, SOURCES, asset_filename, sensor_from_scene_id
from spires_io.hls.spec import L30_SPEC, S30_SPEC
from spires_io.landsat.spec import SPEC as LANDSAT_SPEC
from spires_io.sentinel2.spec import SPEC as S2_SPEC


@pytest.mark.parametrize("scene_id, sensor", [
    ("LC08_L2SP_042034_20240214_02_T1", "landsat"),
    ("LC09_L2SP_042034_20240222_02_T1", "landsat"),
    ("HLS.L30.T11SLB.2024045T182301.v2.0", "hls_l30"),
    ("HLS.S30.T11SLB.2024045T183901.v2.0", "hls_s30"),
    ("S2B_11SLB_20250420_0_L2A", "sentinel2"),
    ("S2C_11SLB_20250420_0_L2A", "sentinel2"),
])
def test_sensor_from_scene_id(scene_id, sensor):
    assert sensor_from_scene_id(scene_id) == sensor


@pytest.mark.parametrize("bad", ["LE07_L2SP_042034_20200101_02_T1", "MOD09GA.A2024045", ""])
def test_sensor_from_scene_id_rejects_unknown(bad):
    with pytest.raises(ValueError, match="cannot infer sensor"):
        sensor_from_scene_id(bad)


def test_sources_cover_registered_sensors():
    assert set(SOURCES) == {"landsat", "hls_l30", "hls_s30", "sentinel2"}
    for name, source in SOURCES.items():
        assert source.sensor == name


def test_mammoth_aoi_contains_site():
    w, s, e, n = AOIS["mammoth"]
    assert w < -119.029 < e and s < 37.643 < n


def _filenames(source, item):
    return [asset_filename(item.assets[key].href) for key in source.assets]


@pytest.mark.parametrize("sensor, item, spec, extra", [
    ("landsat", landsat_item(), LANDSAT_SPEC, ["*_MTL.txt"]),
    ("hls_l30", hls_item("L30"), L30_SPEC, ["*.SZA.tif"]),
    ("hls_s30", hls_item("S30"), S30_SPEC, ["*.SZA.tif"]),
    ("sentinel2", s2_item(), S2_SPEC, []),
])
def test_asset_filenames_match_adapter_globs(sensor, item, spec, extra):
    """Every glob the adapter resolves must match exactly one downloaded filename, and nothing is downloaded unused."""
    names = _filenames(SOURCES[sensor], item)
    globs = list(spec.band_files.values()) + list(spec.qa_files.values()) + extra
    for pattern in globs:
        assert len(fnmatch.filter(names, pattern)) == 1, (pattern, names)
    assert len(names) == len(globs), f"{sensor}: downloading unused assets {names}"


def test_landsat_keeps_only_t1():
    src = SOURCES["landsat"]
    assert src.keep(landsat_item(category="T1"))
    assert not src.keep(landsat_item(category="T2"))


def test_sentinel2_keeps_only_offset_applied():
    src = SOURCES["sentinel2"]
    assert src.keep(s2_item(boa=True))
    assert not src.keep(s2_item(boa=False))
    assert not src.keep(s2_item(boa=None))


def test_keys():
    assert SOURCES["landsat"].tile(landsat_item()) == "042034"
    assert SOURCES["landsat"].acquisition_key(landsat_item())[1] == "landsat-8"
    assert SOURCES["hls_l30"].tile(hls_item("L30")) == "T11SLB"
    assert SOURCES["hls_l30"].acquisition_key(hls_item("L30")) == ("2024045T182301",)
    assert SOURCES["sentinel2"].tile(s2_item()) == "MGRS-11SLB"
    assert SOURCES["sentinel2"].acquisition_key(s2_item()) == ("GS2B_20250420T183909_042410",)
    assert SOURCES["sentinel2"].nodata_pct(s2_item(nodata=47.4)) == 47.4
    assert SOURCES["landsat"].nodata_pct(landsat_item()) is None
