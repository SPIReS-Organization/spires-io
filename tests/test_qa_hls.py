# tests/test_qa_hls.py
import numpy as np
import pytest

from conftest import SHAPE, write_fake_hls_scene
from spires_io.hls.spec import decode_qa, read_acquisition_time, shipped_sza
from spires_io.raster_scene import read_band, resolve_one

CIRRUS, CLOUD, ADJACENT, SHADOW, SNOW, WATER = (1 << b for b in range(6))


@pytest.mark.parametrize("fmask, expected", [
    (0, True),
    (SNOW, True),
    (WATER, True),
    (SNOW | WATER | (3 << 6), True),      # aerosol level bits do not matter
    (CIRRUS, False),
    (CLOUD, False),
    (ADJACENT, False),
    (SHADOW, False),
    (255, False),                          # fill
])
def test_decode_fmask(fmask, expected):
    assert decode_qa({"fmask": np.array([[fmask]], np.uint8)}).tolist() == [[expected]]


def test_read_acquisition_time_from_sensing_time_tag(hls_l30_scene):
    assert read_acquisition_time(hls_l30_scene) == np.datetime64("2024-02-14T18:23:01")


def test_shipped_sza_masks_fill_value(tmp_path):
    sza = np.full(SHAPE, 45.0)
    sza[0, 0] = 400.0                       # 400.0 * 100 == 40000, the HLS fill
    scene = write_fake_hls_scene(tmp_path, "L30", sza=sza)
    template = read_band(resolve_one(scene, "blue", "*.B02.tif"))
    out = shipped_sza(scene, template)
    assert np.isnan(out.values[0, 0])
    assert np.allclose(out.values[1:, :], 45.0) and np.allclose(out.values[0, 1:], 45.0)


@pytest.mark.parametrize("fmask, expected", [
    (ADJACENT, True),                      # cloud_buffer not excluded
    (SHADOW, True),                        # cloud_shadow not excluded
    (CLOUD, False),
    (CIRRUS, False),
    (255, False),                          # fill
])
def test_decode_fmask_core_policy(fmask, expected):
    exclude = frozenset({"fill", "saturated", "cloud", "cirrus"})
    assert decode_qa({"fmask": np.array([[fmask]], np.uint8)}, exclude=exclude).tolist() == [[expected]]


from spires_contract.qa import QA_FLAGS
from spires_io.hls.spec import decode_qa_flags, QA_FLAGS_AVAILABLE


def test_decode_qa_flags_hls_fmask_bits():
    # Fmask 255 is the fill value and has every bit set, so cloud/snow/water are also
    # True at the fill pixel. Check each flag at the pixel it was written for.
    fmask = np.array([[1 << 1, 1 << 4, 1 << 5, 255]], np.uint8)   # cloud, snow, water, fill
    f = decode_qa_flags({"fmask": fmask})
    assert set(f) == set(QA_FLAGS)
    assert f["cloud"][0, 0] and not f["cloud"][0, 1] and not f["cloud"][0, 2]
    assert f["snow"][0, 1] and not f["snow"][0, 0] and not f["snow"][0, 2]
    assert f["water"][0, 2] and not f["water"][0, 0] and not f["water"][0, 1]
    assert f["fill"].tolist() == [[False, False, False, True]]
    assert QA_FLAGS_AVAILABLE == tuple(f for f in QA_FLAGS if f != "saturated")
