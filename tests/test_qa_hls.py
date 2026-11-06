# tests/test_qa_hls.py
import numpy as np
import pytest

from spires_io.hls.spec import decode_qa, read_acquisition_time

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
