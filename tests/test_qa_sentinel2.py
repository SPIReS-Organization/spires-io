# tests/test_qa_sentinel2.py
import json

import numpy as np
import pytest

from spires_io.sensor_spec import SceneLayoutError
from spires_io.sentinel2.spec import decode_qa, read_acquisition_time, read_item, read_platform


@pytest.mark.parametrize("scl, expected", [
    (0, False),    # no data
    (1, False),    # saturated / defective
    (2, False),    # cast shadow
    (3, False),    # cloud shadow
    (4, True),     # vegetation
    (5, True),     # not vegetated
    (6, True),     # water
    (7, True),     # unclassified
    (8, False),    # cloud medium probability
    (9, False),    # cloud high probability
    (10, False),   # thin cirrus
    (11, True),    # snow / ice
])
def test_decode_scl(scl, expected):
    assert decode_qa({"scl": np.array([[scl]], np.uint8)}).tolist() == [[expected]]


def test_read_acquisition_time(sentinel2_scene):
    assert read_acquisition_time(sentinel2_scene) == np.datetime64("2025-04-20T18:54:03")


def test_read_platform(sentinel2_scene):
    assert read_platform(sentinel2_scene) == "sentinel-2b"


def test_missing_item_json_raises(sentinel2_scene):
    (sentinel2_scene / "item.json").unlink()
    with pytest.raises(SceneLayoutError, match="item.json"):
        read_item(sentinel2_scene)


def test_item_without_boa_offset_applied_is_refused(tmp_path):
    from conftest import write_fake_sentinel2_scene
    scene = write_fake_sentinel2_scene(tmp_path, boa_offset_applied=False)
    with pytest.raises(ValueError, match="boa_offset_applied"):
        read_item(scene)
