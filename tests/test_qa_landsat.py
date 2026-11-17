# tests/test_qa_landsat.py
import numpy as np
import pytest

from spires_io.landsat.spec import decode_qa, read_acquisition_time

CLEAR = 21824           # bit 6 clear, all confidences low
FILL = 1 << 0
DILATED = 1 << 1
CIRRUS = 1 << 2
CLOUD = 1 << 3
SHADOW = 1 << 4
SNOW = 1 << 5
WATER = 1 << 7


@pytest.mark.parametrize("qa_pixel, radsat, expected", [
    (CLEAR, 0, True),
    (CLEAR | SNOW, 0, True),            # snow stays valid
    (CLEAR | WATER, 0, True),           # water stays valid
    (FILL, 0, False),
    (CLEAR | DILATED, 0, False),
    (CLEAR | CIRRUS, 0, False),
    (CLEAR | CLOUD, 0, False),
    (CLEAR | SHADOW, 0, False),
    (CLEAR, 1 << 1, False),             # band 2 saturated
    (CLEAR, 1 << 6, False),             # band 7 saturated
    (CLEAR, 1 << 0, True),              # band 1 (coastal, unused) saturated
    (CLEAR, 1 << 8, True),              # band 9 (cirrus, unused) saturated
])
def test_decode_qa_single_pixel(qa_pixel, radsat, expected):
    layers = {"qa_pixel": np.array([[qa_pixel]], np.uint16), "qa_radsat": np.array([[radsat]], np.uint16)}
    assert decode_qa(layers).tolist() == [[expected]]


def test_decode_qa_returns_bool_array_of_same_shape():
    layers = {"qa_pixel": np.full((3, 4), CLEAR, np.uint16), "qa_radsat": np.zeros((3, 4), np.uint16)}
    out = decode_qa(layers)
    assert out.dtype == bool and out.shape == (3, 4) and out.all()


def test_read_acquisition_time_from_mtl(landsat_scene):
    assert read_acquisition_time(landsat_scene) == np.datetime64("2024-02-14T18:23:01")


def test_read_acquisition_time_missing_mtl_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="MTL"):
        read_acquisition_time(tmp_path)


CORE = frozenset({"fill", "saturated", "cloud", "cirrus"})


@pytest.mark.parametrize("qa_pixel, radsat, expected", [
    (CLEAR | DILATED, 0, True),         # cloud_buffer not excluded
    (CLEAR | SHADOW, 0, True),          # cloud_shadow not excluded
    (CLEAR | CLOUD, 0, False),
    (CLEAR | CIRRUS, 0, False),
    (FILL, 0, False),
    (CLEAR, 1 << 1, False),
])
def test_decode_qa_core_policy_keeps_buffer_and_shadow(qa_pixel, radsat, expected):
    layers = {"qa_pixel": np.array([[qa_pixel]], np.uint16), "qa_radsat": np.array([[radsat]], np.uint16)}
    assert decode_qa(layers, exclude=CORE).tolist() == [[expected]]


def test_decode_qa_default_exclude_matches_conservative_policy():
    from spires_io.sensor_spec import DEFAULT_QA_EXCLUDE
    layers = {"qa_pixel": np.array([[CLEAR | DILATED, CLEAR | SHADOW, CLEAR]], np.uint16),
              "qa_radsat": np.zeros((1, 3), np.uint16)}
    assert decode_qa(layers).tolist() == decode_qa(layers, exclude=DEFAULT_QA_EXCLUDE).tolist() == [[False, False, True]]


from spires_contract.qa import QA_FLAGS, QA_INVALID_FLAGS
from spires_io.sensor_spec import DEFAULT_QA_EXCLUDE, QA_FLAGS as IO_QA_FLAGS


def test_qa_vocabulary_is_the_contract_vocabulary():
    assert IO_QA_FLAGS is QA_FLAGS
    assert DEFAULT_QA_EXCLUDE == frozenset(QA_INVALID_FLAGS)
    assert "snow" not in DEFAULT_QA_EXCLUDE


from spires_io.landsat.spec import decode_qa_flags, QA_FLAGS_AVAILABLE


def test_decode_qa_flags_landsat_bits():
    qa_pixel = np.array([[1 << 3, 1 << 5, 1 << 7, 0]], np.uint16)   # cloud, snow, water, clear
    radsat = np.array([[0, 0, 0, 1 << 2]], np.uint16)                 # last px saturated
    f = decode_qa_flags({"qa_pixel": qa_pixel, "qa_radsat": radsat})
    assert set(f) == set(QA_FLAGS)
    assert f["cloud"].tolist() == [[True, False, False, False]]
    assert f["snow"].tolist() == [[False, True, False, False]]
    assert f["water"].tolist() == [[False, False, True, False]]
    assert f["saturated"].tolist() == [[False, False, False, True]]
    assert QA_FLAGS_AVAILABLE == QA_FLAGS


def test_flags_to_layers_fills_missing_flags_and_rejects_unknown():
    from spires_io.sensor_spec import flags_to_layers
    shape = (2, 2)
    out = flags_to_layers({"cloud": np.ones(shape, bool)}, shape)
    assert list(out) == list(QA_FLAGS)
    assert out["cloud"].all() and not out["snow"].any()
    with pytest.raises(ValueError, match="not in QA_FLAGS"):
        flags_to_layers({"haze": np.ones(shape, bool)}, shape)
