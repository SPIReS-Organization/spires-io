# tests/test_resample.py
import numpy as np
import pytest
import rioxarray  # noqa: F401  # register the .rio accessor
import xarray as xr
from affine import Affine

from spires_io.sensor_spec import block_mean, nearest, snap_bbox_outward

CRS = "EPSG:32611"
ORIGIN = (300000.0, 4170000.0)   # upper-left of the fake UTM tile used across the offline tests


def _grid(values, resolution_m, x0=300000.0, y0=4170000.0):
    """North-up DataArray whose pixel centres follow the GeoTIFF convention."""
    ny, nx = values.shape
    x = x0 + resolution_m * (np.arange(nx) + 0.5)
    y = y0 - resolution_m * (np.arange(ny) + 0.5)
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs(CRS).rio.write_transform(
        Affine(resolution_m, 0.0, x0, 0.0, -resolution_m, y0)
    )


def test_block_mean_10_to_20_m_is_analytic_mean():
    checker = np.indices((4, 4)).sum(axis=0) % 2  # 0/1 checkerboard
    da = _grid(checker.astype(np.float32), 10)
    out = block_mean({"blue": da}, 20)["blue"]
    assert out.shape == (2, 2)
    assert np.allclose(out.values, 0.5)
    assert np.allclose(out.x.values, [300010.0, 300030.0])
    assert np.allclose(out.y.values, [4169990.0, 4169970.0])
    assert out.rio.crs.to_string() == CRS
    assert abs(out.rio.resolution()[0]) == 20.0


def test_block_mean_ignores_nan_and_keeps_all_nan_blocks_nan():
    values = np.array([[1.0, np.nan, np.nan, np.nan],
                       [3.0, np.nan, np.nan, np.nan],
                       [5.0, 5.0,    7.0,    7.0],
                       [5.0, 5.0,    7.0,    7.0]], dtype=np.float32)
    out = block_mean({"b": _grid(values, 10)}, 20)["b"]
    assert out.values[0, 0] == pytest.approx(2.0)
    assert np.isnan(out.values[0, 1])
    assert out.values[1, 1] == pytest.approx(7.0)


def test_block_mean_leaves_bands_already_at_target_untouched():
    da = _grid(np.ones((2, 2), dtype=np.float32), 20)
    out = block_mean({"swir1": da}, 20)["swir1"]
    assert out is da


def test_nearest_20_to_10_m_repeats_values():
    da = _grid(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 20)
    out = nearest({"swir1": da}, 10)["swir1"]
    assert out.shape == (4, 4)
    assert np.array_equal(out.values[:2, :2], np.ones((2, 2)))
    assert np.allclose(out.x.values, [300005.0, 300015.0, 300025.0, 300035.0])


def test_nearest_10_to_20_m_takes_block_centre_pixel():
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = nearest({"scl": _grid(values, 10)}, 20)["scl"]
    assert out.shape == (2, 2)
    assert np.array_equal(out.values, [[5.0, 7.0], [13.0, 15.0]])


def test_non_integer_factor_raises():
    # block_mean only downsamples, so its non-integer case must be finer -> coarser (10 -> 15 m).
    with pytest.raises(ValueError, match="integer"):
        block_mean({"blue": _grid(np.ones((3, 3), dtype=np.float32), 10)}, 15)
    with pytest.raises(ValueError, match="integer"):
        nearest({"blue": _grid(np.ones((3, 3), dtype=np.float32), 30)}, 20)


def test_block_mean_requires_downsampling():
    da = _grid(np.ones((2, 2), dtype=np.float32), 20)
    with pytest.raises(ValueError, match="coarser"):
        block_mean({"blue": da}, 10)


def test_snap_bbox_outward_expands_off_lattice_bbox_to_60m_lattice():
    # minx/maxy fall inside 60 m pixels on the origin lattice; min edges floor, max edges ceil.
    snapped = snap_bbox_outward((300010.0, 4169850.0, 300130.0, 4169970.0), ORIGIN, 60)
    assert snapped == pytest.approx((300000.0, 4169820.0, 300180.0, 4170000.0))
    width_px = (snapped[2] - snapped[0]) / 60
    height_px = (snapped[3] - snapped[1]) / 60
    assert width_px == pytest.approx(3.0) and height_px == pytest.approx(3.0)


def test_snap_bbox_outward_leaves_on_lattice_bbox_unchanged():
    on_lattice = (300000.0, 4169880.0, 300120.0, 4170000.0)   # 2x2 pixels at 60 m
    assert snap_bbox_outward(on_lattice, ORIGIN, 60) == pytest.approx(on_lattice)


def test_snap_bbox_outward_ignores_float_noise_on_lattice_edges():
    # Edges sitting 1e-7 m outside the lattice (float noise) must not expand by a whole 60 m pixel.
    noisy = (300000.0 - 1e-7, 4169880.0 - 1e-7, 300120.0 + 1e-7, 4170000.0 + 1e-7)
    assert snap_bbox_outward(noisy, ORIGIN, 60) == pytest.approx((300000.0, 4169880.0, 300120.0, 4170000.0))


def test_snap_bbox_outward_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="min < max"):
        snap_bbox_outward((300120.0, 4169880.0, 300000.0, 4170000.0), ORIGIN, 60)
