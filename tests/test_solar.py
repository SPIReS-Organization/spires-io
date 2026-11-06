# tests/test_solar.py
import numpy as np
import pytest

from spires_io.solar import solar_zenith_angle

MAMMOTH_LAT, MAMMOTH_LON = 37.643, -119.029


def test_equator_equinox_solar_noon_is_overhead():
    # NOAA solar calculator: 2026-03-20 equinox, solar noon at 0 E is 12:07:30 UTC.
    assert solar_zenith_angle("2026-03-20T12:07:30", 0.0, 0.0) == pytest.approx(0.0, abs=0.1)


def test_mammoth_december_solstice_noon():
    # NOAA solar calculator for 37.643 N, 119.029 W on 2026-12-21, solar noon 19:54 UTC.
    assert solar_zenith_angle("2026-12-21T19:54:00", MAMMOTH_LAT, MAMMOTH_LON) == pytest.approx(61.08, abs=0.1)


def test_night_gives_zenith_above_90():
    assert solar_zenith_angle("2026-12-21T08:00:00", MAMMOTH_LAT, MAMMOTH_LON) > 160.0


def test_broadcasts_over_arrays():
    lat = np.array([[0.0, 10.0], [20.0, 30.0]])
    lon = np.zeros_like(lat)
    sza = solar_zenith_angle("2026-03-20T12:07:30", lat, lon)
    assert sza.shape == (2, 2)
    assert np.allclose(sza, lat, atol=0.1)


def test_accepts_numpy_datetime64():
    t = np.datetime64("2026-12-21T19:54:00")
    assert solar_zenith_angle(t, MAMMOTH_LAT, MAMMOTH_LON) == pytest.approx(61.08, abs=0.1)
