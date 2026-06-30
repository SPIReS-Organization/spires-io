"""Small sensor metadata helpers for config validation."""

from dataclasses import dataclass

import numpy as np

from spires_io.modis.bands import MODIS_ANALYSIS_BANDS
from spires_io.viirs.bands import VIIRS_ANALYSIS_BANDS


MODIS_CENTER_WL = {
    "1": 645.0,
    "2": 858.5,
    "3": 469.0,
    "4": 555.0,
    "5": 1240.0,
    "6": 1640.0,
    "7": 2130.0,
}

VIIRS_CENTER_WL = {
    "M1": 415.0,
    "M2": 445.0,
    "M3": 490.0,
    "M4": 555.0,
    "I1": 640.0,
    "M5": 673.0,
    "I2": 865.0,
    "M7": 865.0,
    "M8": 1240.0,
    "I3": 1610.0,
    "M10": 1610.0,
    "M11": 2250.0,
}


@dataclass(frozen=True)
class SensorMetadata:
    resolution: float
    bands: np.ndarray
    wavelength: np.ndarray
    topographic_correction_versions: frozenset[int]


SENSOR_METADATA = {
    "modis": SensorMetadata(
        resolution=500.0,
        bands=np.array(MODIS_ANALYSIS_BANDS),
        wavelength=np.array([MODIS_CENTER_WL[band] for band in MODIS_ANALYSIS_BANDS]),
        topographic_correction_versions=frozenset({4, 5, 6}),
    ),
    "viirs": SensorMetadata(
        resolution=500.0,
        bands=np.array(VIIRS_ANALYSIS_BANDS),
        wavelength=np.array([VIIRS_CENTER_WL[band] for band in VIIRS_ANALYSIS_BANDS]),
        topographic_correction_versions=frozenset({1, 2}),
    ),
}


def get_sensor_metadata(sensor: str) -> SensorMetadata:
    """Return static metadata for a registered sensor family."""
    return SENSOR_METADATA[sensor]
