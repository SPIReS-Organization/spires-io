"""Shared scalar policy constants for SPIReS I/O."""

EPS = 1e-6

MIN_ZENITH = 0.0
MAX_ZENITH = 90.0
MAX_ELEV = 8000.0
MIN_ELEV = -1000.0
MAX_SVF = 1.0
MAX_ASPECT = 360.0

# Topographic correction / atmosphere bounds retained for config validation.
MIN_MU_S = 0.25
MIN_AOD = 0.03
MAX_AOD = 1.0
MIN_H2O = 0.2
MAX_H2O = 5.0

STATIC_DATA = ("canopy_fraction", "dem", "slope", "aspect", "skyview")
