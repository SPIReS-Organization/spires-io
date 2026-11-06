# spires_io/landsat/__init__.py
"""Landsat 8/9 Collection 2 Level-2 surface-reflectance scenes."""
from spires_io.landsat.spec import SPEC, open_landsat_scene

__all__ = ["SPEC", "open_landsat_scene"]
