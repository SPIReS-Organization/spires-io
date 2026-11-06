"""Harmonized Landsat Sentinel-2 (HLS) v2.0 L30 and S30 surface reflectance."""
from spires_io.hls.spec import L30_SPEC, S30_SPEC, open_hls_l30_scene, open_hls_s30_scene

__all__ = ["L30_SPEC", "S30_SPEC", "open_hls_l30_scene", "open_hls_s30_scene"]
