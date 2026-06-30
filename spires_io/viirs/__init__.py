"""VIIRS surface reflectance readers and I/O helpers."""

from spires_io.viirs.bands import (
    VIIRS_1KM_REFLECTANCE_BANDS,
    VIIRS_500M_REFLECTANCE_BANDS,
    VIIRS_ANALYSIS_BANDS,
    infer_viirs_lut_band_names_from_metadata,
    infer_viirs_lut_band_names_from_path,
    normalize_viirs_band_names,
    partition_viirs_band_names,
    reflectance_field_name,
    resolve_viirs_inversion_bands,
    resolve_viirs_inversion_bands_with_source,
)
from spires_io.viirs.geospatial import VIIRS_SINUSOIDAL_CRS
from spires_io.viirs.load_surface_reflectance import (
    PLATFORM_BY_PRODUCT,
    open_viirs_surface_reflectance,
    parse_viirs_surface_reflectance_filename,
    prepare_viirs_scene_for_inversion,
)
from spires_io.viirs.qa import decode_viirs_qa_masks, load_external_cloud_masks

__all__ = [
    "PLATFORM_BY_PRODUCT",
    "VIIRS_1KM_REFLECTANCE_BANDS",
    "VIIRS_500M_REFLECTANCE_BANDS",
    "VIIRS_ANALYSIS_BANDS",
    "VIIRS_SINUSOIDAL_CRS",
    "decode_viirs_qa_masks",
    "infer_viirs_lut_band_names_from_metadata",
    "infer_viirs_lut_band_names_from_path",
    "load_external_cloud_masks",
    "normalize_viirs_band_names",
    "open_viirs_surface_reflectance",
    "parse_viirs_surface_reflectance_filename",
    "partition_viirs_band_names",
    "prepare_viirs_scene_for_inversion",
    "reflectance_field_name",
    "resolve_viirs_inversion_bands",
    "resolve_viirs_inversion_bands_with_source",
]
