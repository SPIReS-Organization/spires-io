"""Band definitions and LUT helpers for MODIS surface reflectance workflows."""

import re
from pathlib import Path
from typing import Any


MODIS_DEFAULT_BAND_NAMES = ("1", "2", "3", "4", "5", "6", "7")
MODIS_ANALYSIS_BANDS = MODIS_DEFAULT_BAND_NAMES
MODIS_PRODUCT_TO_PLATFORM = {
    "MOD09GA": "terra",
    "MYD09GA": "aqua",
}
_MODIS_BAND_TOKEN_RE = re.compile(r"(?:^|_)([1-7])(?=_|$)")


def normalize_modis_band_names(bands: tuple[str, ...] | list[str]) -> list[str]:
    """Return MODIS band labels normalized to plain numeric strings."""
    normalized = []
    for band in bands:
        text = str(band).strip()
        if text.lower().startswith("b"):
            text = text[1:]
        normalized.append(text)
    invalid = [band for band in normalized if band not in MODIS_ANALYSIS_BANDS]
    if invalid:
        raise ValueError(f"Unsupported MODIS band(s): {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate MODIS band(s) requested: {normalized}")
    return normalized


def infer_modis_lut_band_names_from_path(path: str | Path) -> list[str] | None:
    """Infer the MODIS band list from a LUT filename."""
    stem = Path(path).stem
    bands = [match.group(1) for match in _MODIS_BAND_TOKEN_RE.finditer(stem)]
    if not bands:
        return None
    return normalize_modis_band_names(bands)


def infer_modis_lut_band_names_from_metadata(path: str | Path) -> list[str] | None:
    """Read ``SensorTableBandOrder`` from a MATLAB v7.3 / HDF5 LUT file when present."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None

    lut_path = Path(path)
    if not lut_path.exists():
        return None

    try:
        with h5py.File(lut_path, "r") as hdf:
            dataset = hdf.get("SensorTableBandOrder")
            if dataset is None:
                return None
            values = np.asarray(dataset[()]).reshape(-1)
    except OSError:
        return None

    if values.size == 0:
        return None

    bands = [str(int(value)) for value in values]
    try:
        return normalize_modis_band_names(bands)
    except ValueError:
        return None


def resolve_modis_inversion_bands(
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
) -> list[str]:
    """
    Resolve the MODIS inversion band order.

    For the first MODIS implementation we default to bands 1-7 and keep the
    optional ``lut_file`` argument for API symmetry with VIIRS.
    """
    resolved_bands, _ = resolve_modis_inversion_bands_with_source(bands=bands, lut_file=lut_file)
    return resolved_bands


def resolve_modis_inversion_bands_with_source(
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
) -> tuple[list[str], str]:
    """Resolve the MODIS band order for inversion prep and report its source."""
    if bands is not None:
        return normalize_modis_band_names(bands), "explicit"
    if lut_file is not None:
        metadata_bands = infer_modis_lut_band_names_from_metadata(lut_file)
        if metadata_bands is not None:
            return metadata_bands, "metadata"
        path_bands = infer_modis_lut_band_names_from_path(lut_file)
        if path_bands is not None:
            return path_bands, "filename"
    return list(MODIS_DEFAULT_BAND_NAMES), "default"
