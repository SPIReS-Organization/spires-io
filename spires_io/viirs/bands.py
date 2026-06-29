"""Band mappings for VIIRS surface reflectance products."""

import re
from pathlib import Path
from typing import Any

VIIRS_1KM_REFLECTANCE_BANDS = (
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M7",
    "M8",
    "M10",
    "M11",
)

VIIRS_500M_REFLECTANCE_BANDS = (
    "I1",
    "I2",
    "I3",
)

VIIRS_1KM_GEOMETRY_FIELDS = {
    "solar_zenith": "SolarZenith_1",
    "solar_azimuth": "SolarAzimuth_1",
    "sensor_zenith": "SensorZenith_1",
    "sensor_azimuth": "SensorAzimuth_1",
}

VIIRS_1KM_QA_FIELDS = {
    "qa_qf1": "SurfReflect_QF1_1",
    "qa_qf2": "SurfReflect_QF2_1",
    "qa_qf3": "SurfReflect_QF3_1",
    "qa_qf4": "SurfReflect_QF4_1",
    "qa_qf5": "SurfReflect_QF5_1",
    "qa_qf6": "SurfReflect_QF6_1",
    "qa_qf7": "SurfReflect_QF7_1",
    "land_water_mask": "land_water_mask_1",
    "num_observations_1km": "num_observations_1km",
    "obscov_1km": "obscov_1km_1",
    "orbit_pnt": "orbit_pnt_1",
}

VIIRS_500M_SUPPORT_FIELDS = {
    "iobs_res": "iobs_res_1",
    "num_observations_500m": "num_observations_500m",
    "obscov_500m": "obscov_500m_1",
}

VIIRS_ANALYSIS_BANDS = VIIRS_500M_REFLECTANCE_BANDS + VIIRS_1KM_REFLECTANCE_BANDS

_VIIRS_BAND_TOKEN_RE = re.compile(r"(?:^|_)((?:i|m)\d{1,2})(?=_|$)", re.IGNORECASE)


def reflectance_field_name(band_name: str) -> str:
    """Return the surface reflectance dataset name for a VIIRS band."""
    return f"SurfReflect_{band_name}_1"


def normalize_viirs_band_names(bands: tuple[str, ...] | list[str]) -> list[str]:
    """Normalize and validate a user- or LUT-provided VIIRS band list."""
    normalized = [band.upper() for band in bands]
    invalid = [band for band in normalized if band not in VIIRS_ANALYSIS_BANDS]
    if invalid:
        raise ValueError(f"Unsupported VIIRS band(s): {invalid}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate VIIRS band(s) requested: {normalized}")
    return normalized


def infer_viirs_lut_band_names_from_path(path: str | Path) -> list[str] | None:
    """
    Infer the VIIRS band list from a LUT filename.

    This supports LUT names such as
    ``lut_viirs_noaa20_i1_i2_i3_m2_m4_m8_m11_3um_dust_bandpass.mat``.
    """
    stem = Path(path).stem
    bands = [match.group(1).upper() for match in _VIIRS_BAND_TOKEN_RE.finditer(stem)]
    if not bands:
        return None
    return normalize_viirs_band_names(bands)


def _decode_hdf5_string_like(value: Any, hdf: Any | None = None) -> list[str]:
    """Best-effort decoding of MATLAB/HDF5 string-like objects into Python strings."""
    if value is None:
        return []

    if isinstance(value, bytes):
        return [value.decode("utf-8")]

    if isinstance(value, str):
        return [value]

    try:
        import numpy as np
    except ImportError:
        return []

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            flattened = value.reshape(-1).tolist()
            return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in flattened]
        if value.dtype.kind in {"i", "u"}:
            if value.ndim == 1:
                return ["".join(chr(int(item)) for item in value if int(item) != 0)]
            if value.ndim == 2:
                rows = []
                for idx in range(value.shape[1]):
                    rows.append("".join(chr(int(item)) for item in value[:, idx] if int(item) != 0))
                if any(rows):
                    return rows
                return ["".join(chr(int(item)) for item in value.reshape(-1) if int(item) != 0)]
        if value.dtype.kind == "O" and hdf is not None:
            decoded = []
            for ref in value.reshape(-1):
                if ref:
                    decoded.extend(_decode_hdf5_string_like(hdf[ref][()], hdf=hdf))
            return decoded

    return []


def infer_viirs_lut_band_names_from_metadata(path: str | Path) -> list[str] | None:
    """
    Try to read ``SensorTableBandOrder`` from a MATLAB v7.3 / HDF5 LUT file.

    Returns ``None`` when the metadata field, a compatible reader, or decodable
    VIIRS band strings are not available.
    """
    try:
        import h5py
    except ImportError:
        return None

    lut_path = Path(path)
    if not lut_path.exists():
        return None

    try:
        with h5py.File(lut_path, "r") as hdf:
            dataset = None

            def visitor(name: str, obj: Any) -> None:
                nonlocal dataset
                if dataset is not None:
                    return
                if isinstance(obj, h5py.Dataset) and Path(name).name == "SensorTableBandOrder":
                    dataset = obj

            hdf.visititems(visitor)
            if dataset is None:
                return None

            candidates = _decode_hdf5_string_like(dataset[()], hdf=hdf)
    except OSError:
        return None

    bands = []
    for candidate in candidates:
        bands.extend(match.group(1).upper() for match in _VIIRS_BAND_TOKEN_RE.finditer(candidate))

    if not bands:
        return None

    try:
        return normalize_viirs_band_names(bands)
    except ValueError:
        return None


def resolve_viirs_inversion_bands(
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
) -> list[str]:
    """Resolve the VIIRS band order for inversion prep."""
    resolved_bands, _ = resolve_viirs_inversion_bands_with_source(bands=bands, lut_file=lut_file)
    return resolved_bands


def resolve_viirs_inversion_bands_with_source(
    *,
    bands: tuple[str, ...] | list[str] | None = None,
    lut_file: str | Path | None = None,
) -> tuple[list[str], str]:
    """Resolve the VIIRS band order for inversion prep and report its source."""
    if bands is not None:
        return normalize_viirs_band_names(bands), "explicit"
    if lut_file is not None:
        metadata_bands = infer_viirs_lut_band_names_from_metadata(lut_file)
        if metadata_bands is not None:
            return metadata_bands, "metadata"
        path_bands = infer_viirs_lut_band_names_from_path(lut_file)
        if path_bands is not None:
            return path_bands, "filename"
    return list(VIIRS_ANALYSIS_BANDS), "default"


def partition_viirs_band_names(bands: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
    """Split VIIRS analysis bands into native 500 m and 1 km groups."""
    normalized = normalize_viirs_band_names(bands)
    bands_500m = [band for band in normalized if band in VIIRS_500M_REFLECTANCE_BANDS]
    bands_1km = [band for band in normalized if band in VIIRS_1KM_REFLECTANCE_BANDS]
    return bands_500m, bands_1km
