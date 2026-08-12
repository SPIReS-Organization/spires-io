"""Reusable helpers for sensor-specific product readers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SceneMetadata:
    """Minimal normalized scene metadata shared across readers."""

    product: str
    platform: str
    tile: str
    acquisition_date: str
    collection: str
    processing_timestamp: str
    source_path: str


def _decode_attr(value: Any) -> Any:
    """Decode HDF5 attribute values into plain Python objects where possible."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_attr(value.item())
        if value.dtype == object:
            return [_decode_attr(item) for item in value.tolist()]
        return np.array([_decode_attr(item) for item in value.tolist()])
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
        return value
    return repr(value)


def collect_attrs(dataset: Any) -> dict[str, Any]:
    """Return dataset attributes as a plain dictionary."""
    if hasattr(dataset, "attrs"):
        return {key: _decode_attr(value) for key, value in dataset.attrs.items()}
    if hasattr(dataset, "ncattrs"):
        return {key: _decode_attr(dataset.getncattr(key)) for key in dataset.ncattrs()}
    return {}


def collect_decoded_attrs(
    dataset: Any,
    *,
    apply_scale: bool,
    mask_fill: bool,
    dtype: np.dtype = np.float32,
) -> dict[str, Any]:
    """Return metadata consistent with an array decoded by this module.

    Source packing attributes describe the stored integer representation and
    must not remain after values have been converted to physical units. Convert
    any valid range to the physical domain and remove source fill metadata
    after fill values become NaN. The source product remains the authoritative
    record of its original packing.
    """
    attrs = collect_attrs(dataset)
    for name in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME"):
        attrs.pop(name, None)

    if apply_scale:
        scale_factor = attrs.pop("scale_factor", 1.0)
        add_offset = attrs.pop("add_offset", 0.0)

        if "valid_range" in attrs:
            physical_valid_range = np.asarray(attrs["valid_range"]).astype(dtype)
            physical_valid_range *= np.asarray(scale_factor, dtype=dtype)
            physical_valid_range += np.asarray(add_offset, dtype=dtype)
            attrs["valid_range"] = np.sort(physical_valid_range)

    if mask_fill:
        for name in ("_FillValue", "missing_value"):
            attrs.pop(name, None)

    for name in (
        "source_scale_factor",
        "source_add_offset",
        "source_valid_range",
        "source_fill_value",
        "source_missing_value",
    ):
        attrs.pop(name, None)

    return attrs


def read_scaled_array(
    dataset: Any,
    *,
    apply_scale: bool = True,
    mask_fill: bool = True,
    mask_valid_range: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Read an HDF dataset into a NumPy array and apply common scale/mask rules.

    Parameters
    ----------
    dataset
        HDF5 dataset object.
    apply_scale
        Apply ``scale_factor`` and ``add_offset`` if present.
    mask_fill
        Replace ``_FillValue`` values with ``NaN``.
    mask_valid_range
        Mask values outside ``valid_range`` when present.
    dtype
        Floating dtype to use for scaled arrays.
    """
    array = dataset[...]

    if not (apply_scale or mask_fill or mask_valid_range):
        return np.array(array)

    array = array.astype(dtype, copy=False)
    attrs = collect_attrs(dataset)

    fill_value = attrs.get("_FillValue")
    if mask_fill and fill_value is not None:
        array[array == fill_value] = np.nan

    if mask_valid_range and "valid_range" in attrs:
        valid_range = np.asarray(attrs["valid_range"])
        if valid_range.size == 2:
            array[(array < valid_range[0]) | (array > valid_range[1])] = np.nan

    if apply_scale:
        scale_factor = attrs.get("scale_factor", 1.0)
        add_offset = attrs.get("add_offset", 0.0)
        array = array * scale_factor + add_offset

    return array


def normalize_path(path: str | Path) -> Path:
    """Return an expanded, absolute path."""
    return Path(path).expanduser().resolve()
