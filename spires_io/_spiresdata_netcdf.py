"""Shared NetCDF mechanics for serialized :class:`SpiresData` products."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from spires_contract import (
    SpiresData,
    validate_results,
    validate_spatial_alignment,
    validate_spires_data,
)

PRODUCT_TYPE = "SpiresData"
STORAGE_SCHEMA_VERSION = "1"
NETCDF_ENGINE = "netcdf4"

SCENE_GROUP = "scene"
BACKGROUND_GROUP = "background"
ANCILLARY_GROUP = "ancillary"
RESULTS_GROUP = "results"
REQUIRED_GROUPS = (SCENE_GROUP,)
OPTIONAL_GROUPS = (BACKGROUND_GROUP, ANCILLARY_GROUP, RESULTS_GROUP)

PRODUCT_TYPE_ATTR = "spires_product_type"
STORAGE_SCHEMA_ATTR = "spires_storage_schema_version"
PRESENT_GROUPS_ATTR = "spires_present_groups"
BACKGROUND_VARIABLE_ATTR = "spires_background_variable"

BACKGROUND_VARIABLE_DEFAULT = "background_reflectance"
JSON_ATTRIBUTE_PREFIX = "__spires_json_v1__:"
NETCDF_SUFFIXES = frozenset({".nc", ".cdf", ".netcdf"})


def validate_product_path(path: str | Path) -> Path:
    """Return a normalized NetCDF product path or raise an actionable error."""
    normalized = Path(path)
    if normalized.suffix.lower() not in NETCDF_SUFFIXES:
        allowed = ", ".join(sorted(NETCDF_SUFFIXES))
        raise ValueError(f"serialized SpiresData path must use one of: {allowed}")
    return normalized


def validate_product_data(data: SpiresData) -> None:
    """Validate the in-memory contracts required by a serialized product."""
    validate_spires_data(data)
    validate_spatial_alignment(data)
    if data.results is not None:
        validate_results(data.results, scene=data.scene)


def present_groups(data: SpiresData) -> tuple[str, ...]:
    """Return serialized group names in stable field order."""
    groups = [SCENE_GROUP]
    if data.background is not None:
        groups.append(BACKGROUND_GROUP)
    if data.ancillary is not None:
        groups.append(ANCILLARY_GROUP)
    if data.results is not None:
        groups.append(RESULTS_GROUP)
    return tuple(groups)


def background_to_dataset(background: xr.DataArray) -> tuple[xr.Dataset, str]:
    """Represent the optional background field as a named one-variable dataset."""
    variable_name = background.name or BACKGROUND_VARIABLE_DEFAULT
    if not isinstance(variable_name, str):
        variable_name = str(variable_name)
    return background.rename(variable_name).to_dataset(), variable_name


def prepare_dataset_for_netcdf(dataset: xr.Dataset) -> xr.Dataset:
    """Copy a dataset and make arbitrary metadata reversible and NetCDF-safe."""
    prepared = dataset.copy(deep=False)
    prepared.attrs = _encode_attrs(dataset.attrs)
    for name in prepared.variables:
        prepared[name].attrs = _encode_attrs(dataset[name].attrs)
        # Source-file backend encodings should not silently control the SPIReS
        # product representation.
        prepared[name].encoding = {}
    return prepared


def restore_dataset_attrs(dataset: xr.Dataset) -> xr.Dataset:
    """Restore metadata encoded by :func:`prepare_dataset_for_netcdf`."""
    restored = dataset.copy(deep=False)
    restored.attrs = _decode_attrs(dataset.attrs)
    for name in restored.variables:
        restored[name].attrs = _decode_attrs(dataset[name].attrs)
        restored[name].encoding = {}
    return restored


def parse_present_groups(value: Any) -> tuple[str, ...]:
    """Parse and validate the root-level group inventory."""
    if not isinstance(value, str):
        raise ValueError(f"{PRESENT_GROUPS_ATTR!r} must be a space-separated string")
    groups = tuple(value.split())
    if not groups or groups[0] != SCENE_GROUP:
        raise ValueError("serialized SpiresData product must contain the scene group")
    unknown = tuple(
        group for group in groups if group not in REQUIRED_GROUPS + OPTIONAL_GROUPS
    )
    if unknown:
        raise ValueError(f"serialized SpiresData product lists unknown groups: {unknown}")
    if len(set(groups)) != len(groups):
        raise ValueError("serialized SpiresData product lists duplicate groups")
    return groups


def _encode_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): _encode_attr_value(value) for name, value in attrs.items()}


def _decode_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): _decode_attr_value(value) for name, value in attrs.items()}


def _encode_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic) and not isinstance(value, np.bool_):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value

    try:
        encoded = json.dumps(_json_compatible(value), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"attribute value of type {type(value).__name__} is not serializable"
        ) from exc
    return f"{JSON_ATTRIBUTE_PREFIX}{encoded}"


def _decode_attr_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(JSON_ATTRIBUTE_PREFIX):
        return json.loads(value[len(JSON_ATTRIBUTE_PREFIX) :])
    return value


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"unsupported JSON metadata type: {type(value).__name__}")
