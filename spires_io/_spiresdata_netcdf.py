"""Shared NetCDF mechanics for serialized :class:`SpiresData` products."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from spires_contract import (
    COMPLETION_STATUS_COMPLETE,
    CONTENT_PROFILE_INVERSION_RAW,
    CONTENT_PROFILE_POSTPROCESSED_RAW,
    OPERATION_RESULT_VARIABLES,
    PERSISTED_ATTR_ACQUISITION_TIME,
    PERSISTED_ATTR_BACKGROUND_VARIABLE,
    PERSISTED_ATTR_COMPLETED_OPERATIONS,
    PERSISTED_ATTR_COMPLETED_STAGES,
    PERSISTED_ATTR_COMPLETION_STATUS,
    PERSISTED_ATTR_CONTENT_PROFILE,
    PERSISTED_ATTR_CREATED_AT,
    PERSISTED_ATTR_GRID_DIGEST,
    PERSISTED_ATTR_PACKAGE_VERSIONS,
    PERSISTED_ATTR_PLATFORM,
    PERSISTED_ATTR_PRESENT_GROUPS,
    PERSISTED_ATTR_PRODUCT,
    PERSISTED_ATTR_PRODUCT_CONTENTS,
    PERSISTED_ATTR_PRODUCT_TYPE,
    PERSISTED_ATTR_PROVENANCE,
    PERSISTED_ATTR_SCHEMA_VERSION,
    PERSISTED_ATTR_SENSOR,
    PERSISTED_ATTR_SPATIAL_ID,
    PERSISTED_ATTR_UPDATED_AT,
    PERSISTED_GROUP_ANCILLARY,
    PERSISTED_GROUP_BACKGROUND,
    PERSISTED_GROUP_RESULTS,
    PERSISTED_GROUP_SCENE,
    PERSISTED_GROUPS,
    PERSISTED_PRODUCT_TYPE,
    PERSISTED_SCHEMA_VERSION,
    PRODUCT_CONTENTS_FULL,
    PRODUCT_CONTENTS_RESULTS_SUBSET,
    PersistedProductMetadata,
    ProductIdentity,
    SpiresData,
    validate_persisted_data,
    validate_persisted_grid,
    validate_persisted_metadata,
)
from spires_contract import conventions as contract

PRODUCT_TYPE = PERSISTED_PRODUCT_TYPE
STORAGE_SCHEMA_VERSION = PERSISTED_SCHEMA_VERSION
NETCDF_ENGINE = "netcdf4"

SCENE_GROUP = PERSISTED_GROUP_SCENE
BACKGROUND_GROUP = PERSISTED_GROUP_BACKGROUND
ANCILLARY_GROUP = PERSISTED_GROUP_ANCILLARY
RESULTS_GROUP = PERSISTED_GROUP_RESULTS
REQUIRED_GROUPS = (SCENE_GROUP,)
OPTIONAL_GROUPS = (BACKGROUND_GROUP, ANCILLARY_GROUP, RESULTS_GROUP)

PRODUCT_TYPE_ATTR = PERSISTED_ATTR_PRODUCT_TYPE
STORAGE_SCHEMA_ATTR = PERSISTED_ATTR_SCHEMA_VERSION
PRESENT_GROUPS_ATTR = PERSISTED_ATTR_PRESENT_GROUPS
BACKGROUND_VARIABLE_ATTR = PERSISTED_ATTR_BACKGROUND_VARIABLE

BACKGROUND_VARIABLE_DEFAULT = "background_reflectance"
JSON_ATTRIBUTE_PREFIX = "__spires_json_v1__:"
NETCDF_SUFFIXES = frozenset({".nc", ".cdf", ".netcdf"})
DEFAULT_COMPRESSION_LEVEL = 4
DEFAULT_SPATIAL_CHUNK = 512


def validate_product_path(path: str | Path) -> Path:
    """Return a normalized NetCDF product path or raise an actionable error."""
    normalized = Path(path)
    if normalized.suffix.lower() not in NETCDF_SUFFIXES:
        allowed = ", ".join(sorted(NETCDF_SUFFIXES))
        raise ValueError(f"serialized SpiresData path must use one of: {allowed}")
    return normalized


def validate_product_data(
    data: SpiresData,
    metadata: PersistedProductMetadata,
) -> None:
    """Validate the in-memory contracts required by a serialized product."""
    validate_persisted_data(data, metadata)


def product_data_for_contents(
    data: SpiresData,
    product_contents: str,
) -> SpiresData:
    """Return the complete or compact object represented by one product."""
    if product_contents == PRODUCT_CONTENTS_FULL:
        return data
    if product_contents != PRODUCT_CONTENTS_RESULTS_SUBSET:
        raise ValueError(
            "product_contents must be 'full' or 'results_subset', "
            f"got {product_contents!r}"
        )

    retained_names = tuple(
        name
        for name in (*contract.INVERSION_EXCLUSION_VARIABLES, "spatial_ref")
        if name in data.scene.variables
    )
    scene = data.scene[list(retained_names)].copy(deep=False)
    scene.attrs = dict(data.scene.attrs)
    results = None if data.results is None else data.results.copy(deep=False)
    return SpiresData(scene=scene, results=results)


def build_product_metadata(
    data: SpiresData,
    *,
    identity: ProductIdentity,
    content_profile: str,
    product_contents: str,
    completed_operations: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
    package_versions: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> PersistedProductMetadata:
    """Construct validated metadata for a new or replacement product."""
    now = utc_now()
    if content_profile == CONTENT_PROFILE_INVERSION_RAW:
        completed_stages = ("invert",)
    elif content_profile == CONTENT_PROFILE_POSTPROCESSED_RAW:
        completed_stages = ("invert", "albedo")
    else:
        raise ValueError(f"unsupported content profile {content_profile!r}")

    selected_operations = set(completed_operations)
    canonical_operations = tuple(
        operation
        for operation in OPERATION_RESULT_VARIABLES
        if operation in selected_operations
    )
    if len(canonical_operations) != len(selected_operations):
        unknown = sorted(selected_operations - set(OPERATION_RESULT_VARIABLES))
        raise ValueError(f"unknown completed operation(s): {unknown}")

    versions = (
        {}
        if package_versions is None
        else {
            str(name): str(value)
            for name, value in package_versions.items()
        }
    )
    versions.update({
        "spires-contract": _distribution_version("spires-contract"),
        "spires-io": _distribution_version("spires-io"),
    })

    metadata = PersistedProductMetadata(
        identity=identity,
        content_profile=content_profile,
        product_contents=product_contents,
        completed_stages=completed_stages,
        completed_operations=canonical_operations,
        completion_status=COMPLETION_STATUS_COMPLETE,
        grid_digest=spatial_grid_digest(data.scene),
        created_at=created_at or now,
        updated_at=now,
        package_versions=versions,
        provenance={} if provenance is None else dict(provenance),
    )
    validate_persisted_metadata(metadata)
    return metadata


def metadata_to_root_attrs(
    metadata: PersistedProductMetadata,
    *,
    groups: tuple[str, ...],
    background_variable: str | None,
) -> dict[str, Any]:
    """Encode contract metadata as NetCDF-safe root attributes."""
    validate_persisted_metadata(metadata)
    attrs: dict[str, Any] = {
        PERSISTED_ATTR_PRODUCT_TYPE: PERSISTED_PRODUCT_TYPE,
        PERSISTED_ATTR_SCHEMA_VERSION: PERSISTED_SCHEMA_VERSION,
        PERSISTED_ATTR_PRESENT_GROUPS: " ".join(groups),
        PERSISTED_ATTR_CONTENT_PROFILE: metadata.content_profile,
        PERSISTED_ATTR_PRODUCT_CONTENTS: metadata.product_contents,
        PERSISTED_ATTR_COMPLETION_STATUS: metadata.completion_status,
        PERSISTED_ATTR_COMPLETED_STAGES: canonical_json(metadata.completed_stages),
        PERSISTED_ATTR_COMPLETED_OPERATIONS: canonical_json(
            metadata.completed_operations
        ),
        PERSISTED_ATTR_SENSOR: metadata.identity.sensor,
        PERSISTED_ATTR_PRODUCT: metadata.identity.product,
        PERSISTED_ATTR_SPATIAL_ID: metadata.identity.spatial_id,
        PERSISTED_ATTR_ACQUISITION_TIME: metadata.identity.acquisition_time,
        PERSISTED_ATTR_GRID_DIGEST: metadata.grid_digest,
        PERSISTED_ATTR_CREATED_AT: metadata.created_at,
        PERSISTED_ATTR_UPDATED_AT: metadata.updated_at,
        PERSISTED_ATTR_PACKAGE_VERSIONS: canonical_json(metadata.package_versions),
        PERSISTED_ATTR_PROVENANCE: canonical_json(metadata.provenance),
    }
    if metadata.identity.platform is not None:
        attrs[PERSISTED_ATTR_PLATFORM] = metadata.identity.platform
    if background_variable is not None:
        attrs[PERSISTED_ATTR_BACKGROUND_VARIABLE] = background_variable
    return attrs


def metadata_from_root_attrs(attrs: Mapping[str, Any]) -> PersistedProductMetadata:
    """Decode and validate contract metadata from root attributes."""
    if attrs.get(PERSISTED_ATTR_PRODUCT_TYPE) != PERSISTED_PRODUCT_TYPE:
        raise ValueError(
            f"{PERSISTED_ATTR_PRODUCT_TYPE!r} is missing or invalid"
        )
    try:
        schema_version = int(attrs[PERSISTED_ATTR_SCHEMA_VERSION])
        completed_stages = _json_string_tuple(
            attrs[PERSISTED_ATTR_COMPLETED_STAGES],
            PERSISTED_ATTR_COMPLETED_STAGES,
        )
        completed_operations = _json_string_tuple(
            attrs[PERSISTED_ATTR_COMPLETED_OPERATIONS],
            PERSISTED_ATTR_COMPLETED_OPERATIONS,
        )
        package_versions = _json_mapping(
            attrs[PERSISTED_ATTR_PACKAGE_VERSIONS],
            PERSISTED_ATTR_PACKAGE_VERSIONS,
        )
        provenance = _json_mapping(
            attrs[PERSISTED_ATTR_PROVENANCE],
            PERSISTED_ATTR_PROVENANCE,
        )
        identity = ProductIdentity(
            sensor=_required_string(attrs, PERSISTED_ATTR_SENSOR),
            product=_required_string(attrs, PERSISTED_ATTR_PRODUCT),
            platform=_optional_string(attrs.get(PERSISTED_ATTR_PLATFORM)),
            spatial_id=_required_string(attrs, PERSISTED_ATTR_SPATIAL_ID),
            acquisition_time=_required_string(
                attrs,
                PERSISTED_ATTR_ACQUISITION_TIME,
            ),
        )
        metadata = PersistedProductMetadata(
            identity=identity,
            content_profile=_required_string(
                attrs,
                PERSISTED_ATTR_CONTENT_PROFILE,
            ),
            product_contents=_required_string(
                attrs,
                PERSISTED_ATTR_PRODUCT_CONTENTS,
            ),
            completed_stages=completed_stages,
            completed_operations=completed_operations,
            completion_status=_required_string(
                attrs,
                PERSISTED_ATTR_COMPLETION_STATUS,
            ),
            grid_digest=_required_string(attrs, PERSISTED_ATTR_GRID_DIGEST),
            created_at=_required_string(attrs, PERSISTED_ATTR_CREATED_AT),
            updated_at=_required_string(attrs, PERSISTED_ATTR_UPDATED_AT),
            package_versions=package_versions,
            provenance=provenance,
            schema_version=schema_version,
        )
    except KeyError as exc:
        raise ValueError(f"missing persisted-product attribute {exc.args[0]!r}") from exc
    validate_persisted_metadata(metadata)
    return metadata


def spatial_grid_digest(scene: xr.Dataset) -> str:
    """Return a stable digest of exact x/y coordinates and CRS identity."""
    validate_persisted_grid(scene)
    digest = hashlib.sha256()
    for name in contract.SPATIAL_DIMS:
        values = np.asarray(scene.coords[name].values)
        digest.update(name.encode("utf-8"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(canonical_json(values.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(values).tobytes())
    spatial_ref = scene["spatial_ref"]
    crs = (
        spatial_ref.attrs.get("crs_wkt")
        or spatial_ref.attrs.get("spatial_ref")
        or scene.attrs.get("crs_wkt")
    )
    digest.update(str(crs).encode("utf-8"))
    return digest.hexdigest()


def canonical_netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Return the schema-v1 lossless compression policy for data variables."""
    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in dataset.data_vars.items():
        if variable.ndim < 1:
            continue
        chunksizes = tuple(
            max(
                1,
                min(size, DEFAULT_SPATIAL_CHUNK)
                if dim in contract.SPATIAL_DIMS
                else size,
            )
            for dim, size in zip(variable.dims, variable.shape)
        )
        encoding[name] = {
            "zlib": True,
            "complevel": DEFAULT_COMPRESSION_LEVEL,
            "shuffle": True,
            "chunksizes": chunksizes,
        }
    return encoding


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible metadata deterministically."""
    return json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        group for group in groups if group not in PERSISTED_GROUPS
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
    if isinstance(value, str) and not value.startswith(JSON_ATTRIBUTE_PREFIX):
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


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _required_string(attrs: Mapping[str, Any], name: str) -> str:
    value = attrs[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"persisted-product attribute {name!r} must be nonempty")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional persisted-product string attribute must be nonempty")
    return value


def _json_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name!r} must contain a JSON array") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"{name!r} must contain a JSON string array")
    return tuple(parsed)


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name!r} must contain a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name!r} must contain a JSON object")
    return parsed
