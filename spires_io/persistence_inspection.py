"""Lightweight inspection and completion validation for SPIReS products."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from netCDF4 import Dataset as NetCDFDataset
import numpy as np
import xarray as xr
from spires_contract import (
    ALBEDO_RESULT_VARIABLES,
    CONTENT_PROFILE_POSTPROCESSED_RAW,
    DERIVED_RESULT_LONG_NAMES,
    DERIVED_RESULT_UNITS,
    OPERATION_RESULT_VARIABLES,
    PERSISTED_ATTR_BACKGROUND_VARIABLE,
    PERSISTED_ATTR_PRESENT_GROUPS,
    PERSISTED_GROUP_ANCILLARY,
    PERSISTED_GROUP_BACKGROUND,
    PERSISTED_GROUP_RESULTS,
    PERSISTED_GROUP_SCENE,
    PRODUCT_CONTENTS_FULL,
    PRODUCT_CONTENTS_RESULTS_SUBSET,
    ContractError,
    PersistedProductMetadata,
    ProductIdentity,
)
from spires_contract import conventions as contract

from spires_io._spiresdata_netcdf import (
    NETCDF_ENGINE,
    metadata_from_root_attrs,
    parse_present_groups,
    spatial_grid_digest,
    validate_product_path,
)

__all__ = [
    "PersistedProductInspection",
    "inspect_spires_product",
    "validate_spires_product",
]


@dataclass(frozen=True)
class PersistedProductInspection:
    """Header-derived product state with actionable validation issues."""

    path: Path
    readable: bool
    metadata: PersistedProductMetadata | None = None
    declared_groups: tuple[str, ...] = ()
    actual_groups: tuple[str, ...] = ()
    group_variables: Mapping[str, tuple[str, ...]] | None = None
    issues: tuple[str, ...] = ()
    validation: str = "metadata"

    @property
    def complete(self) -> bool:
        return self.readable and self.metadata is not None and not self.issues


def inspect_spires_product(
    path: str | Path,
    *,
    expected_identity: ProductIdentity | None = None,
    expected_profile: str | None = None,
    expected_contents: str | None = None,
) -> PersistedProductInspection:
    """Inspect headers, groups, schema, identity, and required variables."""
    product_path = Path(path)
    issues: list[str] = []
    try:
        product_path = validate_product_path(product_path)
    except (TypeError, ValueError) as exc:
        return PersistedProductInspection(
            path=product_path,
            readable=False,
            issues=(str(exc),),
        )

    metadata = None
    declared_groups: tuple[str, ...] = ()
    actual_groups: tuple[str, ...] = ()
    group_variables: dict[str, tuple[str, ...]] = {}
    try:
        with NetCDFDataset(product_path, mode="r") as root:
            attrs = {
                name: _plain_netcdf_value(root.getncattr(name))
                for name in root.ncattrs()
            }
            try:
                metadata = metadata_from_root_attrs(attrs)
            except (TypeError, ValueError, ContractError) as exc:
                issues.append(f"invalid persisted metadata: {exc}")

            try:
                declared_groups = parse_present_groups(
                    attrs[PERSISTED_ATTR_PRESENT_GROUPS]
                )
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"invalid declared group inventory: {exc}")

            actual_groups = tuple(root.groups)
            group_variables = {
                name: tuple(group.variables)
                for name, group in root.groups.items()
            }
            if declared_groups and declared_groups != actual_groups:
                issues.append(
                    "declared groups do not match physical groups: "
                    f"declared {declared_groups!r}, found {actual_groups!r}"
                )

            unknown_groups = tuple(
                group
                for group in actual_groups
                if group
                not in {
                    PERSISTED_GROUP_SCENE,
                    PERSISTED_GROUP_BACKGROUND,
                    PERSISTED_GROUP_ANCILLARY,
                    PERSISTED_GROUP_RESULTS,
                }
            )
            if unknown_groups:
                issues.append(f"product contains unknown groups {unknown_groups!r}")

            if metadata is not None:
                _inspect_expected_metadata(
                    metadata,
                    expected_identity=expected_identity,
                    expected_profile=expected_profile,
                    expected_contents=expected_contents,
                    issues=issues,
                )
                _inspect_group_contract(root, attrs, metadata, issues)
    except (OSError, RuntimeError, ValueError) as exc:
        return PersistedProductInspection(
            path=product_path,
            readable=False,
            issues=(f"cannot open NetCDF product: {exc}",),
        )

    if metadata is not None and PERSISTED_GROUP_SCENE in actual_groups:
        try:
            with xr.open_dataset(
                product_path,
                group=PERSISTED_GROUP_SCENE,
                engine=NETCDF_ENGINE,
            ) as scene:
                actual_digest = spatial_grid_digest(scene)
            if actual_digest != metadata.grid_digest:
                issues.append(
                    "spatial grid digest does not match persisted metadata"
                )
        except (OSError, ValueError, ContractError) as exc:
            issues.append(f"cannot validate persisted spatial grid: {exc}")

    return PersistedProductInspection(
        path=product_path,
        readable=True,
        metadata=metadata,
        declared_groups=declared_groups,
        actual_groups=actual_groups,
        group_variables=group_variables,
        issues=tuple(issues),
    )


def validate_spires_product(
    path: str | Path,
    *,
    expected_identity: ProductIdentity | None = None,
    expected_profile: str | None = None,
    expected_contents: str | None = None,
    validation: str = "sample",
) -> PersistedProductInspection:
    """Validate one product using metadata, sampled values, or a full read."""
    if validation not in {"metadata", "sample", "full"}:
        raise ValueError("validation must be 'metadata', 'sample', or 'full'")

    inspection = inspect_spires_product(
        path,
        expected_identity=expected_identity,
        expected_profile=expected_profile,
        expected_contents=expected_contents,
    )
    issues = list(inspection.issues)
    if not issues and validation == "sample":
        issues.extend(_sample_value_issues(inspection))
    elif not issues and validation == "full":
        try:
            from spires_io.spiresdata_reader import read_spires_data

            read_spires_data(
                inspection.path,
                expected_identity=expected_identity,
                expected_profile=expected_profile,
                expected_contents=expected_contents,
            )
        except (OSError, TypeError, ValueError, ContractError) as exc:
            issues.append(f"full persisted-product validation failed: {exc}")

    completed = replace(
        inspection,
        issues=tuple(issues),
        validation=validation,
    )
    if completed.issues:
        bullets = "\n".join(f"  - {issue}" for issue in completed.issues)
        raise ContractError(f"persisted_product contract violated:\n{bullets}")
    return completed


def _inspect_expected_metadata(
    metadata,
    *,
    expected_identity,
    expected_profile,
    expected_contents,
    issues,
):
    if expected_identity is not None and metadata.identity != expected_identity:
        issues.append(
            "product identity does not match expected identity: "
            f"found {metadata.identity!r}, expected {expected_identity!r}"
        )
    if (
        expected_profile is not None
        and metadata.content_profile != expected_profile
    ):
        issues.append(
            f"content profile is {metadata.content_profile!r}, "
            f"expected {expected_profile!r}"
        )
    if (
        expected_contents is not None
        and metadata.product_contents != expected_contents
    ):
        issues.append(
            f"product contents are {metadata.product_contents!r}, "
            f"expected {expected_contents!r}"
        )


def _inspect_group_contract(root, attrs, metadata, issues):
    required_groups = {PERSISTED_GROUP_SCENE, PERSISTED_GROUP_RESULTS}
    if metadata.product_contents == PRODUCT_CONTENTS_FULL:
        required_groups.add(PERSISTED_GROUP_BACKGROUND)
    missing_groups = tuple(
        group for group in required_groups if group not in root.groups
    )
    if missing_groups:
        issues.append(f"product is missing required groups {missing_groups!r}")
        return

    if metadata.product_contents == PRODUCT_CONTENTS_RESULTS_SUBSET:
        forbidden = tuple(
            group
            for group in (PERSISTED_GROUP_BACKGROUND, PERSISTED_GROUP_ANCILLARY)
            if group in root.groups
        )
        if forbidden:
            issues.append(
                f"results_subset product contains forbidden groups {forbidden!r}"
            )

    scene = root.groups[PERSISTED_GROUP_SCENE]
    _require_variable(scene, "x", ("x",), None, issues)
    _require_variable(scene, "y", ("y",), None, issues)
    _require_variable(scene, "spatial_ref", (), None, issues)
    _require_variable(
        scene,
        contract.INVERSION_EXCLUSION_FLAGS_VARIABLE,
        contract.SPATIAL_DIMS,
        np.uint16,
        issues,
    )
    _require_variable(
        scene,
        contract.INVERSION_EXCLUSION_ASSESSED_VARIABLE,
        contract.SPATIAL_DIMS,
        np.uint16,
        issues,
    )
    _require_variable(
        scene,
        contract.VALID_INVERSION_MASK_VARIABLE,
        contract.SPATIAL_DIMS,
        None,
        issues,
    )

    if metadata.product_contents == PRODUCT_CONTENTS_FULL:
        _require_variable(
            scene,
            "reflectance",
            contract.SPECTRA_DIMS,
            np.float32,
            issues,
        )
        _require_variable(
            scene,
            "solar_zenith",
            contract.SPATIAL_DIMS,
            np.float32,
            issues,
        )
        background = root.groups[PERSISTED_GROUP_BACKGROUND]
        background_name = attrs.get(PERSISTED_ATTR_BACKGROUND_VARIABLE)
        if not isinstance(background_name, str) or not background_name:
            issues.append("full product is missing its background variable name")
        else:
            _require_variable(
                background,
                background_name,
                contract.SPECTRA_DIMS,
                np.float32,
                issues,
            )
    else:
        allowed = {
            "x",
            "y",
            "spatial_ref",
            *contract.INVERSION_EXCLUSION_VARIABLES,
        }
        extra = tuple(name for name in scene.variables if name not in allowed)
        if extra:
            issues.append(
                f"results_subset scene contains nonessential variables {extra!r}"
            )

    results = root.groups[PERSISTED_GROUP_RESULTS]
    for name in contract.RESULT_VARIABLES:
        _require_variable(
            results,
            name,
            contract.RESULT_DIMS,
            np.float32,
            issues,
        )
    if metadata.content_profile == CONTENT_PROFILE_POSTPROCESSED_RAW:
        for operation in metadata.completed_operations:
            for name in OPERATION_RESULT_VARIABLES[operation]:
                variable = _require_variable(
                    results,
                    name,
                    contract.RESULT_DIMS,
                    np.float32,
                    issues,
                )
                if variable is None:
                    continue
                units = (
                    variable.getncattr("units")
                    if "units" in variable.ncattrs()
                    else None
                )
                if units != DERIVED_RESULT_UNITS[name]:
                    issues.append(
                        f"results.{name} has invalid or missing units"
                    )
                long_name = (
                    variable.getncattr("long_name")
                    if "long_name" in variable.ncattrs()
                    else None
                )
                if long_name != DERIVED_RESULT_LONG_NAMES[name]:
                    issues.append(
                        f"results.{name} has invalid or missing long_name"
                    )


def _require_variable(group, name, expected_dims, expected_dtype, issues):
    if name not in group.variables:
        issues.append(f"group {group.path!r} is missing variable {name!r}")
        return None
    variable = group.variables[name]
    if tuple(variable.dimensions) != tuple(expected_dims):
        issues.append(
            f"{group.path}.{name} dimensions are {tuple(variable.dimensions)!r}, "
            f"expected {tuple(expected_dims)!r}"
        )
    if (
        expected_dtype is not None
        and np.dtype(variable.dtype) != np.dtype(expected_dtype)
    ):
        issues.append(
            f"{group.path}.{name} dtype is {np.dtype(variable.dtype)}, "
            f"expected {np.dtype(expected_dtype)}"
        )
    return variable


def _sample_value_issues(inspection):
    issues = []
    metadata = inspection.metadata
    if metadata is None:
        return issues
    with NetCDFDataset(inspection.path, mode="r") as root:
        scene = root.groups[PERSISTED_GROUP_SCENE]
        flags = _sample_variable(
            scene.variables[contract.INVERSION_EXCLUSION_FLAGS_VARIABLE]
        ).astype(np.uint16)
        assessed = _sample_variable(
            scene.variables[contract.INVERSION_EXCLUSION_ASSESSED_VARIABLE]
        ).astype(np.uint16)
        valid = _sample_variable(
            scene.variables[contract.VALID_INVERSION_MASK_VARIABLE]
        ).astype(bool)
        known_mask = np.uint16(contract.INVERSION_EXCLUSION_KNOWN_MASK)
        unknown_mask = np.uint16(
            np.iinfo(np.uint16).max
            ^ contract.INVERSION_EXCLUSION_KNOWN_MASK
        )
        if np.any(flags & unknown_mask):
            issues.append("sampled QA flags contain reserved or unknown bits")
        if np.any(assessed & unknown_mask):
            issues.append("sampled QA assessed flags contain reserved or unknown bits")
        if np.any(flags & np.bitwise_and(known_mask, np.bitwise_not(assessed))):
            issues.append("sampled QA contains exclusion bits that were not assessed")
        if not np.array_equal(valid, flags == 0):
            issues.append("sampled valid-inversion mask disagrees with QA flags")

        results = root.groups[PERSISTED_GROUP_RESULTS]
        names = list(contract.RESULT_VARIABLES)
        for operation in metadata.completed_operations:
            names.extend(OPERATION_RESULT_VARIABLES[operation])
        for name in names:
            values = np.asarray(_sample_variable(results.variables[name]))
            if np.issubdtype(values.dtype, np.floating) and np.any(np.isinf(values)):
                issues.append(f"sampled results.{name} contains infinite values")
            finite = values[np.isfinite(values)]
            if name in {"fsnow", "fshade", "canopy_adjusted_fsnow", "ice_adjusted_fsnow", *ALBEDO_RESULT_VARIABLES}:
                if np.any((finite < 0) | (finite > 1)):
                    issues.append(
                        f"sampled results.{name} contains values outside [0, 1]"
                    )
            elif name in {"lap_concentration", "grain_radius", "delta_vis"}:
                if np.any(finite < 0):
                    issues.append(
                        f"sampled results.{name} contains negative values"
                    )
    return issues


def _sample_variable(variable):
    index = []
    for size in variable.shape:
        if size <= 3:
            index.append(slice(None))
        else:
            index.append(sorted({0, size // 2, size - 1}))
    return variable[tuple(index)]


def _plain_netcdf_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
