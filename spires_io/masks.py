"""External masks and inversion-exclusion packing for SPIReS scenes."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  # register xarray .rio accessor
import xarray as xr

from spires_contract import (
    SpiresData,
    inversion_exclusion_metadata,
    validate_inversion_exclusion,
    validate_spires_data,
)
from spires_contract import conventions as contract

from spires_io.file_types import RASTER_SUFFIXES, XARRAY_SUFFIXES, ZARR_SUFFIXES


def pack_inversion_exclusions(
    exclusions: Mapping[str, xr.DataArray],
    *,
    assessed: Mapping[str, xr.DataArray] | None = None,
    reference: xr.DataArray | None = None,
) -> xr.Dataset:
    """Pack canonical Boolean exclusion reasons and assessed states."""
    exclusions = dict(exclusions)
    assessed = {} if assessed is None else dict(assessed)
    _check_reason_names(exclusions, label="exclusions")
    _check_reason_names(assessed, label="assessed")

    if reference is None:
        reference = next(
            iter((*exclusions.values(), *assessed.values())),
            None,
        )
    if reference is None:
        raise ValueError(
            "packing exclusions requires a spatial reference or at least one mask"
        )
    reference = _validated_spatial_reference(reference, "reference")

    shape = reference.shape
    flag_values = np.zeros(shape, dtype=contract.INVERSION_EXCLUSION_DTYPE)
    assessed_values = np.zeros(shape, dtype=contract.INVERSION_EXCLUSION_DTYPE)
    for reason in contract.INVERSION_EXCLUSION_REASONS:
        flag_mask = exclusions.get(reason)
        if flag_mask is None:
            flag = np.zeros(shape, dtype=bool)
        else:
            flag = np.asarray(
                _validated_boolean_mask(flag_mask, reference, reason).values,
                dtype=bool,
            )

        assessed_mask = assessed.get(reason)
        if assessed_mask is None:
            known = np.ones(shape, dtype=bool) if flag_mask is not None else np.zeros(
                shape, dtype=bool
            )
        else:
            known = np.asarray(
                _validated_boolean_mask(
                    assessed_mask,
                    reference,
                    f"{reason}_assessed",
                ).values,
                dtype=bool,
            )
        if np.any(flag & ~known):
            raise ValueError(
                f"exclusion reason {reason!r} is true where it was not assessed"
            )

        bit = np.asarray(
            contract.INVERSION_EXCLUSION_BITS[reason],
            dtype=contract.INVERSION_EXCLUSION_DTYPE,
        )
        flag_values[flag] |= bit
        assessed_values[known] |= bit

    coords = {dim: reference.coords[dim] for dim in contract.SPATIAL_DIMS}
    flags = xr.DataArray(
        flag_values,
        dims=contract.SPATIAL_DIMS,
        coords=coords,
        name=contract.INVERSION_EXCLUSION_FLAGS_VARIABLE,
        attrs=inversion_exclusion_metadata(
            contract.INVERSION_EXCLUSION_FLAGS_VARIABLE
        ),
    )
    assessed_flags = xr.DataArray(
        assessed_values,
        dims=contract.SPATIAL_DIMS,
        coords=coords,
        name=contract.INVERSION_EXCLUSION_ASSESSED_VARIABLE,
        attrs=inversion_exclusion_metadata(
            contract.INVERSION_EXCLUSION_ASSESSED_VARIABLE
        ),
    )
    valid = xr.DataArray(
        flag_values == 0,
        dims=contract.SPATIAL_DIMS,
        coords=coords,
        name=contract.VALID_INVERSION_MASK_VARIABLE,
    )
    packed = xr.Dataset(
        {
            contract.INVERSION_EXCLUSION_FLAGS_VARIABLE: flags,
            contract.INVERSION_EXCLUSION_ASSESSED_VARIABLE: assessed_flags,
            contract.VALID_INVERSION_MASK_VARIABLE: valid,
        }
    )
    validate_inversion_exclusion(packed)
    return packed


def decode_inversion_exclusions(
    packed: xr.Dataset,
) -> xr.Dataset:
    """Decode every canonical reason and per-pixel assessed state."""
    validate_inversion_exclusion(packed)
    flags = np.asarray(
        packed[contract.INVERSION_EXCLUSION_FLAGS_VARIABLE].values,
        dtype=contract.INVERSION_EXCLUSION_DTYPE,
    )
    assessed = np.asarray(
        packed[contract.INVERSION_EXCLUSION_ASSESSED_VARIABLE].values,
        dtype=contract.INVERSION_EXCLUSION_DTYPE,
    )
    reference = packed[contract.INVERSION_EXCLUSION_FLAGS_VARIABLE]
    coords = {dim: reference.coords[dim] for dim in contract.SPATIAL_DIMS}
    decoded: dict[str, xr.DataArray] = {}
    for reason in contract.INVERSION_EXCLUSION_REASONS:
        bit = contract.INVERSION_EXCLUSION_BITS[reason]
        decoded[f"mask_{reason}"] = xr.DataArray(
            (flags & bit) != 0,
            dims=contract.SPATIAL_DIMS,
            coords=coords,
        )
        decoded[f"mask_{reason}_assessed"] = xr.DataArray(
            (assessed & bit) != 0,
            dims=contract.SPATIAL_DIMS,
            coords=coords,
        )
    return xr.Dataset(decoded)


def assign_inversion_exclusion_masks(
    data: SpiresData,
    masks: Mapping[str, xr.DataArray],
) -> SpiresData:
    """Add external exclusions and return a replacement shared container."""
    if not isinstance(data, SpiresData):
        raise TypeError(f"data must be SpiresData, got {type(data).__name__}")
    validate_spires_data(data)
    if not masks:
        return data

    scene = data.scene
    packed_names = set(contract.INVERSION_EXCLUSION_VARIABLES)
    present = packed_names.intersection(scene.data_vars)
    if present and present != packed_names:
        raise ValueError(
            "scene contains a partial inversion-exclusion provenance set; "
            "expected all three packed variables or none"
        )

    reference = _scene_spatial_reference(scene)
    exclusions: dict[str, xr.DataArray] = {}
    assessed: dict[str, xr.DataArray] = {}
    if present:
        decoded = decode_inversion_exclusions(scene)
        for reason in contract.INVERSION_EXCLUSION_REASONS:
            exclusions[reason] = decoded[f"mask_{reason}"]
            assessed[reason] = decoded[f"mask_{reason}_assessed"]

    for name, mask in masks.items():
        reason = _external_mask_reason(name)
        normalized = _validated_boolean_mask(mask, reference, name)
        if reason in exclusions:
            exclusions[reason] = exclusions[reason] | normalized
            assessed[reason] = assessed[reason] | xr.ones_like(
                normalized, dtype=bool
            )
        else:
            exclusions[reason] = normalized
            assessed[reason] = xr.ones_like(normalized, dtype=bool)

    replacement = pack_inversion_exclusions(
        exclusions,
        assessed=assessed,
        reference=reference,
    )
    updated = scene.drop_vars(list(present), errors="ignore").copy(deep=False)
    updated.update(replacement)
    result = data.assign_scene(updated)
    validate_spires_data(result)
    return result


def load_external_mask(
    source: str | Path | xr.DataArray,
    *,
    target_scene: xr.Dataset,
    variable: str | None = None,
) -> xr.DataArray:
    """Load an external inversion-exclusion mask aligned to a prepared scene."""
    if isinstance(source, xr.DataArray):
        mask = source
    else:
        mask = _open_mask(Path(source), variable=variable)

    mask = _normalize_mask_dims(mask)
    mask = _align_mask_to_scene(mask, target_scene)
    mask = mask.astype(bool)
    mask.name = variable or "external_mask"
    return mask


def load_external_mask_on_grid(
    source: str | Path | xr.Dataset | xr.DataArray,
    *,
    target_x: xr.DataArray,
    target_y: xr.DataArray,
    variable: str | None = None,
    name: str = "external_mask",
) -> xr.DataArray:
    """Load an external mask and align it to target x/y grid coordinates."""
    close_source = None
    if isinstance(source, xr.DataArray):
        mask = source
    elif isinstance(source, xr.Dataset):
        mask = _mask_variable_from_dataset(source, variable=variable)
    else:
        mask = _open_mask(Path(source), variable=variable)
        close_source = mask

    try:
        mask = _normalize_mask_dims(mask)
        target = xr.DataArray(
            np.zeros((target_y.size, target_x.size), dtype=bool),
            dims=("y", "x"),
            coords={"y": target_y.values, "x": target_x.values},
        )
        if _coords_match(mask, target, ("y", "x")):
            aligned = mask
        elif mask.sizes["y"] == target.sizes["y"] and mask.sizes["x"] == target.sizes["x"]:
            aligned = mask.assign_coords(y=target_y.values, x=target_x.values)
        else:
            reprojected = _try_reproject_match(mask, target)
            if reprojected is None or not _coords_match(reprojected, target, ("y", "x")):
                raise ValueError("external mask cannot be aligned to the target grid")
            aligned = reprojected

        aligned = aligned.astype(bool).load()
        aligned.name = name
        return aligned
    finally:
        if close_source is not None:
            close_source.close()


def _check_reason_names(masks: Mapping[str, xr.DataArray], *, label: str) -> None:
    unknown = sorted(set(masks) - set(contract.INVERSION_EXCLUSION_REASONS))
    if unknown:
        raise ValueError(
            f"{label} contains unknown inversion-exclusion reason(s): {unknown}"
        )


def _validated_spatial_reference(
    reference: xr.DataArray,
    label: str,
) -> xr.DataArray:
    if not isinstance(reference, xr.DataArray):
        raise TypeError(f"{label} must be an xarray.DataArray")
    if tuple(reference.dims) != contract.SPATIAL_DIMS:
        raise ValueError(f"{label} must have dims {contract.SPATIAL_DIMS!r}")
    missing = [dim for dim in contract.SPATIAL_DIMS if dim not in reference.coords]
    if missing:
        raise ValueError(f"{label} is missing coordinate(s) {missing!r}")
    return reference


def _validated_boolean_mask(
    mask: xr.DataArray,
    reference: xr.DataArray,
    label: str,
) -> xr.DataArray:
    mask = _validated_spatial_reference(mask, label)
    reference = _validated_spatial_reference(reference, "reference")
    for dim in contract.SPATIAL_DIMS:
        if not np.array_equal(mask.coords[dim].values, reference.coords[dim].values):
            raise ValueError(f"{label} coordinate {dim!r} does not match reference")
    values = np.asarray(mask.values)
    if np.issubdtype(values.dtype, np.bool_):
        return mask.astype(bool)
    if np.issubdtype(values.dtype, np.integer) and np.all(
        (values == 0) | (values == 1)
    ):
        return mask.astype(bool)
    raise ValueError(f"{label} must be Boolean or an integer 0/1 mask")


def _scene_spatial_reference(scene: xr.Dataset) -> xr.DataArray:
    if "reflectance" not in scene:
        raise ValueError("scene is missing required variable 'reflectance'")
    reflectance = scene["reflectance"]
    if "band" not in reflectance.dims:
        raise ValueError("scene reflectance is missing the 'band' dimension")
    return reflectance.isel(band=0, drop=True).transpose(*contract.SPATIAL_DIMS)


def _external_mask_reason(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith("mask_"):
        normalized = normalized[5:]
    aliases = {
        "bad_geometry": "invalid_geometry",
        "low_observation_support": "insufficient_observations",
        "bad_modland_qa": "poor_surface_reflectance_quality",
        "external_inversion": "user_exclusion",
        "external": "user_exclusion",
        "user": "user_exclusion",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in contract.INVERSION_EXCLUSION_REASONS:
        return normalized
    return "user_exclusion"


def _open_mask(path: Path, *, variable: str | None) -> xr.DataArray:
    suffix = path.suffix.lower()
    if suffix in ZARR_SUFFIXES:
        dataset = xr.open_zarr(path)
        return _mask_variable_from_dataset(dataset, variable=variable)
    if suffix in XARRAY_SUFFIXES:
        try:
            return xr.open_dataarray(path)
        except ValueError:
            dataset = xr.open_dataset(path)
            return _mask_variable_from_dataset(dataset, variable=variable)
    if suffix in RASTER_SUFFIXES:
        return rioxarray.open_rasterio(path, masked=True)

    raise ValueError(
        "external masks must be xarray products (.nc, .cdf, .netcdf, .zarr) "
        "or GeoTIFF rasters (.tif, .tiff)"
    )


def _mask_variable_from_dataset(
    dataset: xr.Dataset,
    *,
    variable: str | None,
) -> xr.DataArray:
    if variable is not None:
        if variable not in dataset:
            raise ValueError(f"mask variable {variable!r} is not present in dataset")
        return dataset[variable]

    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return dataset[data_vars[0]]

    raise ValueError("mask datasets with multiple variables require a variable name")


def _normalize_mask_dims(mask: xr.DataArray) -> xr.DataArray:
    rename: dict[str, str] = {}
    if "latitude" in mask.dims:
        rename["latitude"] = "y"
    if "longitude" in mask.dims:
        rename["longitude"] = "x"
    if rename:
        mask = mask.rename(rename)

    if set(mask.dims) == {"band", "y", "x"}:
        if mask.sizes["band"] != 1:
            raise ValueError("external mask rasters must have exactly one band")
        mask = mask.isel(band=0, drop=True)

    if set(mask.dims) != {"y", "x"}:
        raise ValueError("external masks must have y and x dimensions")

    return mask.transpose("y", "x")


def _align_mask_to_scene(mask: xr.DataArray, target_scene: xr.Dataset) -> xr.DataArray:
    target = _scene_spatial_reference(target_scene)
    if _coords_match(mask, target, ("y", "x")):
        return mask

    reprojected = _try_reproject_match(mask, target)
    if reprojected is not None and _coords_match(reprojected, target, ("y", "x")):
        return reprojected

    if (
        mask.sizes["y"] == target.sizes["y"]
        and mask.sizes["x"] == target.sizes["x"]
    ):
        return mask.assign_coords(
            y=target.coords["y"].values,
            x=target.coords["x"].values,
        )

    raise ValueError("external mask cannot be aligned to the target scene grid")


def _try_reproject_match(mask: xr.DataArray, target: xr.DataArray) -> xr.DataArray | None:
    try:
        reprojected = mask.rio.write_crs(mask.rio.crs, inplace=False)
        reprojected = reprojected.rio.reproject_match(target)
        return _normalize_mask_dims(reprojected)
    except Exception:
        return None


def _coords_match(
    left: xr.DataArray,
    right: xr.DataArray,
    dims: tuple[str, ...],
) -> bool:
    for dim in dims:
        if dim not in left.coords or dim not in right.coords:
            return False
        if not np.array_equal(left.coords[dim].values, right.coords[dim].values):
            return False
    return True
