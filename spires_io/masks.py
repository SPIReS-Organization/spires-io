"""External mask readers for SPIReS data loading."""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  # register xarray .rio accessor
import xarray as xr

from spires_io.file_types import RASTER_SUFFIXES, XARRAY_SUFFIXES, ZARR_SUFFIXES


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
    target = target_scene["valid_inversion_mask"]
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
