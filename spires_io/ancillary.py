"""Ancillary layer readers for SPIReS data loading."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401  # register xarray .rio accessor
import xarray as xr

import spires_io.constants as constants
from spires_io.file_types import RASTER_SUFFIXES, XARRAY_SUFFIXES, ZARR_SUFFIXES


SUPPORTED_ANCILLARY_NAMES = tuple(constants.STATIC_DATA)


def load_ancillary_layers(
    sources: Mapping[str, Any],
    *,
    target_scene: xr.Dataset,
) -> xr.Dataset | None:
    """Load ancillary layers aligned to a prepared scene.

    ``sources`` maps public layer names such as ``"canopy_fraction"`` or
    ``"dem"`` to paths, DataArrays, or mapping specs containing ``path`` and an
    optional ``variable``/``var`` key.
    """
    if not sources:
        return None

    data_vars = {}
    for name, spec in sources.items():
        if spec is None:
            continue
        data_vars[name] = load_ancillary_layer(
            spec,
            name=name,
            target_scene=target_scene,
        )

    if not data_vars:
        return None
    return xr.Dataset(data_vars)


def load_ancillary_layer(
    source: str | Path | xr.DataArray | Mapping[str, Any],
    *,
    name: str,
    target_scene: xr.Dataset,
) -> xr.DataArray:
    """Load one ancillary layer as a ``(y, x)`` DataArray."""
    variable = None
    if isinstance(source, Mapping):
        if "path" in source:
            variable = source.get("variable") or source.get("var")
            source = source["path"]
        elif "value" in source:
            source = source["value"]
        else:
            raise ValueError(f"ancillary layer {name!r} must include a 'path' or 'value'")

    if isinstance(source, xr.DataArray):
        layer = source
    else:
        layer = _open_ancillary(Path(source), variable=variable, fallback_name=name)

    layer = _normalize_layer_dims(layer, name=name)
    layer = _align_layer_to_scene(layer, target_scene, name=name)
    layer = _clean_layer_values(layer, name=name)
    layer.name = name
    return layer


def _open_ancillary(
    path: Path,
    *,
    variable: str | None,
    fallback_name: str,
) -> xr.DataArray:
    suffix = path.suffix.lower()
    if suffix in ZARR_SUFFIXES:
        dataset = xr.open_zarr(path)
        return _ancillary_variable_from_dataset(
            dataset,
            variable=variable,
            fallback_name=fallback_name,
        )
    if suffix in XARRAY_SUFFIXES:
        try:
            return xr.open_dataarray(path)
        except ValueError:
            dataset = xr.open_dataset(path)
            return _ancillary_variable_from_dataset(
                dataset,
                variable=variable,
                fallback_name=fallback_name,
            )
    if suffix in RASTER_SUFFIXES:
        return rioxarray.open_rasterio(path, masked=True)

    raise ValueError(
        "ancillary layers must be xarray products (.nc, .cdf, .netcdf, .zarr) "
        "or GeoTIFF rasters (.tif, .tiff)"
    )


def _ancillary_variable_from_dataset(
    dataset: xr.Dataset,
    *,
    variable: str | None,
    fallback_name: str,
) -> xr.DataArray:
    if variable is not None:
        if variable not in dataset:
            raise ValueError(f"ancillary variable {variable!r} is not present in dataset")
        return dataset[variable]

    if fallback_name in dataset:
        return dataset[fallback_name]

    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return dataset[data_vars[0]]

    raise ValueError(
        f"ancillary dataset for {fallback_name!r} has multiple variables; "
        "provide a variable name"
    )


def _normalize_layer_dims(layer: xr.DataArray, *, name: str) -> xr.DataArray:
    rename: dict[str, str] = {}
    if "latitude" in layer.dims:
        rename["latitude"] = "y"
    if "longitude" in layer.dims:
        rename["longitude"] = "x"
    if rename:
        layer = layer.rename(rename)

    if set(layer.dims) == {"band", "y", "x"}:
        if layer.sizes["band"] != 1:
            raise ValueError(f"ancillary layer {name!r} must have exactly one band")
        layer = layer.isel(band=0, drop=True)

    if set(layer.dims) != {"y", "x"}:
        raise ValueError(f"ancillary layer {name!r} must have y and x dimensions")

    return layer.transpose("y", "x")


def _align_layer_to_scene(
    layer: xr.DataArray,
    target_scene: xr.Dataset,
    *,
    name: str,
) -> xr.DataArray:
    target = target_scene["valid_inversion_mask"]
    if _coords_match(layer, target, ("y", "x")):
        return layer

    reprojected = _try_reproject_aspect_match(layer, target) if name == "aspect" else None
    if reprojected is None:
        reprojected = _try_reproject_match(layer, target)
    if reprojected is not None and _coords_match(reprojected, target, ("y", "x")):
        return reprojected

    if layer.sizes["y"] == target.sizes["y"] and layer.sizes["x"] == target.sizes["x"]:
        return layer.assign_coords(
            y=target.coords["y"].values,
            x=target.coords["x"].values,
        )

    raise ValueError(f"ancillary layer {name!r} cannot be aligned to the target scene grid")


def _try_reproject_match(
    layer: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray | None:
    try:
        reprojected = layer.rio.write_crs(layer.rio.crs, inplace=False)
        reprojected = reprojected.rio.reproject_match(target)
        return _normalize_layer_dims(reprojected, name=layer.name or "ancillary")
    except Exception:
        return None


def _try_reproject_aspect_match(
    layer: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray | None:
    try:
        radians = np.deg2rad(layer)
        sin_aspect = np.sin(radians)
        cos_aspect = np.cos(radians)
        sin_warped = _try_reproject_match(sin_aspect, target)
        cos_warped = _try_reproject_match(cos_aspect, target)
        if sin_warped is None or cos_warped is None:
            return None
        aspect = (np.rad2deg(np.arctan2(sin_warped, cos_warped)) % 360.0)
        aspect.name = "aspect"
        return aspect
    except Exception:
        return None


def _clean_layer_values(layer: xr.DataArray, *, name: str) -> xr.DataArray:
    layer = layer.astype("float32")

    if name in {"canopy_fraction", "ice_fraction"}:
        if bool((layer > 1.0 + constants.EPS).fillna(False).any()):
            layer = layer / 100.0
        layer = layer.fillna(0.0)
        layer = layer.where(
            (layer >= 0.0 - constants.EPS) & (layer <= 1.0 + constants.EPS)
        )
        layer = layer.clip(min=0.0, max=1.0)
    elif name == "slope":
        layer = layer.where((layer >= 0.0) & (layer <= constants.MAX_ZENITH))
    elif name == "aspect":
        layer = layer.where((layer >= 0.0) & (layer <= constants.MAX_ASPECT))
    elif name == "dem":
        layer = layer.where((layer >= constants.MIN_ELEV) & (layer <= constants.MAX_ELEV))
    elif name == "skyview":
        layer = layer.where((layer >= 0.0 + constants.EPS) & (layer <= constants.MAX_SVF))

    return layer


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
