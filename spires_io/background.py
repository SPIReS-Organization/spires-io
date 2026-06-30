"""Background reflectance readers for SPIReS inversion inputs."""

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  # register xarray .rio accessor
import xarray as xr

from spires_io.file_types import RASTER_SUFFIXES, XARRAY_SUFFIXES, ZARR_SUFFIXES

try:
    from spires_contract.spectra import validate_background_spectra
except ImportError:  # pragma: no cover - contract package is optional at runtime.
    validate_background_spectra = None


BACKGROUND_VARIABLE_CANDIDATES = (
    "background_reflectance",
    "r0_reflectance",
    "reflectance",
)


def load_background_reflectance(
    path: str | Path,
    *,
    target_scene: xr.Dataset | None = None,
) -> xr.DataArray:
    """Load background reflectance as canonical ``(y, x, band)`` spectra.

    NetCDF/Zarr-style xarray products are the preferred path for derived
    MODIS/VIIRS backgrounds produced by ``spires-r0``. GeoTIFF remains a
    supported single-scene background format for sensors such as EMIT and
    Sentinel-2.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in ZARR_SUFFIXES:
        background = _open_zarr_background(path)
    elif suffix in XARRAY_SUFFIXES:
        background = _open_netcdf_background(path)
    elif suffix in RASTER_SUFFIXES:
        background = _open_raster_background(path)
    else:
        raise ValueError(
            "background_image must be an xarray product (.nc, .cdf, .netcdf, .zarr) "
            "or a GeoTIFF (.tif, .tiff)"
        )

    background = _canonicalize_background(background, target_scene=target_scene)
    background = background.astype("float64")
    background.name = "background_reflectance"
    _validate_background_contract(background)
    return background


def _open_netcdf_background(path: Path) -> xr.DataArray:
    try:
        return xr.open_dataarray(path)
    except ValueError:
        dataset = xr.open_dataset(path)
        return _background_variable_from_dataset(dataset)


def _open_zarr_background(path: Path) -> xr.DataArray:
    dataset = xr.open_zarr(path)
    return _background_variable_from_dataset(dataset)


def _background_variable_from_dataset(dataset: xr.Dataset) -> xr.DataArray:
    for variable_name in BACKGROUND_VARIABLE_CANDIDATES:
        if variable_name in dataset:
            return dataset[variable_name]

    data_vars = list(dataset.data_vars)
    if len(data_vars) == 1:
        return dataset[data_vars[0]]

    raise ValueError(
        "background xarray datasets must contain one of "
        f"{BACKGROUND_VARIABLE_CANDIDATES!r} or exactly one data variable"
    )


def _open_raster_background(path: Path) -> xr.DataArray:
    return rioxarray.open_rasterio(path, masked=True)


def _canonicalize_background(
    background: xr.DataArray,
    *,
    target_scene: xr.Dataset | None,
) -> xr.DataArray:
    background = _normalize_background_dims(background)

    if target_scene is not None:
        background = _align_background_to_scene(background, target_scene)
        background = _assign_target_band_coord(background, target_scene)

    if background.dims != ("y", "x", "band"):
        raise ValueError("background reflectance must have dims ('y', 'x', 'band')")
    if "band" not in background.coords:
        raise ValueError("background reflectance must include a 'band' coordinate")

    return background


def _normalize_background_dims(background: xr.DataArray) -> xr.DataArray:
    rename: dict[str, str] = {}
    if "latitude" in background.dims:
        rename["latitude"] = "y"
    if "longitude" in background.dims:
        rename["longitude"] = "x"
    if "bands" in background.dims:
        rename["bands"] = "band"
    if rename:
        background = background.rename(rename)

    if background.dims == ("y", "x"):
        background = background.expand_dims(band=[1]).transpose("y", "x", "band")
    elif set(background.dims) == {"y", "x", "band"}:
        background = background.transpose("y", "x", "band")
    else:
        raise ValueError("background reflectance must have y, x, and band dimensions")

    return background


def _align_background_to_scene(
    background: xr.DataArray,
    target_scene: xr.Dataset,
) -> xr.DataArray:
    target = target_scene["reflectance"]
    if _coords_match(background, target, ("y", "x")):
        return background

    reprojected = _try_reproject_match(background, target)
    if reprojected is not None and _coords_match(reprojected, target, ("y", "x")):
        return reprojected

    if (
        background.sizes["y"] == target.sizes["y"]
        and background.sizes["x"] == target.sizes["x"]
    ):
        # Some single-scene rasters arrive without durable CRS metadata in tests
        # or intermediate workflows. If shape matches but geospatial reprojection
        # is unavailable, use the prepared scene as the coordinate authority.
        return background.assign_coords(
            y=target.coords["y"].values,
            x=target.coords["x"].values,
        )

    raise ValueError("background reflectance cannot be aligned to the target scene grid")


def _try_reproject_match(
    background: xr.DataArray,
    target: xr.DataArray,
) -> xr.DataArray | None:
    try:
        target_grid = target.isel(band=0, drop=True)
        reprojected = background.rio.write_crs(background.rio.crs, inplace=False)
        reprojected = reprojected.rio.reproject_match(target_grid)
        return _normalize_background_dims(reprojected)
    except Exception:
        return None


def _assign_target_band_coord(
    background: xr.DataArray,
    target_scene: xr.Dataset,
) -> xr.DataArray:
    target_band = target_scene["reflectance"].coords["band"]
    if background.sizes["band"] != target_band.size:
        raise ValueError(
            "background reflectance band count does not match the target scene "
            f"({background.sizes['band']} != {target_band.size})"
        )
    return background.assign_coords(band=target_band.values)


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


def _validate_background_contract(background: xr.DataArray) -> None:
    if validate_background_spectra is not None:
        validate_background_spectra(background)
