"""Shared geospatial metadata helpers for sinusoidal MODIS/VIIRS grids."""

from dataclasses import dataclass
from typing import Any

from affine import Affine
import numpy as np
from pyproj import CRS
import rioxarray  # noqa: F401  # register xarray .rio accessor
import xarray as xr


SINUSOIDAL_RADIUS = 6371007.181
SINUSOIDAL_CRS = CRS.from_proj4(f"+proj=sinu +R={SINUSOIDAL_RADIUS} +units=m +no_defs")


@dataclass(frozen=True)
class HdfEosGridMetadata:
    """Georeferencing metadata parsed from an HDF-EOS grid."""

    grid_name: str
    x_size: int
    y_size: int
    upper_left: tuple[float, float]
    lower_right: tuple[float, float]
    projection: str
    proj_params: tuple[float, ...]
    sphere_code: int | None
    grid_origin: str

    @property
    def x_resolution(self) -> float:
        return (self.lower_right[0] - self.upper_left[0]) / self.x_size

    @property
    def y_resolution(self) -> float:
        return (self.upper_left[1] - self.lower_right[1]) / self.y_size

    @property
    def transform(self) -> Affine:
        return Affine(
            self.x_resolution,
            0.0,
            self.upper_left[0],
            0.0,
            -self.y_resolution,
            self.upper_left[1],
        )

    def x_coords(self) -> np.ndarray:
        return self.upper_left[0] + (
            np.arange(self.x_size, dtype=np.float64) + 0.5
        ) * self.x_resolution

    def y_coords(self) -> np.ndarray:
        return self.upper_left[1] - (
            np.arange(self.y_size, dtype=np.float64) + 0.5
        ) * self.y_resolution

    def to_attrs(self) -> dict[str, Any]:
        return {
            "grid_name": self.grid_name,
            "upper_left_point_meters": self.upper_left,
            "lower_right_meters": self.lower_right,
            "projection": self.projection,
            "proj_params": self.proj_params,
            "sphere_code": self.sphere_code,
            "grid_origin": self.grid_origin,
            "x_resolution": self.x_resolution,
            "y_resolution": self.y_resolution,
            "GeoTransform": format_geotransform(self.transform),
        }


def format_geotransform(transform: Affine) -> str:
    """Return a GDAL-style GeoTransform attribute string."""
    return " ".join(
        f"{value:.12g}"
        for value in (
            transform.c,
            transform.a,
            transform.b,
            transform.f,
            transform.d,
            transform.e,
        )
    )


def spatial_ref_attrs(crs: CRS, transform: Affine | None = None) -> dict[str, Any]:
    """Build CF/rioxarray spatial reference attributes."""
    attrs = dict(crs.to_cf())
    wkt = crs.to_wkt()
    attrs["spatial_ref"] = wkt
    attrs["crs_wkt"] = wkt
    if transform is not None:
        attrs["GeoTransform"] = format_geotransform(transform)
    return attrs


def attach_spatial_ref(
    ds: xr.Dataset,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    grid_metadata: HdfEosGridMetadata | None = None,
    crs: CRS = SINUSOIDAL_CRS,
    data_var_names: list[str] | tuple[str, ...] | None = None,
) -> xr.Dataset:
    """Attach CF/rioxarray-compatible spatial metadata without touching array values."""
    result = ds.copy()
    transform = grid_metadata.transform if grid_metadata is not None else None

    attrs = spatial_ref_attrs(crs, transform)
    if grid_metadata is None and "spatial_ref" in result.coords:
        attrs.update(result["spatial_ref"].attrs)
    elif grid_metadata is None and "spatial_ref" in result:
        attrs.update(result["spatial_ref"].attrs)
    result = result.assign_coords(spatial_ref=xr.DataArray(0, attrs=attrs))
    result.attrs["crs_wkt"] = crs.to_wkt()
    if grid_metadata is not None:
        result.attrs["geospatial_grid"] = grid_metadata.grid_name
        result.attrs["geospatial_transform"] = format_geotransform(
            grid_metadata.transform
        )
        result.attrs["geospatial_upper_left_meters"] = grid_metadata.upper_left
        result.attrs["geospatial_lower_right_meters"] = grid_metadata.lower_right

    if x_dim in result.coords:
        result[x_dim].attrs.update(
            {
                "axis": "X",
                "long_name": "x coordinate of projection",
                "standard_name": "projection_x_coordinate",
                "units": "m",
            }
        )
    if y_dim in result.coords:
        result[y_dim].attrs.update(
            {
                "axis": "Y",
                "long_name": "y coordinate of projection",
                "standard_name": "projection_y_coordinate",
                "units": "m",
            }
        )

    try:
        result = result.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
        result = result.rio.write_crs(crs, inplace=False)
        if transform is not None:
            result = result.rio.write_transform(transform, inplace=False)
    except Exception:
        pass

    if data_var_names is None:
        data_var_names = [
            name
            for name, variable in result.data_vars.items()
            if name != "spatial_ref" and x_dim in variable.dims and y_dim in variable.dims
        ]
    for name in data_var_names:
        if name in result and x_dim in result[name].dims and y_dim in result[name].dims:
            result[name].attrs["grid_mapping"] = "spatial_ref"
            result[name].encoding.pop("grid_mapping", None)

    return result


def copy_spatial_metadata(source: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Copy spatial coordinates, CRS variable, and grid_mapping attrs to a derived dataset."""
    result = target.copy()

    if "spatial_ref" in source:
        result = result.assign_coords(spatial_ref=source["spatial_ref"].copy())

    for coord_name in ("x", "y", "x_500m", "y_500m", "x_1km", "y_1km"):
        if coord_name in source.coords and coord_name in result.coords:
            result[coord_name].attrs.update(source[coord_name].attrs)

    for name, variable in result.data_vars.items():
        if "spatial_ref" in result.coords and any(
            dim in variable.dims for dim in ("x", "x_500m", "x_1km")
        ):
            result[name].attrs["grid_mapping"] = "spatial_ref"

    if "crs_wkt" in source.attrs:
        result.attrs["crs_wkt"] = source.attrs["crs_wkt"]

    return result


VIIRS_MODIS_WATER_MASK_INTS = (0, 2, 3, 4, 5, 6, 7)
