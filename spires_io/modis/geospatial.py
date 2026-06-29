"""Geospatial metadata helpers for MODIS HDF-EOS grids."""

from dataclasses import dataclass
import re
from typing import Any

from affine import Affine
import numpy as np
from pyproj import CRS
import rioxarray  # noqa: F401  # register xarray .rio accessor
import xarray as xr


MODIS_SINUSOIDAL_RADIUS = 6371007.181
MODIS_SINUSOIDAL_CRS = CRS.from_proj4(f"+proj=sinu +R={MODIS_SINUSOIDAL_RADIUS} +units=m +no_defs")


@dataclass(frozen=True)
class ModisGridMetadata:
    """Georeferencing metadata parsed from a MODIS HDF-EOS grid."""

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
        return Affine(self.x_resolution, 0.0, self.upper_left[0], 0.0, -self.y_resolution, self.upper_left[1])

    def x_coords(self) -> np.ndarray:
        return self.upper_left[0] + (np.arange(self.x_size, dtype=np.float64) + 0.5) * self.x_resolution

    def y_coords(self) -> np.ndarray:
        return self.upper_left[1] - (np.arange(self.y_size, dtype=np.float64) + 0.5) * self.y_resolution

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
            "GeoTransform": _format_geotransform(self.transform),
        }


def _find_grid_block(struct_metadata: str, grid_name: str) -> str | None:
    pattern = re.compile(
        r"GROUP=GRID_\d+\s+"
        rf'GridName="{re.escape(grid_name)}"'
        r"(?P<body>.*?)"
        r"\n\tEND_GROUP=GRID_\d+",
        re.DOTALL,
    )
    match = pattern.search(struct_metadata)
    if match is None:
        return None
    return match.group(0)


def _parse_pair(text: str, key: str) -> tuple[float, float]:
    match = re.search(rf"{re.escape(key)}=\(([^,]+),([^)]+)\)", text)
    if match is None:
        raise ValueError(f"Could not parse {key!r} from MODIS StructMetadata.0")
    return float(match.group(1)), float(match.group(2))


def _parse_scalar(text: str, key: str) -> str | None:
    match = re.search(rf"{re.escape(key)}=([^\n\r]+)", text)
    if match is None:
        return None
    return match.group(1).strip().strip('"')


def _parse_proj_params(text: str) -> tuple[float, ...]:
    match = re.search(r"ProjParams=\(([^)]+)\)", text)
    if match is None:
        return ()
    return tuple(float(item.strip()) for item in match.group(1).split(","))


def _format_geotransform(transform: Affine) -> str:
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


def _spatial_ref_attrs(crs: CRS, transform: Affine | None = None) -> dict[str, Any]:
    attrs = dict(crs.to_cf())
    wkt = crs.to_wkt()
    attrs["spatial_ref"] = wkt
    attrs["crs_wkt"] = wkt
    if transform is not None:
        attrs["GeoTransform"] = _format_geotransform(transform)
    return attrs


def parse_modis_grid_metadata(dataset: Any, grid_name: str) -> ModisGridMetadata | None:
    """Parse HDF-EOS georeferencing metadata for one MODIS grid."""
    struct_metadata = getattr(dataset, "StructMetadata.0", None)
    if struct_metadata is None:
        return None

    grid_block = _find_grid_block(struct_metadata, grid_name)
    if grid_block is None:
        return None

    x_size = _parse_scalar(grid_block, "XDim")
    y_size = _parse_scalar(grid_block, "YDim")
    projection = _parse_scalar(grid_block, "Projection") or ""
    sphere_code = _parse_scalar(grid_block, "SphereCode")

    return ModisGridMetadata(
        grid_name=grid_name,
        x_size=int(x_size) if x_size is not None else 0,
        y_size=int(y_size) if y_size is not None else 0,
        upper_left=_parse_pair(grid_block, "UpperLeftPointMtrs"),
        lower_right=_parse_pair(grid_block, "LowerRightMtrs"),
        projection=projection,
        proj_params=_parse_proj_params(grid_block),
        sphere_code=int(sphere_code) if sphere_code is not None else None,
        grid_origin=_parse_scalar(grid_block, "GridOrigin") or "",
    )


def attach_spatial_ref(
    ds: xr.Dataset,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    grid_metadata: ModisGridMetadata | None = None,
    crs: CRS = MODIS_SINUSOIDAL_CRS,
    data_var_names: list[str] | tuple[str, ...] | None = None,
) -> xr.Dataset:
    """Attach CF/rioxarray-compatible spatial metadata without touching array values."""
    result = ds.copy()
    transform = grid_metadata.transform if grid_metadata is not None else None

    spatial_ref_attrs = _spatial_ref_attrs(crs, transform)
    if grid_metadata is None and "spatial_ref" in result.coords:
        spatial_ref_attrs.update(result["spatial_ref"].attrs)
    elif grid_metadata is None and "spatial_ref" in result:
        spatial_ref_attrs.update(result["spatial_ref"].attrs)
    result = result.assign_coords(spatial_ref=xr.DataArray(0, attrs=spatial_ref_attrs))
    result.attrs["crs_wkt"] = crs.to_wkt()
    if grid_metadata is not None:
        result.attrs["geospatial_grid"] = grid_metadata.grid_name
        result.attrs["geospatial_transform"] = _format_geotransform(grid_metadata.transform)
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

    for key in (
        "crs_wkt",
        "geospatial_grid",
        "geospatial_transform",
        "geospatial_upper_left_meters",
        "geospatial_lower_right_meters",
    ):
        if key in source.attrs:
            result.attrs[key] = source.attrs[key]

    for name, variable in result.data_vars.items():
        if name != "spatial_ref" and "x" in variable.dims and "y" in variable.dims and "spatial_ref" in result:
            result[name].attrs.setdefault("grid_mapping", "spatial_ref")
            result[name].encoding.pop("grid_mapping", None)

    return result
