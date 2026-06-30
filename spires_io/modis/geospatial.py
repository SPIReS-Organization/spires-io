"""MODIS HDF-EOS grid metadata adapters."""

import re
from typing import Any

from spires_io.geospatial import (
    HdfEosGridMetadata,
    SINUSOIDAL_CRS,
    attach_spatial_ref,
    copy_spatial_metadata,
)


MODIS_SINUSOIDAL_CRS = SINUSOIDAL_CRS
ModisGridMetadata = HdfEosGridMetadata


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
